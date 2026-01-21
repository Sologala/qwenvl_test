
import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


# -----------------------------
# 工具函数
# -----------------------------

def extract_json(text: str):
    """
    尝试从模型输出中提取 JSON
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def visualize_result(image, save_path, json_text):
    """
    将 JSON 结果绘制到图片左上角
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        font = ImageFont.load_default()

    draw.multiline_text(
        (10, 10),
        json_text,
        fill=(0, 0, 0),
        font=font,
        spacing=4
    )
    img.save(save_path)


def is_image_file(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


# -----------------------------
# 参数解析
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="船舶识别系统 - 使用 Qwen3-VL 4bit 模型")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./test_boat_images",
        help="输入图片目录 (默认: ./test_boat_images)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录 (默认: ./output)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="DavidWen2025/Qwen3-VL-8B-Instruct-4bit-GPTQ",
        help="4bit GPTQ 模型路径 (默认: DavidWen2025/Qwen3-VL-8B-Instruct-4bit-GPTQ)"
    )
    return parser.parse_args()


# -----------------------------
# 加载模型（4bit GPTQ 版本）
# -----------------------------

def load_model(model_path):
    """
    加载 4bit GPTQ 量化的 Qwen3-VL 模型
    """
    print(f"正在加载模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("模型加载完成!")
    return model, processor


# -----------------------------
# 主函数
# -----------------------------

def main():
    args = parse_args()
    
    # 路径配置（argparse 会将 '-' 转换为 '_'）
    image_dir = args.input_dir
    output_dir = args.output_dir
    model_path = args.model_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入目录是否存在
    if not os.path.isdir(image_dir):
        print(f"[错误] 输入目录不存在: {image_dir}")
        return
    
    # 加载模型
    model, processor = load_model(model_path)
    
    # 获取图片文件列表
    files = [
        f for f in os.listdir(image_dir)
        if is_image_file(f)
    ]
    
    if not files:
        print(f"[警告] 在 {image_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(files)} 张图片，开始处理...")
    
    # -----------------------------
    # 主推理循环
    # -----------------------------
    
    for fname in tqdm(files, desc="Processing images"):
        img_path = os.path.join(image_dir, fname)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {fname}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "You are a maritime recognition system.\n"
                            "Given the image of a ship, output JSON with fields:\n"
                            "ship_type, estimated_length_range_m, estimated_length_range_confidence, estimated_length_m, length_conf .\n"
                            "Return JSON only."
                        )
                    }
                ]
            }
        ]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        json_obj = extract_json(output_text)

        base_name = os.path.splitext(fname)[0]

        # -------------------------
        # 保存 JSON（如果合法）
        # -------------------------
        if json_obj is not None:
            json_path = os.path.join(output_dir, base_name + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)
        else:
            print(f"[WARN] Invalid JSON output for {fname}")

        # -------------------------
        # 保存可视化图片
        # -------------------------
        vis_img_path = os.path.join(output_dir, fname)
        visualize_result(
            image,
            vis_img_path,
            json.dumps(json_obj, ensure_ascii=False, indent=2)
            if json_obj else output_text
        )


if __name__ == "__main__":
    main()
