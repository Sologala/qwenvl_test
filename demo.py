
import os
import json
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
# 加载模型（保持不变）
# -----------------------------

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "ikkk222/qwen3-vl-8b-instruct-abliterated-CV",
    trust_remote_code=True,
    device_map="auto",
    dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(
    "ikkk222/qwen3-vl-8b-instruct-abliterated-CV",
    trust_remote_code=True
)


# -----------------------------
# 路径配置
# -----------------------------

image_dir = "./test_boat_images"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# 主推理循环
# -----------------------------

files = [
    f for f in os.listdir(image_dir)
    if is_image_file(f)
]

for fname in tqdm(files, desc="Processing images"):
    img_path = os.path.join(image_dir, fname)

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {fname}: {e}")
        continue

    prompt = {
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

    text = processor.apply_chat_template(
        [prompt],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device, torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    output_text = processor.batch_decode(
        outputs,
        skip_special_tokens=True
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

