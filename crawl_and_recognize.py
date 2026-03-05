#!/usr/bin/env python3
"""
从网络爬取船只图片，使用 Qwen3-VL 进行识别，并保存为 JSON。
支持本地加载模型（modelscope）或调用已运行的 llama.cpp 服务（--api-base）。
支持 --build-dataset：按多种船只类型关键词爬取，用 Qwen3-VL 筛选「仅一艘船、未遮挡」，
  resize 后保存为分类数据集（按船型分目录 + 含尺寸预测的 metadata）。
"""

import os
import re
import json
import argparse
import time
import hashlib
import base64
import shutil
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import requests
import random

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

# -----------------------------
# 船只类型关键词（用于爬取与分类）
# -----------------------------
SHIP_TYPE_KEYWORDS = [
    "cargo ship",
    "container ship",
    "bulk carrier",
    "oil tanker",
    "LNG carrier",
    "chemical tanker",
    "cruise ship",
    "passenger ferry",
    "car ferry",
    "fishing boat",
    "fishing trawler",
    "sailboat sailing",
    "yacht",
    "tugboat",
    "warship navy",
    "naval destroyer",
    "naval frigate",
    "aircraft carrier",
    "patrol boat",
    "speedboat",
    "barge",
    "research vessel",
    "icebreaker ship",
    "catamaran",
    "lifeboat rescue",
    "houseboat",
    "freighter",
    "general cargo vessel",
]

# -----------------------------
# 工具函数
# -----------------------------

def extract_json(text: str):
    """从模型输出中提取 JSON"""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def is_image_file(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def resize_to_height(image: Image.Image, target_height: int = 512) -> Image.Image:
    """
    等比例缩放到目标高度 target_height；若当前高度已 < target_height 则不做放大，直接返回。
    """
    h, w = image.height, image.width
    if h <= target_height:
        return image
    new_w = int(round(w * target_height / h))
    new_h = target_height
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def download_image(url: str, timeout: int = 15) -> bytes | None:
    """下载图片，返回字节内容，失败返回 None"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def save_image_from_bytes(data: bytes, out_dir: str, index: int, url: str, prefix: str = "ship") -> str | None:
    """将字节保存为图片文件，返回保存路径或 None。url 用于生成 hash，无 URL 时可传任意唯一字符串。"""
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None
    name_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    ext = ".jpg"
    fname = f"{prefix}_{index:03d}_{name_hash}{ext}"
    path = os.path.join(out_dir, fname)
    img.save(path)
    return path


def crawl_ship_images_duckduckgo(keywords: str, max_results: int, save_dir: str) -> list[dict]:
    """使用 DuckDuckGo 搜索并下载图片。"""
    os.makedirs(save_dir, exist_ok=True)
    ddgs = DDGS()
    for attempt in range(3):
        try:
            results = list(ddgs.images(keywords=keywords, max_results=max_results))
            break
        except RatelimitException:
            wait_sec = 60 * (attempt + 1)
            print(f"[限流] DuckDuckGo 返回 403，{wait_sec} 秒后重试 ({attempt + 1}/3)...")
            time.sleep(wait_sec)
    else:
        return []
    downloaded = []
    for i, item in enumerate(tqdm(results, desc="DuckDuckGo 下载", leave=False)):
        url = item.get("image") or item.get("url")
        if not url:
            continue
        data = download_image(url)
        if not data:
            continue
        path = save_image_from_bytes(data, save_dir, i, url, prefix="ddg")
        if path:
            downloaded.append({"url": url, "local_path": path, "source": "duckduckgo"})
        time.sleep(random.uniform(0.5, 2))
    return downloaded


def crawl_ship_images_bing(keywords: str, max_results: int, save_dir: str) -> list[dict]:
    """使用 Bing 搜索并下载图片（依赖 bing-image-downloader）。"""
    try:
        from bing_image_downloader import downloader
    except ImportError:
        print("[跳过] Bing 爬取需要安装: pip install bing-image-downloader")
        return []
    os.makedirs(save_dir, exist_ok=True)
    # 库会创建 save_dir/<query>/ 子目录
    try:
        downloader.download(
            keywords,
            limit=max_results,
            output_dir=save_dir,
            adult_filter_off=False,
            force_replace=False,
            timeout=15,
            verbose=False,
        )
    except Exception as e:
        print(f"[WARN] Bing 下载异常: {e}")
        return []
    downloaded = []
    # 库保存为 save_dir/<sanitized_query>/Image_1.jpg 等
    for root, _, files in os.walk(save_dir):
        for f in sorted(files):
            if is_image_file(f):
                path = os.path.join(root, f)
                downloaded.append({"url": "", "local_path": path, "source": "bing"})
    return downloaded


def crawl_ship_images_baidu(keywords: str, max_results: int, save_dir: str) -> list[dict]:
    """使用 BaiduImageCrawling 库爬取百度图片（见 https://github.com/SWHL/BaiduImageCrawling）。"""
    try:
        from baidu_image_crawling.main import Crawler
    except ImportError:
        print("[跳过] 百度爬取需要安装: pip install baidu_image_crawling")
        return []
    os.makedirs(save_dir, exist_ok=True)
    per_page = 30
    total_page = max(1, (max_results + per_page - 1) // per_page)
    try:
        crawler = Crawler(time_sleep=0.1, save_dir=save_dir)
        crawler(word=keywords, total_page=total_page, start_page=1, per_page=per_page)
    except Exception as e:
        print(f"[WARN] 百度爬取异常: {e}")
        return []
    # 库会保存到 save_dir/<word>/ 下，按文件名排序收集
    word_dir = os.path.join(save_dir, keywords)
    downloaded = []
    if os.path.isdir(word_dir):
        for f in sorted(os.listdir(word_dir)):
            if is_image_file(f):
                path = os.path.join(word_dir, f)
                downloaded.append({"url": "", "local_path": path, "source": "baidu"})
    return downloaded


def crawl_ship_images(keywords: str, max_results: int, save_dir: str, sources: list[str] | None = None) -> list[dict]:
    """
    按指定来源（duckduckgo / bing / baidu）搜索并下载图片到 save_dir。
    返回列表 [{"url": ..., "local_path": ..., "source": ...}, ...]
    每种来源最多下载 max_results 张，总数为各来源之和。
    """
    if sources is None:
        sources = ["duckduckgo"]
    sources = [s.strip().lower() for s in sources if s.strip()]
    if not sources:
        sources = ["duckduckgo"]
    os.makedirs(save_dir, exist_ok=True)
    print(f"正在搜索关键词: {keywords}, 最多每种来源 {max_results} 张, 来源: {sources}")
    all_downloaded = []
    for src in sources:
        if src == "duckduckgo":
            lst = crawl_ship_images_duckduckgo(keywords, max_results, os.path.join(save_dir, "ddg"))
        elif src == "bing":
            lst = crawl_ship_images_bing(keywords, max_results, os.path.join(save_dir, "bing"))
        elif src == "baidu":
            lst = crawl_ship_images_baidu(keywords, max_results, os.path.join(save_dir, "baidu"))
        else:
            print(f"[跳过] 未知来源: {src}，支持 duckduckgo / bing / baidu")
            continue
        all_downloaded.extend(lst)
        if src != sources[-1]:
            time.sleep(random.uniform(2, 4))
    if not all_downloaded and "duckduckgo" in sources:
        raise SystemExit(
            "DuckDuckGo 未获取到图片（可能被限流）。可稍后重试或使用其它来源：\n"
            "  python crawl_and_recognize.py --sources bing,baidu"
        )
    return all_downloaded


# 船舶识别 prompt（本地模型与 API 共用）
SHIP_RECOGNITION_PROMPT = (
    "You are a maritime recognition system.\n"
    "Given the image of a ship, output JSON with fields:\n"
    "ship_type, estimated_length_range_m, estimated_length_range_confidence, estimated_length_m, length_conf.\n"
    "Return JSON only."
)

# 数据集构建：筛选 + 识别（必须包含船、仅一艘、未遮挡，并返回尺寸预测）
SHIP_FILTER_AND_RECOGNITION_PROMPT = (
    "You are a maritime image checker. Look at the image and output a single JSON object only (no other text).\n"
    "Required boolean fields (be strict; when in doubt, use false):\n"
    "  - contains_ship: true ONLY when there is at least one real ship/vessel/boat (watercraft) clearly present in the image; false if no ship, or only land vehicles, buildings, or non-ship objects.\n"
    "  - only_one_ship: true ONLY when there is exactly one ship/vessel in the image; false if zero or more than one.\n"
    "  - not_occluded: true ONLY when that one ship is clearly visible and not blocked by other objects or heavy occlusion.\n"
    "If contains_ship, only_one_ship and not_occluded are all true, also provide:\n"
    "  - ship_type: string, the type of ship (e.g. cargo ship, tanker, yacht). Must be a real ship type, not \"none\" or \"unknown\" or empty.\n"
    "  - estimated_length_m: number or null, estimated length in meters.\n"
    "  - estimated_length_range_m: string, e.g. \"50-80\" or null.\n"
    "  - length_confidence: string, e.g. \"low\", \"medium\", \"high\" or null.\n"
    "Return JSON only."
)


def _image_to_base64_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """PIL Image 转为 data URL (base64)"""
    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def recognize_ship_api(api_base: str, image: Image.Image, timeout: int = 120) -> dict | None:
    """
    调用 llama.cpp 兼容的 /v1/chat/completions 接口做船舶识别。
    api_base 例如 http://127.0.0.1:8080，不要带末尾斜杠。
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    img_data = _image_to_base64_data_url(image)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_data, "detail": "auto"}},
                    {"type": "text", "text": SHIP_RECOGNITION_PROMPT},
                ],
            }
        ],
        "max_tokens": 512,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return extract_json(text)
    except Exception as e:
        print(f"[WARN] API 请求失败: {e}")
        return None


def filter_and_recognize_ship_api(api_base: str, image: Image.Image, timeout: int = 120) -> dict | None:
    """
    调用 API 做「仅一艘船 + 未遮挡」筛选，并返回船型与尺寸预测。
    返回的 JSON 需包含 only_one_ship, not_occluded，以及通过时的 ship_type, estimated_length_m 等。
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    img_data = _image_to_base64_data_url(image)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_data, "detail": "auto"}},
                    {"type": "text", "text": SHIP_FILTER_AND_RECOGNITION_PROMPT},
                ],
            }
        ],
        "max_tokens": 512,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return extract_json(text)
    except Exception as e:
        print(f"[WARN] API 请求失败: {e}")
        return None


def ship_type_to_dirname(ship_type: str) -> str:
    """将模型输出的 ship_type 转为可做目录名的字符串"""
    if not ship_type or not isinstance(ship_type, str):
        return "unknown"
    s = ship_type.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s[:64] if s else "unknown"


def load_model(model_path: str):
    """加载 Qwen3-VL 模型（仅在不使用 --api-base 时调用）"""
    import torch
    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"正在加载模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("模型加载完成!")
    return model, processor


def recognize_ship_local(model, processor, image: Image.Image) -> dict | None:
    """本地模型对单张图片做船舶识别"""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SHIP_RECOGNITION_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extract_json(output_text)


def parse_args():
    parser = argparse.ArgumentParser(description="爬取船只图片并用 Qwen3-VL 识别，保存为 JSON")
    parser.add_argument(
        "--keywords",
        type=str,
        default="ship boat vessel maritime",
        help="搜索关键词 (默认: ship boat vessel maritime)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="每种来源最多爬取图片数量 (默认: 10)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="duckduckgo,bing,baidu",
        help="图片爬取来源，逗号分隔，可选: duckduckgo, bing, baidu (默认: duckduckgo,bing,baidu)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./crawled_ships",
        help="下载图片保存目录 (默认: ./crawled_ships)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./ship_recognition_results.json",
        help="识别结果 JSON 输出路径 (默认: ./ship_recognition_results.json)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="DavidWen2025/Qwen3-VL-8B-Instruct-4bit-GPTQ",
        help="本地模型路径（未指定 --api-base 时使用）",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://127.0.0.1:8080",
        metavar="URL",
        help="llama.cpp 服务地址，例如 http://127.0.0.1:8080；指定后不再加载本地模型",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="不使用本地服务，改为加载 --model-path 本地模型",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="跳过爬取，仅对 images-dir 中已有图片做识别",
    )
    # 构建分类数据集
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="按多种船型关键词爬取，用 Qwen3-VL 筛选「仅一艘、未遮挡」，resize 后保存为分类数据集",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./ship_classification_dataset",
        help="分类数据集根目录（默认: ./ship_classification_dataset）",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        metavar="H",
        help="通过筛选的图片等比例缩放到该高度（高≥此值才缩放，< 则保持原样，默认: 512）",
    )
    parser.add_argument(
        "--max-per-keyword",
        type=int,
        default=20,
        help="每种船型关键词最多爬取图片数（默认: 20，仅 --build-dataset 时有效）",
    )
    return parser.parse_args()


def _build_dataset(args) -> None:
    """按多种船型关键词爬取，Qwen3-VL 筛选「仅一艘、未遮挡」，等比例缩放到指定高后保存为分类数据集。"""
    api_base = args.api_base.rstrip("/")
    dataset_dir = args.dataset_dir
    resize_height = args.resize_height
    max_per_keyword = args.max_per_keyword
    temp_crawl_dir = os.path.join(dataset_dir, ".crawl_temp")
    os.makedirs(temp_crawl_dir, exist_ok=True)

    annotations = []
    global_id = 0

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or ["duckduckgo"]
    for ki, keyword in enumerate(tqdm(SHIP_TYPE_KEYWORDS, desc="船型关键词")):
        try:
            crawled = crawl_ship_images(keyword, max_per_keyword, temp_crawl_dir, sources=sources)
        except SystemExit:
            break
        for item in tqdm(crawled, desc=f"筛选 [{keyword[:20]}]", leave=False):
            path = item["local_path"]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            rec = filter_and_recognize_ship_api(api_base, image)
            if not rec:
                continue
            contains = rec.get("contains_ship") in (True, "true", "yes", "True")
            only_one = rec.get("only_one_ship") in (True, "true", "yes", "True")
            not_occ = rec.get("not_occluded") in (True, "true", "yes", "True")
            if not (contains and only_one and not_occ):
                continue
            ship_type_raw = (rec.get("ship_type") or "").strip()
            # 拒绝没有给出有效船型的（避免无船图被误收）
            if not ship_type_raw or ship_type_raw.lower() in ("none", "no ship", "n/a", "no", "unknown", "no vessel"):
                continue
            ship_type_raw = ship_type_raw or "unknown"
            dirname = ship_type_to_dirname(ship_type_raw)
            global_id += 1
            out_subdir = os.path.join(dataset_dir, dirname)
            os.makedirs(out_subdir, exist_ok=True)
            fname = f"{global_id:06d}.jpg"
            out_path = os.path.join(out_subdir, fname)
            image_resized = resize_to_height(image, target_height=resize_height)
            image_resized.save(out_path, quality=95)
            rel_path = os.path.join(dirname, fname)
            annotations.append({
                "id": global_id,
                "image_path": rel_path,
                "ship_type": ship_type_raw,
                "estimated_length_m": rec.get("estimated_length_m"),
                "estimated_length_range_m": rec.get("estimated_length_range_m"),
                "length_confidence": rec.get("length_confidence"),
                "source_url": item.get("url", ""),
                "search_keyword": keyword,
            })
        time.sleep(random.uniform(2, 5))  # 关键词之间稍作间隔，减轻限流

    # 保存标注
    ann_path = os.path.join(dataset_dir, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    # 数据集说明
    readme_path = os.path.join(dataset_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("船舶分类数据集\n")
        f.write("- 目录结构: <ship_type>/<id>.jpg，可直接用 torchvision.datasets.ImageFolder 加载\n")
        f.write("- annotations.json: 每张图的 ship_type、estimated_length_m、estimated_length_range_m 等\n")
        f.write(f"- 图片等比例缩放到高度 {resize_height}（高 < {resize_height} 则保持原样）\n")
        f.write(f"- 样本数: {len(annotations)}\n")
    # 删除临时爬取目录
    if os.path.isdir(temp_crawl_dir):
        shutil.rmtree(temp_crawl_dir, ignore_errors=True)
    print(f"数据集已写入: {dataset_dir}, 共 {len(annotations)} 条，标注: {ann_path}")


def main():
    args = parse_args()

    if args.build_dataset:
        if args.no_server:
            print("[错误] --build-dataset 当前仅支持 --api-base 本地服务，请勿使用 --no-server")
            return
        print(f"使用本地服务: {args.api_base}")
        _build_dataset(args)
        return

    # 1. 爬取图片（或使用已有目录）
    if args.skip_crawl:
        image_list = []
        if os.path.isdir(args.images_dir):
            for f in sorted(os.listdir(args.images_dir)):
                if is_image_file(f):
                    image_list.append({"url": "", "local_path": os.path.join(args.images_dir, f)})
        if not image_list:
            print(f"[错误] --skip-crawl 时请在 {args.images_dir} 下放置图片")
            return
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()] or ["duckduckgo"]
        image_list = crawl_ship_images(args.keywords, args.max_images, args.images_dir, sources=sources)
        if not image_list:
            print("[警告] 未成功下载任何图片")
            return
    print(f"共 {len(image_list)} 张图片待识别")

    # 2. 选择推理方式：本地服务 vs 本地模型
    use_server = not args.no_server and args.api_base
    if use_server:
        print(f"使用本地服务: {args.api_base}")
    else:
        model, processor = load_model(args.model_path)

    # 3. 逐张识别并汇总
    results = []
    for item in tqdm(image_list, desc="识别中"):
        path = item["local_path"]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 无法打开 {path}: {e}")
            results.append({"local_path": path, "url": item.get("url", ""), "error": str(e), "recognition": None})
            continue
        if use_server:
            rec = recognize_ship_api(args.api_base, image)
        else:
            rec = recognize_ship_local(model, processor, image)
        results.append({
            "local_path": path,
            "url": item.get("url", ""),
            "recognition": rec,
        })

    # 4. 保存为 JSON
    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
