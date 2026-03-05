import base64
import requests
import json
from PIL import Image
import io

# ======== 配置区域 ==========
IMAGE_PATH = "./test_boat_images/boat.png"  # 你的图片路径
API_BASE_URL = "http://localhost:8080"  # 你的 API 地址
INSTRUCTION = "What do you see?"  # 你的提示词
# ===========================

def load_image_to_base64(image_path):
    """加载图片并转为 Base64 编码"""
    with Image.open(image_path) as img:
        # 保持原图格式
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def send_to_api(base64_image, instruction, api_url):
    """发送图片和指令到 API"""
    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Loading image...")
    img_base64 = load_image_to_base64(IMAGE_PATH)
    print("Sending to API...")
    result = send_to_api(img_base64, INSTRUCTION, API_BASE_URL)
    print("=== API Response ===")
    print(result)
    print("====================")

if __name__ == "__main__":
    main()
