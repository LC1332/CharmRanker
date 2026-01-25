"""
测试各家MLLM API的目标检测能力
- 先测试API连通性（文本问答）
- 然后测试目标检测（框出骆驼）
- 可视化结果并保存
"""

from __future__ import annotations

import os
import sys

# 清除可能干扰的环境变量，必须在导入SDK之前
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import json
import base64
import requests
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.getenv("LUMOS_API")
if not API_KEY:
    print("❌ 错误: 请在.env文件中设置LUMOS_API")
    sys.exit(1)


# API base URLs (从环境变量读取)
BASE_URLS = {
    "openai": os.getenv("OPENAI_BASE_URL", ""),
    "gemini": os.getenv("GEMINI_BASE_URL", ""),
    "qwen": os.getenv("QWEN_BASE_URL", ""),
    "claude": os.getenv("CLAUDE_BASE_URL", ""),
}

# 模型配置
MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-3-flash-preview",  # Gemini 3 支持目标检测
    "qwen": "qwen-vl-max",  # 通义千问视觉模型
    "claude": "claude-sonnet-4-20250514",
}

# 测试图片路径
TEST_IMAGE = Path(__file__).parent / "data" / "test_luotuo.jpg"
OUTPUT_DIR = Path(__file__).parent / "local_data" / "output"


def gemini_generate_content(model: str, contents: list, response_mime_type: str = None) -> dict:
    """调用 Gemini REST API"""
    url = f"{BASE_URLS['gemini']}/v1/models/{model}:generateContent"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"contents": contents}
    if response_mime_type:
        payload["generationConfig"] = {"responseMimeType": response_mime_type}
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def test_text_api(provider: str) -> bool:
    """测试文本API连通性"""
    print(f"\n🔄 测试 {provider} 文本API连通性...")
    
    try:
        if provider == "gemini":
            # Gemini 使用 REST API
            response = gemini_generate_content(
                model=MODELS["gemini"],
                contents=[{"parts": [{"text": "Hello, please respond with 'API connection successful' in Chinese."}], "role": "user"}]
            )
            result = response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # OpenAI兼容接口
            client = OpenAI(api_key=API_KEY, base_url=BASE_URLS[provider])
            response = client.chat.completions.create(
                model=MODELS[provider],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, please respond with 'API connection successful' in Chinese."},
                ],
                max_tokens=100,
            )
            result = response.choices[0].message.content
        
        print(f"✅ {provider} 连通性测试成功: {result[:50]}...")
        return True
    except Exception as e:
        print(f"❌ {provider} 连通性测试失败: {e}")
        return False


def load_image_as_base64(image_path: Path) -> str:
    """加载图片并转换为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_detection_gemini() -> Optional[List[Dict]]:
    """使用Gemini进行目标检测"""
    print(f"\n🔄 使用 Gemini 进行骆驼检测...")
    
    try:
        # 加载图片并转换为base64
        image_base64 = load_image_as_base64(TEST_IMAGE)
        
        # 检测提示词
        prompt = """Detect all camels in the image. 
Output a JSON list where each item has:
- "label": the object label (e.g. "camel")
- "box_2d": bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000

Only output the JSON array, no other text."""

        # 使用 REST API 调用 Gemini
        contents = [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                    {"text": prompt}
                ]
            }
        ]
        
        response = gemini_generate_content(
            model=MODELS["gemini"],
            contents=contents,
            response_mime_type="application/json"
        )
        
        result_text = response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"📝 Gemini 原始响应: {result_text}")
        
        # 解析JSON
        detections = json.loads(result_text)
        print(f"✅ Gemini 检测到 {len(detections)} 个目标")
        return detections
        
    except Exception as e:
        print(f"❌ Gemini 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_detection_openai() -> Optional[List[Dict]]:
    """使用OpenAI GPT-4V进行目标检测"""
    print(f"\n🔄 使用 OpenAI 进行骆驼检测...")
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URLS["openai"])
        
        # 加载图片为base64
        image_base64 = load_image_as_base64(TEST_IMAGE)
        
        # 检测提示词
        prompt = """Detect all camels in the image. 
Output a JSON list where each item has:
- "label": the object label (e.g. "camel")
- "box_2d": bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000

IMPORTANT: Estimate the bounding box coordinates carefully. The values should be between 0 and 1000.
Only output the JSON array, no other text or markdown formatting."""

        response = client.chat.completions.create(
            model=MODELS["openai"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        result_text = response.choices[0].message.content
        print(f"📝 OpenAI 原始响应: {result_text}")
        
        # 清理可能的markdown格式
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result_text = result_text.strip()
        
        # 解析JSON
        detections = json.loads(result_text)
        print(f"✅ OpenAI 检测到 {len(detections)} 个目标")
        return detections
        
    except Exception as e:
        print(f"❌ OpenAI 检测失败: {e}")
        return None


def fix_json(text: str) -> str:
    """尝试修复常见的JSON格式错误"""
    import re
    text = text.strip()
    
    # 修复 box_2d 数组中 ] 写成 } 的情况，如 [1, 2, 3, 4}} -> [1, 2, 3, 4]}
    # 匹配 box_2d": [数字, 数字, 数字, 数字}} 的模式
    text = re.sub(r'(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\}(\})', r'\1]\2', text)
    
    # 修复缺少结尾 ] 的情况
    if text.startswith("[") and not text.endswith("]"):
        if text.endswith("}"):
            text = text + "]"
        elif text.endswith("},"):
            text = text[:-1] + "]"
    
    # 检查括号匹配
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    if open_brackets > close_brackets:
        text = text + "]" * (open_brackets - close_brackets)
    
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text = text + "}" * (open_braces - close_braces)
        
    return text


def test_detection_qwen() -> Optional[List[Dict]]:
    """使用Qwen-VL进行目标检测"""
    print(f"\n🔄 使用 Qwen-VL 进行骆驼检测...")
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URLS["qwen"])
        
        # 加载图片为base64
        image_base64 = load_image_as_base64(TEST_IMAGE)
        
        # 检测提示词
        prompt = """Detect all camels in the image. 
Output a JSON list where each item has:
- "label": the object label (e.g. "camel")
- "box_2d": bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000

IMPORTANT: Estimate the bounding box coordinates carefully. The values should be between 0 and 1000.
Only output the JSON array, no other text or markdown formatting."""

        response = client.chat.completions.create(
            model=MODELS["qwen"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        result_text = response.choices[0].message.content
        print(f"📝 Qwen 原始响应: {result_text}")
        
        # 清理可能的markdown格式
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result_text = result_text.strip()
        
        # 尝试修复JSON格式
        try:
            detections = json.loads(result_text)
        except json.JSONDecodeError:
            result_text = fix_json(result_text)
            detections = json.loads(result_text)
            
        print(f"✅ Qwen 检测到 {len(detections)} 个目标")
        return detections
        
    except Exception as e:
        print(f"❌ Qwen 检测失败: {e}")
        return None


def test_detection_claude() -> Optional[List[Dict]]:
    """使用Claude进行目标检测"""
    print(f"\n🔄 使用 Claude 进行骆驼检测...")
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URLS["claude"])
        
        # 加载图片为base64
        image_base64 = load_image_as_base64(TEST_IMAGE)
        
        # 检测提示词
        prompt = """Detect all camels in the image. 
Output a JSON list where each item has:
- "label": the object label (e.g. "camel")
- "box_2d": bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000

IMPORTANT: Estimate the bounding box coordinates carefully. The values should be between 0 and 1000.
Only output the JSON array, no other text or markdown formatting."""

        response = client.chat.completions.create(
            model=MODELS["claude"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        result_text = response.choices[0].message.content
        print(f"📝 Claude 原始响应: {result_text}")
        
        # 清理可能的markdown格式
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result_text = result_text.strip()
        
        # 解析JSON
        detections = json.loads(result_text)
        print(f"✅ Claude 检测到 {len(detections)} 个目标")
        return detections
        
    except Exception as e:
        print(f"❌ Claude 检测失败: {e}")
        return None


def visualize_detections(
    image_path: Path, 
    detections: List[Dict], 
    output_path: Path,
    model_name: str,
    coord_format: str = "yxyx",  # "yxyx" for [ymin, xmin, ymax, xmax], "xyxy" for [xmin, ymin, xmax, ymax]
    normalized: bool = True  # True if coords are 0-1000, False if pixel coords
) -> None:
    """可视化检测结果"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 定义颜色
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    
    for i, det in enumerate(detections):
        box = det.get("box_2d", [])
        label = det.get("label", "unknown")
        
        if len(box) != 4:
            continue
        
        # 根据坐标格式转换
        if coord_format == "yxyx":
            # Gemini格式: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = box
        else:
            # 其他模型可能使用: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
        
        # 如果是归一化坐标，转换为像素坐标
        if normalized:
            x1 = int(xmin / 1000 * width)
            y1 = int(ymin / 1000 * height)
            x2 = int(xmax / 1000 * width)
            y2 = int(ymax / 1000 * height)
        else:
            # 直接使用像素坐标
            x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)
        
        color = colors[i % len(colors)]
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # 绘制标签背景
        text = f"{label}"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 25), text, fill="white", font=font)
    
    # 添加模型名称水印
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), f"Model: {model_name}", fill="#333333", font=font)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"💾 结果已保存至: {output_path}")


def main():
    print("=" * 60)
    print("🐫 MLLM 目标检测能力测试")
    print("=" * 60)
    
    # Step 1: 测试API连通性
    print("\n" + "=" * 60)
    print("📡 第一步: 测试API连通性")
    print("=" * 60)
    
    gemini_ok = test_text_api("gemini")
    openai_ok = test_text_api("openai")
    
    if not gemini_ok and not openai_ok:
        print("\n❌ 错误: Gemini 和 OpenAI API 都无法连接，退出测试")
        sys.exit(1)
    
    # Step 2: 测试目标检测
    print("\n" + "=" * 60)
    print("🎯 第二步: 测试目标检测能力")
    print("=" * 60)
    
    if not TEST_IMAGE.exists():
        print(f"❌ 错误: 测试图片不存在: {TEST_IMAGE}")
        sys.exit(1)
    
    print(f"📷 测试图片: {TEST_IMAGE}")
    
    # 测试各个模型
    results = {}
    
    if gemini_ok:
        detections = test_detection_gemini()
        if detections:
            results["gemini"] = detections
            output_path = OUTPUT_DIR / "gemini_luotuo.jpg"
            # Gemini 使用 [ymin, xmin, ymax, xmax] 格式
            visualize_detections(TEST_IMAGE, detections, output_path, "Gemini", coord_format="yxyx")
    
    if openai_ok:
        detections = test_detection_openai()
        if detections:
            results["openai"] = detections
            output_path = OUTPUT_DIR / "openai_luotuo.jpg"
            visualize_detections(TEST_IMAGE, detections, output_path, "OpenAI GPT-4o", coord_format="yxyx")
    
    # 也测试其他厂商
    qwen_ok = test_text_api("qwen")
    if qwen_ok:
        detections = test_detection_qwen()
        if detections:
            results["qwen"] = detections
            output_path = OUTPUT_DIR / "qwen_luotuo.jpg"
            # Qwen 返回的是像素坐标 [xmin, ymin, xmax, ymax]，不是归一化坐标
            visualize_detections(TEST_IMAGE, detections, output_path, "Qwen-VL", coord_format="xyxy", normalized=False)
    
    claude_ok = test_text_api("claude")
    if claude_ok:
        detections = test_detection_claude()
        if detections:
            results["claude"] = detections
            output_path = OUTPUT_DIR / "claude_luotuo.jpg"
            visualize_detections(TEST_IMAGE, detections, output_path, "Claude", coord_format="yxyx")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for model, dets in results.items():
        print(f"\n{model}:")
        for det in dets:
            print(f"  - {det.get('label', 'unknown')}: {det.get('box_2d', [])}")
    
    if not results:
        print("⚠️ 没有任何模型成功检测到目标")
    else:
        print(f"\n✅ 成功完成! 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

