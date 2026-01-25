"""
人物属性分类模块 - 使用 Gemini 进行性别、亚洲人等属性判断

功能：
- 对图片中红色框标注的人物进行属性分类
- 输出包括性别、是否亚洲人、是否唯一主体等信息
"""

from __future__ import annotations

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 清除可能干扰的环境变量，必须在导入SDK之前
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.getenv("LUMOS_API")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # 支持视觉的模型

# 分类 Prompt（英文优化版）
CLASSIFICATION_PROMPT = """Based on the input image, analyze the person highlighted by the RED bounding box and determine their attributes.

Please output a JSON object with the following fields:

- "analysis": A brief analysis describing whether the red box clearly identifies a primary subject, whether the person appears to be Asian, their gender, and any other relevant observations.

- "gender": Output "male" for male, "female" for female. If it's a false detection or cannot be determined with confidence, output "unpredictable".

- "if_asian": Output "yes" if the person appears to be Asian (East Asian, Southeast Asian, etc.), output "no" if they appear to be non-Asian, output "uncertain" if it cannot be determined.

- "if_ambiguous": Whether the red bounding box clearly identifies exactly one person. Output "no" if the box primarily contains one person (even if other people partially appear at the edges). Output "yes" if the bounding box clearly contains two or more complete persons.

- "if_correct_face": Whether the GREEN face bounding box (if present) belongs to the person highlighted by the red box. Output "yes" if it matches, "no" if it doesn't match, "no_face_box" if there is no green face box visible.

- "if_frontal": Whether the person is facing the camera (frontal or side view where facial features are visible). Output "yes" if the face and facial features (eyes, nose, mouth) can be seen. Output "no" if the person is facing away or the face is not visible.

- "false_alarm": Whether the red box is a false detection (no person inside). Output "yes" if the red box does NOT contain any person (false alarm). Output "no" if there IS a person inside the red box.

Output ONLY the JSON object, no additional text or markdown formatting."""


def load_image_as_base64(image_path: str | Path) -> str:
    """加载图片并转换为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str | Path) -> str:
    """根据文件扩展名获取MIME类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def gemini_generate_content(
    model: str,
    contents: list,
    response_mime_type: str = None,
    timeout: int = 120
) -> dict:
    """调用 Gemini REST API"""
    url = f"{GEMINI_BASE_URL}/v1/models/{model}:generateContent"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {"contents": contents}
    if response_mime_type:
        payload["generationConfig"] = {"responseMimeType": response_mime_type}
    
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_json_response(text: str) -> Dict[str, Any]:
    """解析 JSON 响应，处理可能的格式问题"""
    # 清理可能的 markdown 格式
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    text = text.strip()
    return json.loads(text)


def classify_gender_and_asian(
    image_path: str | Path,
    model: str = None,
    prompt: str = None
) -> Dict[str, Any]:
    """
    对图片中红色框标注的人物进行属性分类
    
    Args:
        image_path: 图片文件路径
        model: Gemini 模型名称，默认使用 GEMINI_MODEL
        prompt: 自定义 prompt，默认使用 CLASSIFICATION_PROMPT
    
    Returns:
        包含分类结果的字典，包括:
        - analysis: 分析描述
        - gender: 性别 (male/female/unpredictable)
        - if_asian: 是否亚洲人 (yes/no/uncertain)
        - if_ambiguous: 是否唯一主体 (yes/no)
        - if_correct_face: 绿色人脸框是否正确 (yes/no/no_face_box)
        - if_frontal: 是否正脸 (yes/no)
        - false_alarm: 是否误检 (yes/no)
    
    Raises:
        ValueError: 如果 API KEY 未设置或图片不存在
        requests.RequestException: 如果 API 调用失败
        json.JSONDecodeError: 如果响应解析失败
    """
    # 检查 API KEY
    if not API_KEY:
        raise ValueError("请在 .env 文件中设置 LUMOS_API")
    
    # 检查图片是否存在
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f"图片文件不存在: {image_path}")
    
    # 使用默认值
    model = model or GEMINI_MODEL
    prompt = prompt or CLASSIFICATION_PROMPT
    
    # 加载图片并转换为 base64
    image_base64 = load_image_as_base64(image_path)
    mime_type = get_image_mime_type(image_path)
    
    # 构建请求内容
    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                {"text": prompt}
            ]
        }
    ]
    
    # 调用 Gemini API
    response = gemini_generate_content(
        model=model,
        contents=contents,
        response_mime_type="application/json"
    )
    
    # 提取并解析响应
    result_text = response["candidates"][0]["content"]["parts"][0]["text"]
    result = parse_json_response(result_text)
    
    return result


def main():
    """测试函数 - 对 cropped_sample 下的图片进行分类"""
    print("=" * 60)
    print("🔍 人物属性分类测试")
    print("=" * 60)
    
    # 检查 API KEY
    if not API_KEY:
        print("❌ 错误: 请在 .env 文件中设置 LUMOS_API")
        sys.exit(1)
    
    # 测试图片路径
    test_dir = Path(__file__).parent.parent.parent / "local_data" / "visualization" / "cropped_sample"
    
    if not test_dir.exists():
        print(f"❌ 错误: 测试目录不存在: {test_dir}")
        sys.exit(1)
    
    # 获取所有图片
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ 错误: 测试目录中没有图片: {test_dir}")
        sys.exit(1)
    
    print(f"📁 测试目录: {test_dir}")
    print(f"📷 找到 {len(image_files)} 张图片")
    
    # 逐一测试
    for image_path in image_files:
        print(f"\n{'='*60}")
        print(f"📷 处理图片: {image_path.name}")
        print("=" * 60)
        
        try:
            result = classify_gender_and_asian(image_path)
            
            print("\n✅ 分类结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 验证 JSON 可以正确解析
            print("\n📋 字段验证:")
            expected_fields = ["analysis", "gender", "if_asian", "if_ambiguous", 
                             "if_correct_face", "if_frontal", "false_alarm"]
            for field in expected_fields:
                status = "✓" if field in result else "✗"
                value = result.get(field, "MISSING")
                print(f"  {status} {field}: {value}")
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

