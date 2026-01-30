"""
Triplet to Message 模块 - 使用 MLLM 比较三张照片中人物的颜值

功能：
- triplet2message: 将三张图片路径生成 MLLM 消息格式
- call_mllm: 通用的 MLLM 调用接口，支持 Gemini 和 OpenAI
- parse_triplet_response: 解析 MLLM 响应
"""

from __future__ import annotations

import os
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from dotenv import load_dotenv

# 清除可能干扰的环境变量，必须在导入SDK之前
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

# 加载环境变量 - 从项目根目录加载 .env
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# API 配置 - 必须通过 .env 文件设置
LUMOS_API = os.getenv("LUMOS_API")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# 默认模型
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-5.2-chat-latest"

# 颜值比较 Prompt
ATTRACTIVENESS_PROMPT = """You are a talent scout selecting potential idol candidates or annual gala hosts for your company.

Please analyze these THREE images and identify which person is the MOST attractive and which person is the LEAST attractive.

Focus primarily on the person's inherent facial attractiveness and physical appearance (face, figure, body proportions) rather than clothing, background, or photo quality.

The person to be evaluated in each image has been highlighted with a red bounding box.

Please examine all three images carefully:
- Image A (first image)
- Image B (second image)
- Image C (third image)

**IMPORTANT: You MUST respond with ONLY a valid JSON object, no other text.**

Output Format (copy this structure exactly):
{
  "analysis": "Give a comprehensive analysis that you can detailedly comparing all three images",
  "most_attractive": "A" or "B" or "C" or "unpredictable",
  "least_attractive": "A" or "B" or "C" or "unpredictable"
}

- Set "most_attractive" to the letter of the image with the MOST attractive person.
- Set "least_attractive" to the letter of the image with the LEAST attractive person.
- Use "unpredictable" only if you genuinely cannot make a confident judgment.

Now analyze the three images and respond with ONLY the JSON object:
"""


def load_image_as_base64(image_path: str | Path) -> str:
    """加载图片并转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str | Path) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def encode_image_for_openai(image_path: str | Path) -> str:
    """编码图片为 OpenAI 格式的 data URL"""
    mime_type = get_image_mime_type(image_path)
    base64_data = load_image_as_base64(image_path)
    return f"data:{mime_type};base64,{base64_data}"


def triplet2message(
    image_a: str | Path,
    image_b: str | Path,
    image_c: str | Path,
    prompt: str = None,
    api_type: Literal["gemini", "openai"] = "gemini"
) -> Dict[str, Any]:
    """
    将三张图片路径生成 MLLM 消息格式
    
    Args:
        image_a: 图片A的路径
        image_b: 图片B的路径
        image_c: 图片C的路径
        prompt: 自定义 prompt，默认使用 ATTRACTIVENESS_PROMPT
        api_type: API 类型，"gemini" 或 "openai"
    
    Returns:
        符合对应 API 格式的消息字典
    """
    image_a = Path(image_a)
    image_b = Path(image_b)
    image_c = Path(image_c)
    
    # 验证图片存在
    for img_path, label in [(image_a, "A"), (image_b, "B"), (image_c, "C")]:
        if not img_path.exists():
            raise FileNotFoundError(f"图片 {label} 不存在: {img_path}")
    
    prompt = prompt or ATTRACTIVENESS_PROMPT
    
    if api_type == "gemini":
        return _build_gemini_message(image_a, image_b, image_c, prompt)
    elif api_type == "openai":
        return _build_openai_message(image_a, image_b, image_c, prompt)
    else:
        raise ValueError(f"不支持的 API 类型: {api_type}")


def _build_gemini_message(
    image_a: Path, 
    image_b: Path, 
    image_c: Path, 
    prompt: str
) -> Dict[str, Any]:
    """构建 Gemini API 格式的消息"""
    parts = [
        {"text": prompt},
        {"text": "Image A:"},
        {
            "inline_data": {
                "mime_type": get_image_mime_type(image_a),
                "data": load_image_as_base64(image_a)
            }
        },
        {"text": "Image B:"},
        {
            "inline_data": {
                "mime_type": get_image_mime_type(image_b),
                "data": load_image_as_base64(image_b)
            }
        },
        {"text": "Image C:"},
        {
            "inline_data": {
                "mime_type": get_image_mime_type(image_c),
                "data": load_image_as_base64(image_c)
            }
        },
    ]
    
    return {
        "contents": [{"role": "user", "parts": parts}]
    }


def _build_openai_message(
    image_a: Path, 
    image_b: Path, 
    image_c: Path, 
    prompt: str
) -> Dict[str, Any]:
    """构建 OpenAI API 格式的消息"""
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Image A:"},
        {"type": "image_url", "image_url": {"url": encode_image_for_openai(image_a)}},
        {"type": "text", "text": "Image B:"},
        {"type": "image_url", "image_url": {"url": encode_image_for_openai(image_b)}},
        {"type": "text", "text": "Image C:"},
        {"type": "image_url", "image_url": {"url": encode_image_for_openai(image_c)}},
    ]
    
    return {
        "messages": [{"role": "user", "content": content}]
    }


def call_mllm(
    message: Dict[str, Any],
    api_type: Literal["gemini", "openai"] = "gemini",
    model: str = None,
    temperature: float = 0.1,
    timeout: int = 120
) -> str:
    """
    通用的 MLLM 调用接口
    
    Args:
        message: triplet2message 生成的消息
        api_type: API 类型，"gemini" 或 "openai"
        model: 模型名称，默认根据 api_type 选择
        temperature: 温度参数
        timeout: 超时时间（秒）
    
    Returns:
        MLLM 的响应文本
    """
    if api_type == "gemini":
        return _call_gemini(message, model, temperature, timeout)
    elif api_type == "openai":
        return _call_openai(message, model, temperature, timeout)
    else:
        raise ValueError(f"不支持的 API 类型: {api_type}")


def _call_gemini(
    message: Dict[str, Any],
    model: str = None,
    temperature: float = 0.1,
    timeout: int = 120
) -> str:
    """调用 Gemini API"""
    if not LUMOS_API:
        raise ValueError("请在 .env 文件中设置 LUMOS_API")
    if not GEMINI_BASE_URL:
        raise ValueError("请在 .env 文件中设置 GEMINI_BASE_URL")
    
    model = model or DEFAULT_GEMINI_MODEL
    url = f"{GEMINI_BASE_URL}/v1/models/{model}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {LUMOS_API}",
        "Content-Type": "application/json"
    }
    
    payload = {
        **message,
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(
    message: Dict[str, Any],
    model: str = None,
    temperature: float = 0.1,
    timeout: int = 120
) -> str:
    """调用 OpenAI 兼容 API"""
    if not LUMOS_API:
        raise ValueError("请在 .env 文件中设置 LUMOS_API")
    if not OPENAI_BASE_URL:
        raise ValueError("请在 .env 文件中设置 OPENAI_BASE_URL")
    
    model = model or DEFAULT_OPENAI_MODEL
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    
    # gpt-5.2-chat-latest 只支持 temperature=1
    if "gpt-5.2-chat" in model:
        temperature = 1.0
    
    headers = {
        "Authorization": f"Bearer {LUMOS_API}",
        "Content-Type": "application/json"
    }
    
    payload = {
        **message,
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    
    result = response.json()
    message = result["choices"][0]["message"]
    content = message.get("content")
    
    # 处理模型拒绝回答的情况
    if content is None:
        refusal = message.get("refusal", "Unknown refusal reason")
        raise ValueError(f"OpenAI 拒绝了请求: {refusal}")
    
    return content


def parse_triplet_response(response_text: str) -> Dict[str, Any]:
    """
    解析 MLLM 响应
    
    Args:
        response_text: MLLM 返回的文本
    
    Returns:
        解析后的字典，包含:
        - analysis: 分析描述
        - most_attractive: "A" / "B" / "C" / "unpredictable"
        - least_attractive: "A" / "B" / "C" / "unpredictable"
    """
    # 清理响应文本
    cleaned = response_text.strip()
    
    # 移除可能的 markdown 标记
    if '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    # 解析 JSON
    data = json.loads(cleaned.strip())
    
    # 标准化字段
    result = {
        "analysis": data.get("analysis", ""),
        "most_attractive": _normalize_choice(data.get("most_attractive")),
        "least_attractive": _normalize_choice(data.get("least_attractive")),
    }
    
    return result


def _normalize_choice(value: Any) -> Optional[str]:
    """标准化选择值"""
    if value is None:
        return "unpredictable"
    
    value_str = str(value).strip().upper()
    
    if value_str in ["A", "B", "C"]:
        return value_str
    elif value_str.lower() == "unpredictable":
        return "unpredictable"
    else:
        return "unpredictable"


def compare_triplet(
    image_a: str | Path,
    image_b: str | Path,
    image_c: str | Path,
    api_type: Literal["gemini", "openai"] = "gemini",
    model: str = None,
    prompt: str = None,
    temperature: float = 0.1,
    timeout: int = 120
) -> Dict[str, Any]:
    """
    便捷函数：比较三张图片中人物的颜值
    
    这是一个封装函数，内部调用 triplet2message -> call_mllm -> parse_triplet_response
    
    Args:
        image_a, image_b, image_c: 三张图片的路径
        api_type: API 类型，"gemini" 或 "openai"
        model: 模型名称
        prompt: 自定义 prompt
        temperature: 温度参数
        timeout: 超时时间
    
    Returns:
        解析后的比较结果
    """
    # 生成消息
    message = triplet2message(image_a, image_b, image_c, prompt, api_type)
    
    # 调用 MLLM
    response_text = call_mllm(message, api_type, model, temperature, timeout)
    
    # 解析响应
    result = parse_triplet_response(response_text)
    
    # 添加元数据
    result["metadata"] = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "image_c": str(image_c),
        "api_type": api_type,
        "model": model or (DEFAULT_GEMINI_MODEL if api_type == "gemini" else DEFAULT_OPENAI_MODEL),
    }
    result["raw_response"] = response_text
    
    return result
