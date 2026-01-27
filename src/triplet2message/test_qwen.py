"""
Qwen VL 模型测试脚本

测试 qwen-vl-max 模型的 triplet 比较功能
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

LUMOS_API = os.getenv("LUMOS_API")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "")

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from triplet2message import ATTRACTIVENESS_PROMPT


def get_test_images() -> tuple[Path, Path, Path]:
    """获取测试图片路径"""
    test_dir = Path(__file__).parent.parent.parent / "local_data" / "test" / "triplet"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")
    
    images = sorted(test_dir.glob("*.jpg"))
    
    if len(images) < 3:
        raise ValueError(f"测试目录中图片不足3张: {test_dir}")
    
    return images[0], images[1], images[2]


def encode_image(image_path: Path) -> str:
    """编码图片为 base64 data URL"""
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{base64_data}"


def build_qwen_message(image_a: Path, image_b: Path, image_c: Path) -> dict:
    """构建 Qwen 消息格式（OpenAI 兼容）"""
    content = [
        {"type": "text", "text": ATTRACTIVENESS_PROMPT},
        {"type": "text", "text": "Image A:"},
        {"type": "image_url", "image_url": {"url": encode_image(image_a)}},
        {"type": "text", "text": "Image B:"},
        {"type": "image_url", "image_url": {"url": encode_image(image_b)}},
        {"type": "text", "text": "Image C:"},
        {"type": "image_url", "image_url": {"url": encode_image(image_c)}},
    ]
    
    return {"messages": [{"role": "user", "content": content}]}


def call_qwen(message: dict, model: str = "qwen-vl-max", temperature: float = 0.1) -> str:
    """调用 Qwen API"""
    if not LUMOS_API:
        raise ValueError("请在 .env 文件中设置 LUMOS_API")
    if not QWEN_BASE_URL:
        raise ValueError("请在 .env 文件中设置 QWEN_BASE_URL")
    
    url = f"{QWEN_BASE_URL}/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {LUMOS_API}",
        "Content-Type": "application/json"
    }
    
    payload = {
        **message,
        "model": model,
        "temperature": temperature,
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    content = result["choices"][0]["message"].get("content")
    
    if content is None:
        refusal = result["choices"][0]["message"].get("refusal", "Unknown reason")
        raise ValueError(f"Qwen 拒绝了请求: {refusal}")
    
    return content


def parse_response(response_text: str) -> dict:
    """解析响应"""
    cleaned = response_text.strip()
    
    if '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    return json.loads(cleaned.strip())


def save_result(result: dict, test_dir: Path):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = test_dir / f"result_qwen_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"   结果已保存到: {result_file}")


def main():
    """测试 Qwen VL 模型"""
    print("=" * 60)
    print("测试 Qwen VL 模型 (qwen-vl-max)")
    print("=" * 60)
    
    # 检查配置
    if not LUMOS_API:
        print("错误: 请在 .env 文件中设置 LUMOS_API")
        sys.exit(1)
    if not QWEN_BASE_URL:
        print("错误: 请在 .env 文件中设置 QWEN_BASE_URL")
        sys.exit(1)
    
    # 获取测试图片
    try:
        image_a, image_b, image_c = get_test_images()
        print(f"\n测试图片目录: {image_a.parent}")
        print(f"图片 A: {image_a.name}")
        print(f"图片 B: {image_b.name}")
        print(f"图片 C: {image_c.name}")
    except Exception as e:
        print(f"获取测试图片失败: {e}")
        sys.exit(1)
    
    # 构建消息
    print("\n正在调用 Qwen API...")
    try:
        message = build_qwen_message(image_a, image_b, image_c)
        response_text = call_qwen(message)
        result = parse_response(response_text)
        
        print("\n结果:")
        print(f"   最具吸引力: {result.get('most_attractive', 'N/A')}")
        print(f"   最不具吸引力: {result.get('least_attractive', 'N/A')}")
        print(f"\n分析:")
        analysis = result.get('analysis', '')
        print(f"   {analysis[:200]}..." if len(analysis) > 200 else f"   {analysis}")
        
        # 保存结果
        save_data = {
            "api_type": "qwen",
            "model": "qwen-vl-max",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "analysis": result.get("analysis", ""),
            "most_attractive": result.get("most_attractive"),
            "least_attractive": result.get("least_attractive"),
            "metadata": {
                "image_a": str(image_a),
                "image_b": str(image_b),
                "image_c": str(image_c),
            }
        }
        save_result(save_data, image_a.parent)
        
        print("\n" + "=" * 60)
        print("测试成功!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
