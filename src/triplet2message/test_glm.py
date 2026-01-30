"""
GLM-4V 模型测试脚本

测试智谱 AI glm-4.6v 模型的 triplet 比较功能
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

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

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


def build_glm_message(image_a: Path, image_b: Path, image_c: Path) -> dict:
    """构建 GLM 消息格式（OpenAI 兼容）"""
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


def call_glm(message: dict, model: str = "glm-4.6v", temperature: float = 0.1) -> str:
    """调用 GLM API"""
    if not GLM_API_KEY:
        raise ValueError("请在 .env 文件中设置 GLM_API_KEY")
    
    url = f"{GLM_BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
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
        raise ValueError(f"GLM 拒绝了请求: {refusal}")
    
    return content


def parse_response(response_text: str) -> dict:
    """解析响应"""
    cleaned = response_text.strip()
    
    if '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    return json.loads(cleaned.strip())


def save_result(result: dict, test_dir: Path, model: str):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("-", "_").replace(".", "_")
    result_file = test_dir / f"result_glm_{model_name}_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"   结果已保存到: {result_file}")


def main():
    """测试 GLM-4V 模型"""
    import argparse
    parser = argparse.ArgumentParser(description="测试智谱 AI GLM 视觉模型")
    parser.add_argument("--model", "-m", default="glm-4.6v", 
                        help="模型名称 (默认: glm-4.6v，可选: glm-4v-plus, glm-4v-flash)")
    args = parser.parse_args()
    
    model = args.model
    
    print("=" * 60)
    print(f"测试智谱 AI GLM 视觉模型 ({model})")
    print("=" * 60)
    
    # 检查配置
    if not GLM_API_KEY:
        print("错误: 请在 .env 文件中设置 GLM_API_KEY")
        sys.exit(1)
    
    print(f"\nAPI Base URL: {GLM_BASE_URL}")
    print(f"模型: {model}")
    
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
    print(f"\n正在调用 GLM API ({model})...")
    try:
        message = build_glm_message(image_a, image_b, image_c)
        response_text = call_glm(message, model=model)
        result = parse_response(response_text)
        
        print("\n结果:")
        print(f"   最具吸引力: {result.get('most_attractive', 'N/A')}")
        print(f"   最不具吸引力: {result.get('least_attractive', 'N/A')}")
        print(f"\n分析:")
        analysis = result.get('analysis', '')
        print(f"   {analysis[:200]}..." if len(analysis) > 200 else f"   {analysis}")
        
        # 保存结果
        save_data = {
            "api_type": "glm",
            "model": model,
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
        save_result(save_data, image_a.parent, model)
        
        print("\n" + "=" * 60)
        print("测试成功!")
        print("=" * 60)
        
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP 错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
