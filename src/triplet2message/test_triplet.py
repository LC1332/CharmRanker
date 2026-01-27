"""
Triplet 比较测试脚本

测试 triplet2message 模块的 Gemini 和 OpenAI 调用功能
使用 local_data/test/triplet 中的图片进行测试
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from triplet2message import compare_triplet, triplet2message, call_mllm, parse_triplet_response


def get_test_images() -> tuple[Path, Path, Path]:
    """获取测试图片路径"""
    test_dir = Path(__file__).parent.parent.parent / "local_data" / "test" / "triplet"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")
    
    # 获取所有 jpg 图片
    images = sorted(test_dir.glob("*.jpg"))
    
    if len(images) < 3:
        raise ValueError(f"测试目录中图片不足3张: {test_dir}")
    
    return images[0], images[1], images[2]


def save_result(result: dict, api_type: str, test_dir: Path):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = test_dir / f"result_{api_type}_{timestamp}.json"
    
    # 不保存 raw_response 中的完整文本，太长
    save_data = {
        "api_type": api_type,
        "timestamp": timestamp,
        "analysis": result.get("analysis", ""),
        "most_attractive": result.get("most_attractive"),
        "least_attractive": result.get("least_attractive"),
        "metadata": result.get("metadata", {}),
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"   结果已保存到: {result_file}")


def test_gemini():
    """测试 Gemini API"""
    print("\n" + "=" * 60)
    print("测试 Gemini API")
    print("=" * 60)
    
    try:
        image_a, image_b, image_c = get_test_images()
        print(f"图片 A: {image_a.name}")
        print(f"图片 B: {image_b.name}")
        print(f"图片 C: {image_c.name}")
        
        print("\n正在调用 Gemini API...")
        result = compare_triplet(
            image_a, image_b, image_c,
            api_type="gemini"
        )
        
        print("\n结果:")
        print(f"   最具吸引力: {result['most_attractive']}")
        print(f"   最不具吸引力: {result['least_attractive']}")
        print(f"\n分析:")
        print(f"   {result['analysis'][:200]}..." if len(result['analysis']) > 200 else f"   {result['analysis']}")
        
        # 保存结果
        test_dir = Path(__file__).parent.parent.parent / "local_data" / "test" / "triplet"
        save_result(result, "gemini", test_dir)
        
        return True
        
    except Exception as e:
        print(f"\nGemini 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai():
    """测试 OpenAI API"""
    print("\n" + "=" * 60)
    print("测试 OpenAI API")
    print("=" * 60)
    
    try:
        image_a, image_b, image_c = get_test_images()
        print(f"图片 A: {image_a.name}")
        print(f"图片 B: {image_b.name}")
        print(f"图片 C: {image_c.name}")
        
        print("\n正在调用 OpenAI API...")
        result = compare_triplet(
            image_a, image_b, image_c,
            api_type="openai"
        )
        
        print("\n结果:")
        print(f"   最具吸引力: {result['most_attractive']}")
        print(f"   最不具吸引力: {result['least_attractive']}")
        print(f"\n分析:")
        print(f"   {result['analysis'][:200]}..." if len(result['analysis']) > 200 else f"   {result['analysis']}")
        
        # 保存结果
        test_dir = Path(__file__).parent.parent.parent / "local_data" / "test" / "triplet"
        save_result(result, "openai", test_dir)
        
        return True
        
    except Exception as e:
        print(f"\nOpenAI 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Triplet2Message 模块测试")
    print("=" * 60)
    
    # 显示测试图片信息
    try:
        image_a, image_b, image_c = get_test_images()
        print(f"\n测试图片目录: {image_a.parent}")
        print(f"图片 A: {image_a.name}")
        print(f"图片 B: {image_b.name}")
        print(f"图片 C: {image_c.name}")
    except Exception as e:
        print(f"获取测试图片失败: {e}")
        sys.exit(1)
    
    # 运行测试
    results = {}
    
    # 测试 Gemini
    results["gemini"] = test_gemini()
    
    # 测试 OpenAI
    results["openai"] = test_openai()
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for api_type, success in results.items():
        status = "成功" if success else "失败"
        symbol = "✓" if success else "✗"
        print(f"   {symbol} {api_type}: {status}")
    
    # 返回状态码
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
