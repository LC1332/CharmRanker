"""
测试 GenderNormalizer 类
从 celeb_female 和 celeb_male 中随机选取5张图片进行测试
"""

import random
from pathlib import Path

try:
    from normalizer import GenderNormalizer
except ImportError:
    from .normalizer import GenderNormalizer

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA = PROJECT_ROOT / "local_data"
CELEB_FEMALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_female"
CELEB_MALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_male"
OUTPUT_DIR = LOCAL_DATA / "test" / "gender_normalizer"


def get_random_images(directory: Path, count: int = 5, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """从目录中随机选取指定数量的图片"""
    all_images = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ]
    
    if len(all_images) < count:
        return all_images
    
    return random.sample(all_images, count)


def test_normalize():
    """测试基本归一化功能"""
    print("\n" + "=" * 60)
    print("测试 GenderNormalizer.normalize()")
    print("=" * 60)
    
    # 创建 normalizer
    normalizer = GenderNormalizer()
    
    # 测试女性图片
    print("\n--- 女性图片测试 (require_gender='female') ---")
    female_images = get_random_images(CELEB_FEMALE_DIR, count=5)
    female_success = 0
    
    for i, image_path in enumerate(female_images):
        print(f"\n[{i+1}/5] {image_path.name}")
        result = normalizer.normalize(str(image_path), require_gender='female')
        
        if result:
            output_path = OUTPUT_DIR / f"female_{i+1}_{image_path.name}"
            result.save(str(output_path))
            print(f"  -> 成功! gender={result.gender}, age={result.age}")
            female_success += 1
        else:
            print(f"  -> 失败: 未检测到女性人脸")
    
    # 测试男性图片
    print("\n--- 男性图片测试 (require_gender='male') ---")
    male_images = get_random_images(CELEB_MALE_DIR, count=5)
    male_success = 0
    
    for i, image_path in enumerate(male_images):
        print(f"\n[{i+1}/5] {image_path.name}")
        result = normalizer.normalize(str(image_path), require_gender='male')
        
        if result:
            output_path = OUTPUT_DIR / f"male_{i+1}_{image_path.name}"
            result.save(str(output_path))
            print(f"  -> 成功! gender={result.gender}, age={result.age}")
            male_success += 1
        else:
            print(f"  -> 失败: 未检测到男性人脸")
    
    return female_success, male_success


def test_render_bbox():
    """测试渲染 bbox 功能"""
    print("\n" + "=" * 60)
    print("测试渲染 Body/Face BBox 功能")
    print("=" * 60)
    
    # 创建 normalizer 并启用渲染
    normalizer = GenderNormalizer()
    normalizer.set_render_body(True)
    normalizer.set_render_face(True)
    
    # 选取一张女性图片和一张男性图片测试
    female_images = get_random_images(CELEB_FEMALE_DIR, count=1)
    male_images = get_random_images(CELEB_MALE_DIR, count=1)
    
    if female_images:
        image_path = female_images[0]
        print(f"\n渲染测试 (女性): {image_path.name}")
        result = normalizer.normalize(str(image_path), require_gender='female')
        if result:
            output_path = OUTPUT_DIR / f"render_female_{image_path.name}"
            result.save(str(output_path))
            print(f"  -> 保存到: {output_path.name}")
    
    if male_images:
        image_path = male_images[0]
        print(f"\n渲染测试 (男性): {image_path.name}")
        result = normalizer.normalize(str(image_path), require_gender='male')
        if result:
            output_path = OUTPUT_DIR / f"render_male_{image_path.name}"
            result.save(str(output_path))
            print(f"  -> 保存到: {output_path.name}")


def main():
    """主函数"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 设置随机种子
    random.seed(456)
    
    # 测试基本功能
    female_success, male_success = test_normalize()
    
    # 测试渲染 bbox
    test_render_bbox()
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试统计")
    print("=" * 60)
    print(f"女性归一化: {female_success}/5 成功")
    print(f"男性归一化: {male_success}/5 成功")
    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
