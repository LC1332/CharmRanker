"""
测试使用性别检测进行图片裁剪
从 celeb_female 和 celeb_male 中随机选取5张图片进行测试
"""

import random
import cv2
from pathlib import Path

from cropper import crop_with_gender

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA = PROJECT_ROOT / "local_data"
CELEB_FEMALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_female"
CELEB_MALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_male"
OUTPUT_DIR = LOCAL_DATA / "test" / "gender_crop"


def get_random_images(directory: Path, count: int = 5, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """从目录中随机选取指定数量的图片"""
    all_images = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ]
    
    if len(all_images) < count:
        return all_images
    
    return random.sample(all_images, count)


def test_female_crop():
    """测试女性图片裁剪"""
    print("\n" + "=" * 50)
    print("测试女性图片裁剪 (celeb_female)")
    print("=" * 50)
    
    images = get_random_images(CELEB_FEMALE_DIR, count=5)
    print(f"选取 {len(images)} 张图片")
    
    success_count = 0
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] 处理: {image_path.name}")
        
        cropped, result = crop_with_gender(str(image_path), require_gender='female')
        
        if cropped is not None:
            output_path = OUTPUT_DIR / f"female_{i+1}_{image_path.name}"
            cv2.imwrite(str(output_path), cropped)
            print(f"  -> 成功! gender={result['gender']}, age={result['age']}")
            print(f"     保存到: {output_path.name}")
            success_count += 1
        else:
            print(f"  -> 失败: 未检测到女性人脸")
    
    return success_count, len(images)


def test_male_crop():
    """测试男性图片裁剪"""
    print("\n" + "=" * 50)
    print("测试男性图片裁剪 (celeb_male)")
    print("=" * 50)
    
    images = get_random_images(CELEB_MALE_DIR, count=5)
    print(f"选取 {len(images)} 张图片")
    
    success_count = 0
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] 处理: {image_path.name}")
        
        cropped, result = crop_with_gender(str(image_path), require_gender='male')
        
        if cropped is not None:
            output_path = OUTPUT_DIR / f"male_{i+1}_{image_path.name}"
            cv2.imwrite(str(output_path), cropped)
            print(f"  -> 成功! gender={result['gender']}, age={result['age']}")
            print(f"     保存到: {output_path.name}")
            success_count += 1
        else:
            print(f"  -> 失败: 未检测到男性人脸")
    
    return success_count, len(images)


def main():
    """主函数"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 设置随机种子以便复现
    random.seed(123)
    
    # 运行测试
    female_success, female_total = test_female_crop()
    male_success, male_total = test_male_crop()
    
    # 统计结果
    print("\n" + "=" * 50)
    print("测试统计")
    print("=" * 50)
    print(f"女性裁剪: {female_success}/{female_total} 成功")
    print(f"男性裁剪: {male_success}/{male_total} 成功")
    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
