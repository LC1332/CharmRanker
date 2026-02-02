"""
测试 gender_detector 模块
从 celeb_female 和 celeb_male 中随机选取图片进行测试
"""

import random
import json
import cv2
from pathlib import Path
from datetime import datetime

from gender_detector import detect_with_gender, GenderDetectionResult, BoundingBox

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA = PROJECT_ROOT / "local_data"
CELEB_FEMALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_female"
CELEB_MALE_DIR = LOCAL_DATA / "claw_by_minimax" / "celeb_male"
OUTPUT_DIR = LOCAL_DATA / "test" / "gender_and_normalize"


def get_random_images(directory: Path, count: int = 5, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """从目录中随机选取指定数量的图片"""
    all_images = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ]
    
    if len(all_images) < count:
        return all_images
    
    return random.sample(all_images, count)


def visualize_result(image_path: str, result: GenderDetectionResult, output_path: str):
    """可视化检测结果并保存"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # 绘制 face_bbox（绿色）
    if result.face_bbox:
        x, y, w, h = result.face_bbox.to_pixel(width, height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 添加性别和年龄标签
        label = f"{result.gender or 'unknown'}"
        if result.age:
            label += f", {result.age}y"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制 body_bbox（蓝色）
    if result.body_bbox:
        x, y, w, h = result.body_bbox.to_pixel(width, height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imwrite(output_path, image)
    print(f"结果已保存: {output_path}")


def test_female():
    """测试女性检测"""
    print("\n" + "=" * 50)
    print("测试 require_gender='female'")
    print("=" * 50)
    
    images = get_random_images(CELEB_FEMALE_DIR, count=5)
    print(f"从 {CELEB_FEMALE_DIR} 选取 {len(images)} 张图片")
    
    results = []
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] 处理: {image_path.name}")
        
        result = detect_with_gender(str(image_path), require_gender='female')
        
        if result:
            print(f"  -> 检测成功: gender={result.gender}, age={result.age}")
            print(f"     face_bbox: {result.face_bbox}")
            print(f"     body_bbox: {result.body_bbox}")
            
            # 保存可视化结果
            output_path = OUTPUT_DIR / f"female_{i+1}_{image_path.name}"
            visualize_result(str(image_path), result, str(output_path))
            
            results.append({
                "image": image_path.name,
                "result": result.to_dict()
            })
        else:
            print(f"  -> 未检测到女性（可能是男性或无人脸）")
            results.append({
                "image": image_path.name,
                "result": None,
                "note": "未检测到女性"
            })
    
    return results


def test_male():
    """测试男性检测"""
    print("\n" + "=" * 50)
    print("测试 require_gender='male'")
    print("=" * 50)
    
    images = get_random_images(CELEB_MALE_DIR, count=5)
    print(f"从 {CELEB_MALE_DIR} 选取 {len(images)} 张图片")
    
    results = []
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] 处理: {image_path.name}")
        
        result = detect_with_gender(str(image_path), require_gender='male')
        
        if result:
            print(f"  -> 检测成功: gender={result.gender}, age={result.age}")
            print(f"     face_bbox: {result.face_bbox}")
            print(f"     body_bbox: {result.body_bbox}")
            
            # 保存可视化结果
            output_path = OUTPUT_DIR / f"male_{i+1}_{image_path.name}"
            visualize_result(str(image_path), result, str(output_path))
            
            results.append({
                "image": image_path.name,
                "result": result.to_dict()
            })
        else:
            print(f"  -> 未检测到男性（可能是女性或无人脸）")
            results.append({
                "image": image_path.name,
                "result": None,
                "note": "未检测到男性"
            })
    
    return results


def main():
    """主函数"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 设置随机种子以便复现
    random.seed(42)
    
    # 运行测试
    female_results = test_female()
    male_results = test_male()
    
    # 保存测试报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "female_test": {
            "source_dir": str(CELEB_FEMALE_DIR),
            "require_gender": "female",
            "results": female_results
        },
        "male_test": {
            "source_dir": str(CELEB_MALE_DIR),
            "require_gender": "male",
            "results": male_results
        }
    }
    
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n测试报告已保存: {report_path}")
    
    # 统计结果
    female_success = sum(1 for r in female_results if r.get("result"))
    male_success = sum(1 for r in male_results if r.get("result"))
    
    print("\n" + "=" * 50)
    print("测试统计")
    print("=" * 50)
    print(f"女性检测: {female_success}/{len(female_results)} 成功")
    print(f"男性检测: {male_success}/{len(male_results)} 成功")


if __name__ == "__main__":
    main()
