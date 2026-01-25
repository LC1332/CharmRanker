"""
ImageNormalizer 测试脚本
测试有 body 框和没有 body 框的两种场景
"""

import cv2
import json
from pathlib import Path
import sys

# 添加 src 目录到 path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ImageNormalizer.image_normalizer import ImageNormalizer


def find_test_images():
    """
    从 Album_A_detect_result.jsonl 中找一张有 body 框和一张没有 body 框但有 face 框的图片
    
    Returns:
        (有body框的记录, 没有body框但有face框的记录)
    """
    project_root = Path(__file__).parent.parent.parent
    jsonl_path = project_root / "local_data" / "Album_A_detect_result.jsonl"
    
    with_body = None
    without_body = None
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            
            has_face = 'face_bbox' in record and record['face_bbox'] is not None
            has_body = 'body_bbox' in record and record['body_bbox'] is not None
            
            if has_body and with_body is None:
                with_body = record
            
            if has_face and not has_body and without_body is None:
                without_body = record
            
            if with_body and without_body:
                break
    
    return with_body, without_body


def test_normalizer():
    """测试 ImageNormalizer"""
    project_root = Path(__file__).parent.parent.parent
    local_data_dir = project_root / "local_data"
    output_dir = local_data_dir / "visualization" / "test_normalizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("初始化 ImageNormalizer...")
    normalizer = ImageNormalizer()
    
    print("查找测试图片...")
    with_body_record, without_body_record = find_test_images()
    
    # 测试1: 有 body 框的图片
    print("\n=== 测试1: 有 body 框的图片 ===")
    if with_body_record:
        image_path = local_data_dir / with_body_record['image_path']
        print(f"图片路径: {image_path}")
        print(f"Body bbox: {with_body_record['body_bbox']}")
        
        result = normalizer.normalize(str(image_path))
        output_path = output_dir / "with_body.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"输出尺寸: {result.shape}")
        print(f"结果保存到: {output_path}")
    else:
        print("未找到有 body 框的测试图片")
    
    # 测试2: 没有 body 框但有 face 框的图片
    print("\n=== 测试2: 没有 body 框但有 face 框的图片 ===")
    if without_body_record:
        image_path = local_data_dir / without_body_record['image_path']
        print(f"图片路径: {image_path}")
        print(f"Face bbox: {without_body_record['face_bbox']}")
        
        result = normalizer.normalize(str(image_path))
        output_path = output_dir / "without_body_with_face.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"输出尺寸: {result.shape}")
        print(f"结果保存到: {output_path}")
    else:
        print("未找到没有 body 框但有 face 框的测试图片")
    
    # 测试3: 使用提供的 body_bbox（像素坐标）
    print("\n=== 测试3: 使用提供的 body_bbox ===")
    if with_body_record:
        image_path = local_data_dir / with_body_record['image_path']
        img_width = with_body_record['image_width']
        img_height = with_body_record['image_height']
        
        body_bbox_norm = with_body_record['body_bbox']
        body_bbox_pixel = {
            "x": int(body_bbox_norm['x_min'] * img_width),
            "y": int(body_bbox_norm['y_min'] * img_height),
            "width": int(body_bbox_norm['width'] * img_width),
            "height": int(body_bbox_norm['height'] * img_height)
        }
        print(f"提供的 body_bbox (像素坐标): {body_bbox_pixel}")
        
        result = normalizer.normalize(str(image_path), body_bbox=body_bbox_pixel)
        output_path = output_dir / "with_provided_body.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"输出尺寸: {result.shape}")
        print(f"结果保存到: {output_path}")
    
    # 测试4: 使用提供的 face_bbox（像素坐标）
    print("\n=== 测试4: 使用提供的 face_bbox ===")
    if without_body_record:
        image_path = local_data_dir / without_body_record['image_path']
        img_width = without_body_record['image_width']
        img_height = without_body_record['image_height']
        
        face_bbox_norm = without_body_record['face_bbox']
        face_bbox_pixel = {
            "x": int(face_bbox_norm['x_min'] * img_width),
            "y": int(face_bbox_norm['y_min'] * img_height),
            "width": int(face_bbox_norm['width'] * img_width),
            "height": int(face_bbox_norm['height'] * img_height)
        }
        print(f"提供的 face_bbox (像素坐标): {face_bbox_pixel}")
        
        result = normalizer.normalize(str(image_path), face_bbox=face_bbox_pixel)
        output_path = output_dir / "with_provided_face.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"输出尺寸: {result.shape}")
        print(f"结果保存到: {output_path}")
    
    print(f"\n所有测试完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    test_normalizer()

