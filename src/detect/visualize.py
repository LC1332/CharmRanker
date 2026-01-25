"""
检测结果可视化模块
用于测试和验证检测坐标是否正确
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .detector import DetectionResult, FaceBodyDetector


def draw_detection_result(
    image: np.ndarray,
    result: DetectionResult,
    draw_face: bool = True,
    draw_body: bool = True,
    draw_keypoints: bool = True
) -> np.ndarray:
    """
    在图片上绘制检测结果
    
    Args:
        image: BGR格式图片
        result: 检测结果
        draw_face: 是否绘制人脸框
        draw_body: 是否绘制人体框
        draw_keypoints: 是否绘制人脸关键点
        
    Returns:
        绘制后的图片
    """
    image = image.copy()
    height, width = image.shape[:2]
    
    # 绘制人脸框（绿色）
    if draw_face and result.face_bbox:
        x, y, w, h = result.face_bbox.to_pixel(width, height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 绘制置信度
        if result.face_confidence:
            label = f"Face: {result.face_confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制人脸关键点（蓝色）
    if draw_keypoints and result.face_keypoints:
        kp = result.face_keypoints
        keypoint_pairs = [
            ("RE", kp.right_eye),
            ("LE", kp.left_eye),
            ("N", kp.nose_tip),
            ("M", kp.mouth_center),
            ("RET", kp.right_ear_tragion),
            ("LET", kp.left_ear_tragion),
        ]
        
        for name, (px, py) in keypoint_pairs:
            x = int(px * width)
            y = int(py * height)
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            cv2.putText(image, name, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 绘制人体框（红色）
    if draw_body and result.body_bbox:
        x, y, w, h = result.body_bbox.to_pixel(width, height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 绘制置信度
        if result.body_confidence:
            label = f"Body: {result.body_confidence:.2f}"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return image


def visualize_single_image(
    image_path: str,
    output_path: str,
    detector: Optional[FaceBodyDetector] = None
) -> DetectionResult:
    """
    可视化单张图片的检测结果
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        detector: 检测器实例，如果为None则创建新的
        
    Returns:
        检测结果
    """
    if detector is None:
        detector = FaceBodyDetector()
    
    # 检测
    result = detector.detect(image_path)
    
    if result.error:
        print(f"检测错误: {result.error}")
        return result
    
    # 读取图片并绘制
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return result
    
    visualized = draw_detection_result(image, result)
    
    # 保存结果
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, visualized)
    
    print(f"可视化结果已保存到: {output_path}")
    return result


def visualize_multiple_images(
    image_paths: list,
    output_dir: str,
    detector: Optional[FaceBodyDetector] = None
) -> list:
    """
    可视化多张图片的检测结果
    
    Args:
        image_paths: 输入图片路径列表
        output_dir: 输出目录
        detector: 检测器实例
        
    Returns:
        检测结果列表
    """
    if detector is None:
        detector = FaceBodyDetector()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for image_path in image_paths:
        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_vis.jpg"
        result = visualize_single_image(image_path, str(output_path), detector)
        results.append(result)
    
    return results


if __name__ == "__main__":
    import sys
    
    # 默认测试路径
    project_root = Path(__file__).parent.parent.parent
    local_data = project_root / "local_data"
    vis_output_dir = local_data / "visualization"
    
    # 获取测试图片
    test_images = []
    
    # 从 Album_A 获取一张测试图片
    album_a_path = local_data / "Album_A" / "A" / "Meizi1"
    if album_a_path.exists():
        for img_path in album_a_path.rglob("*.jpg"):
            test_images.append(str(img_path))
            if len(test_images) >= 2:
                break
    
    # 从 univer-light 获取一张测试图片
    univer_light_path = local_data / "univer-light"
    if univer_light_path.exists():
        for img_path in univer_light_path.rglob("*.jpg"):
            test_images.append(str(img_path))
            if len(test_images) >= 4:
                break
    
    if not test_images:
        print("未找到测试图片")
        sys.exit(1)
    
    print(f"找到 {len(test_images)} 张测试图片")
    for img in test_images:
        print(f"  - {img}")
    
    # 进行可视化
    detector = FaceBodyDetector()
    results = visualize_multiple_images(test_images, str(vis_output_dir), detector)
    
    print("\n检测结果摘要:")
    for result in results:
        print(f"\n图片: {Path(result.image_path).name}")
        print(f"  尺寸: {result.image_width}x{result.image_height}")
        if result.face_bbox:
            print(f"  人脸: 置信度={result.face_confidence:.2f}")
        else:
            print(f"  人脸: 未检测到")
        if result.body_bbox:
            print(f"  人体: 置信度={result.body_confidence:.2f}")
        else:
            print(f"  人体: 未检测到")

