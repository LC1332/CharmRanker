"""
人脸和人体检测模块
使用 MediaPipe Tasks API 进行检测，只保留最大的人脸和人体
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path
import urllib.request
import os


# MediaPipe Tasks 模块
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 模型下载地址
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# 模型保存路径
MODEL_DIR = Path(__file__).parent / "models"
FACE_MODEL_PATH = MODEL_DIR / "blaze_face_short_range.tflite"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"


@dataclass
class BoundingBox:
    """边界框"""
    x_min: float  # 归一化坐标 [0, 1]
    y_min: float
    width: float
    height: float
    
    def to_pixel(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标 (x, y, w, h)"""
        x = int(self.x_min * img_width)
        y = int(self.y_min * img_height)
        w = int(self.width * img_width)
        h = int(self.height * img_height)
        return x, y, w, h
    
    def area(self) -> float:
        """计算面积（归一化）"""
        return self.width * self.height
    
    def intersection_area(self, other: 'BoundingBox') -> float:
        """计算与另一个边界框的交集面积"""
        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)
        x2 = min(self.x_min + self.width, other.x_min + other.width)
        y2 = min(self.y_min + self.height, other.y_min + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)
    
    def overlap_ratio_in(self, other: 'BoundingBox') -> float:
        """计算自身在另一个边界框中的重叠比例（交集面积 / 自身面积）"""
        self_area = self.area()
        if self_area == 0:
            return 0.0
        return self.intersection_area(other) / self_area


@dataclass
class FaceKeypoints:
    """人脸关键点（归一化坐标）"""
    right_eye: Tuple[float, float]
    left_eye: Tuple[float, float]
    nose_tip: Tuple[float, float]
    mouth_center: Tuple[float, float]
    right_ear_tragion: Tuple[float, float]
    left_ear_tragion: Tuple[float, float]


@dataclass
class DetectionResult:
    """单张图片的检测结果"""
    image_path: str
    image_width: int
    image_height: int
    face_bbox: Optional[BoundingBox] = None
    face_keypoints: Optional[FaceKeypoints] = None
    face_confidence: Optional[float] = None
    body_bbox: Optional[BoundingBox] = None
    body_confidence: Optional[float] = None
    # backup_face: 在body中重叠面积>=60%的最大人脸（当最大人脸不在body中时的备选）
    backup_face_bbox: Optional[BoundingBox] = None
    backup_face_keypoints: Optional[FaceKeypoints] = None
    backup_face_confidence: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典，方便JSON序列化"""
        result = {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "error": self.error
        }
        
        if self.face_bbox:
            result["face_bbox"] = asdict(self.face_bbox)
            result["face_confidence"] = self.face_confidence
        if self.face_keypoints:
            result["face_keypoints"] = asdict(self.face_keypoints)
        if self.body_bbox:
            result["body_bbox"] = asdict(self.body_bbox)
            result["body_confidence"] = self.body_confidence
        if self.backup_face_bbox:
            result["backup_face_bbox"] = asdict(self.backup_face_bbox)
            result["backup_face_confidence"] = self.backup_face_confidence
        if self.backup_face_keypoints:
            result["backup_face_keypoints"] = asdict(self.backup_face_keypoints)
            
        return result


def download_model(url: str, save_path: Path) -> None:
    """下载模型文件"""
    if save_path.exists():
        return
    
    print(f"正在下载模型: {save_path.name}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    urllib.request.urlretrieve(url, str(save_path))
    print(f"模型已下载: {save_path}")


def ensure_models_exist() -> None:
    """确保模型文件存在"""
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    download_model(POSE_MODEL_URL, POSE_MODEL_PATH)


class FaceBodyDetector:
    """人脸和人体检测器"""
    
    def __init__(
        self,
        face_min_detection_confidence: float = 0.5,
        pose_min_detection_confidence: float = 0.5,
    ):
        """
        初始化检测器
        
        Args:
            face_min_detection_confidence: 人脸检测最小置信度
            pose_min_detection_confidence: 姿态检测最小置信度
        """
        # 确保模型已下载
        ensure_models_exist()
        
        # 初始化人脸检测器（使用CPU）
        face_base_options = BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU
        )
        face_options = FaceDetectorOptions(
            base_options=face_base_options,
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=face_min_detection_confidence
        )
        self.face_detector = FaceDetector.create_from_options(face_options)
        
        # 初始化姿态检测器（使用CPU）
        pose_base_options = BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU
        )
        pose_options = PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=pose_min_detection_confidence
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)
        
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()
    
    def _detection_to_face(self, detection, img_width: int, img_height: int) -> Tuple[BoundingBox, Optional[FaceKeypoints], float]:
        """
        将单个检测结果转换为 (bbox, keypoints, confidence)
        """
        bbox_raw = detection.bounding_box
        bbox = BoundingBox(
            x_min=max(0, bbox_raw.origin_x / img_width),
            y_min=max(0, bbox_raw.origin_y / img_height),
            width=bbox_raw.width / img_width,
            height=bbox_raw.height / img_height
        )
        
        # 提取关键点
        keypoints_raw = detection.keypoints
        if keypoints_raw and len(keypoints_raw) >= 6:
            keypoints = FaceKeypoints(
                right_eye=(keypoints_raw[0].x, keypoints_raw[0].y),
                left_eye=(keypoints_raw[1].x, keypoints_raw[1].y),
                nose_tip=(keypoints_raw[2].x, keypoints_raw[2].y),
                mouth_center=(keypoints_raw[3].x, keypoints_raw[3].y),
                right_ear_tragion=(keypoints_raw[4].x, keypoints_raw[4].y),
                left_ear_tragion=(keypoints_raw[5].x, keypoints_raw[5].y),
            )
        else:
            keypoints = None
        
        # 获取置信度
        confidence = detection.categories[0].score if detection.categories else 0.0
        
        return bbox, keypoints, confidence

    def _get_largest_face(self, detections, img_width: int, img_height: int) -> Tuple[Optional[BoundingBox], Optional[FaceKeypoints], Optional[float]]:
        """
        从多个人脸检测结果中获取最大的一个
        
        Returns:
            (bbox, keypoints, confidence) 或 (None, None, None)
        """
        if not detections:
            return None, None, None
        
        largest_face = None
        largest_area = 0
        
        for detection in detections:
            bbox_raw = detection.bounding_box
            # bbox_raw 是像素坐标，转换为归一化坐标
            area = (bbox_raw.width / img_width) * (bbox_raw.height / img_height)
            
            if area > largest_area:
                largest_area = area
                largest_face = detection
        
        if largest_face is None:
            return None, None, None
        
        return self._detection_to_face(largest_face, img_width, img_height)
    
    def _get_largest_face_in_body(
        self, 
        detections, 
        body_bbox: BoundingBox, 
        img_width: int, 
        img_height: int,
        min_overlap_ratio: float = 0.6
    ) -> Tuple[Optional[BoundingBox], Optional[FaceKeypoints], Optional[float]]:
        """
        从多个人脸检测结果中获取在body中（重叠面积>=阈值）的最大人脸
        
        Args:
            detections: 人脸检测结果列表
            body_bbox: 人体边界框
            img_width: 图片宽度
            img_height: 图片高度
            min_overlap_ratio: 最小重叠比例（人脸在body中的面积占人脸面积的比例）
            
        Returns:
            (bbox, keypoints, confidence) 或 (None, None, None)
        """
        if not detections or body_bbox is None:
            return None, None, None
        
        largest_face_in_body = None
        largest_area_in_body = 0
        
        for detection in detections:
            bbox_raw = detection.bounding_box
            # 转换为归一化坐标的bbox
            face_bbox = BoundingBox(
                x_min=max(0, bbox_raw.origin_x / img_width),
                y_min=max(0, bbox_raw.origin_y / img_height),
                width=bbox_raw.width / img_width,
                height=bbox_raw.height / img_height
            )
            
            # 计算人脸在body中的重叠比例
            overlap_ratio = face_bbox.overlap_ratio_in(body_bbox)
            
            if overlap_ratio >= min_overlap_ratio:
                area = face_bbox.area()
                if area > largest_area_in_body:
                    largest_area_in_body = area
                    largest_face_in_body = detection
        
        if largest_face_in_body is None:
            return None, None, None
        
        return self._detection_to_face(largest_face_in_body, img_width, img_height)
    
    def _get_body_bbox_from_pose(self, pose_result) -> Tuple[Optional[BoundingBox], Optional[float]]:
        """
        从姿态关键点计算人体边界框
        
        Returns:
            (bbox, confidence) 或 (None, None)
        """
        if not pose_result.pose_landmarks:
            return None, None
        
        # 使用第一个检测到的人的姿态
        landmarks = pose_result.pose_landmarks[0]
        
        # 收集所有可见的关键点
        visible_points = []
        visibilities = []
        
        for lm in landmarks:
            if lm.visibility > 0.5:  # 只考虑可见度较高的点
                visible_points.append((lm.x, lm.y))
                visibilities.append(lm.visibility)
        
        if len(visible_points) < 5:  # 至少需要5个可见点
            return None, None
        
        xs = [p[0] for p in visible_points]
        ys = [p[1] for p in visible_points]
        
        x_min = max(0, min(xs))
        y_min = max(0, min(ys))
        x_max = min(1, max(xs))
        y_max = min(1, max(ys))
        
        # 添加一些padding
        padding = 0.05
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1, x_max + padding)
        y_max = min(1, y_max + padding)
        
        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )
        
        # 平均可见度作为置信度
        confidence = sum(visibilities) / len(visibilities)
        
        return bbox, confidence
    
    def detect(self, image_path: str) -> DetectionResult:
        """
        对单张图片进行人脸和人体检测
        
        Args:
            image_path: 图片路径
            
        Returns:
            DetectionResult 对象
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return DetectionResult(
                image_path=image_path,
                image_width=0,
                image_height=0,
                error=f"无法读取图片: {image_path}"
            )
        
        height, width = image.shape[:2]
        
        # 转换为RGB（MediaPipe需要RGB格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 人脸检测
        face_results = self.face_detector.detect(mp_image)
        face_detections = face_results.detections if face_results.detections else []
        face_bbox, face_keypoints, face_confidence = self._get_largest_face(
            face_detections,
            width, height
        )
        
        # 人体检测（使用姿态估计）
        pose_results = self.pose_detector.detect(mp_image)
        body_bbox, body_confidence = self._get_body_bbox_from_pose(pose_results)
        
        # 计算 backup_face：在body中重叠面积>=60%的最大人脸
        backup_face_bbox = None
        backup_face_keypoints = None
        backup_face_confidence = None
        
        if body_bbox is not None and face_detections:
            backup_face_bbox, backup_face_keypoints, backup_face_confidence = \
                self._get_largest_face_in_body(face_detections, body_bbox, width, height, min_overlap_ratio=0.6)
        
        return DetectionResult(
            image_path=image_path,
            image_width=width,
            image_height=height,
            face_bbox=face_bbox,
            face_keypoints=face_keypoints,
            face_confidence=face_confidence,
            body_bbox=body_bbox,
            body_confidence=body_confidence,
            backup_face_bbox=backup_face_bbox,
            backup_face_keypoints=backup_face_keypoints,
            backup_face_confidence=backup_face_confidence
        )
    
    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        批量检测图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            DetectionResult 列表
        """
        results = []
        for path in image_paths:
            results.append(self.detect(path))
        return results


if __name__ == "__main__":
    # 简单测试
    import sys
    
    if len(sys.argv) > 1:
        detector = FaceBodyDetector()
        result = detector.detect(sys.argv[1])
        print(result.to_dict())
