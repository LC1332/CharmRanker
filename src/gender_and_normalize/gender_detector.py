"""
性别检测模块
使用 InsightFace 进行人脸检测和性别分类
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Literal
from pathlib import Path
import sys

# 添加 src 目录到 path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# InsightFace 导入
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("警告: insightface 未安装，请运行 pip install insightface onnxruntime")


@dataclass
class BoundingBox:
    """边界框（归一化坐标）"""
    x_min: float  # [0, 1]
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
    
    def get_center(self) -> Tuple[float, float]:
        """获取中心点（归一化坐标）"""
        return self.x_min + self.width / 2, self.y_min + self.height / 2
    
    def get_size(self) -> float:
        """获取边长的几何平均值"""
        return math.sqrt(self.width * self.height)


def _to_python_float(value):
    """将 numpy 类型转换为 Python 原生类型"""
    if value is None:
        return None
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    return float(value)


@dataclass
class GenderDetectionResult:
    """带性别信息的检测结果"""
    image_path: str
    image_width: int
    image_height: int
    face_bbox: Optional[BoundingBox] = None
    face_confidence: Optional[float] = None
    body_bbox: Optional[BoundingBox] = None
    body_confidence: Optional[float] = None
    gender: Optional[str] = None  # 'male' 或 'female'
    age: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典（确保 JSON 可序列化）"""
        result = {
            "image_path": self.image_path,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "gender": self.gender,
            "age": int(self.age) if self.age is not None else None,
            "error": self.error
        }
        
        if self.face_bbox:
            result["face_bbox"] = {
                "x_min": _to_python_float(self.face_bbox.x_min),
                "y_min": _to_python_float(self.face_bbox.y_min),
                "width": _to_python_float(self.face_bbox.width),
                "height": _to_python_float(self.face_bbox.height)
            }
            result["face_confidence"] = _to_python_float(self.face_confidence)
        if self.body_bbox:
            result["body_bbox"] = {
                "x_min": _to_python_float(self.body_bbox.x_min),
                "y_min": _to_python_float(self.body_bbox.y_min),
                "width": _to_python_float(self.body_bbox.width),
                "height": _to_python_float(self.body_bbox.height)
            }
            result["body_confidence"] = _to_python_float(self.body_confidence)
            
        return result


class GenderDetector:
    """带性别检测的人脸人体检测器"""
    
    # Face to Body 推算参数（与 ImageNormalizer 保持一致）
    BODY_WIDTH_RATIO = 3.0
    BODY_HEIGHT_RATIO = 6.0
    BODY_Y_OFFSET_RATIO = 2.5
    BODY_X_OFFSET_RATIO = 0.0
    
    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        """
        初始化检测器
        
        Args:
            det_size: 检测尺寸
            det_thresh: 检测阈值
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface 未安装，请运行: pip install insightface onnxruntime")
        
        # 初始化 InsightFace
        # 使用 buffalo_l 模型（包含人脸检测、性别年龄估计等）
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'genderage'],
            providers=['CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)
        
    def _get_largest_face(self, faces: list) -> Optional[dict]:
        """获取最大的人脸"""
        if not faces:
            return None
        
        largest_face = None
        largest_area = 0
        
        for face in faces:
            bbox = face.bbox  # [x1, y1, x2, y2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > largest_area:
                largest_area = area
                largest_face = face
        
        return largest_face
    
    def _estimate_body_from_face(self, face_bbox: BoundingBox) -> BoundingBox:
        """
        根据 face bbox 推算 body bbox
        
        Args:
            face_bbox: 人脸边界框（归一化坐标）
            
        Returns:
            推算的人体边界框（归一化坐标）
        """
        # 计算 face_size（几何平均值）
        face_size = face_bbox.get_size()
        
        # 计算 body 尺寸
        body_width = face_size * self.BODY_WIDTH_RATIO
        body_height = face_size * self.BODY_HEIGHT_RATIO
        
        # 计算 face 中心
        face_center_x, face_center_y = face_bbox.get_center()
        
        # 计算 body 中心
        body_center_x = face_center_x + face_size * self.BODY_X_OFFSET_RATIO
        body_center_y = face_center_y + face_size * self.BODY_Y_OFFSET_RATIO
        
        # 计算 body bbox
        body_x_min = body_center_x - body_width / 2
        body_y_min = body_center_y - body_height / 2
        
        # 确保不超出图片边界（归一化坐标）
        body_x_min = max(0, min(1 - body_width, body_x_min))
        body_y_min = max(0, min(1 - body_height, body_y_min))
        body_width = min(body_width, 1 - body_x_min)
        body_height = min(body_height, 1 - body_y_min)
        
        return BoundingBox(
            x_min=body_x_min,
            y_min=body_y_min,
            width=body_width,
            height=body_height
        )
    
    def detect(
        self,
        image_path: str,
        require_gender: Literal['any', 'male', 'female'] = 'any'
    ) -> Optional[GenderDetectionResult]:
        """
        检测图片中的人脸和人体，并进行性别过滤
        
        Args:
            image_path: 图片路径
            require_gender: 性别过滤
                - 'any': 返回所有检测结果（包含性别信息）
                - 'male': 只返回男性
                - 'female': 只返回女性
                
        Returns:
            GenderDetectionResult 或 None（如果没有匹配的结果）
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return GenderDetectionResult(
                image_path=image_path,
                image_width=0,
                image_height=0,
                error=f"无法读取图片: {image_path}"
            )
        
        height, width = image.shape[:2]
        
        # 使用 InsightFace 检测人脸和性别
        # InsightFace 需要 BGR 格式（OpenCV 默认）
        faces = self.face_app.get(image)
        
        # 获取最大的人脸
        largest_face = self._get_largest_face(faces)
        
        if largest_face is None:
            return GenderDetectionResult(
                image_path=image_path,
                image_width=width,
                image_height=height,
                error="未检测到人脸"
            )
        
        # 获取性别信息
        # InsightFace: gender 0=女性, 1=男性
        gender_code = largest_face.gender if hasattr(largest_face, 'gender') else None
        age = largest_face.age if hasattr(largest_face, 'age') else None
        
        if gender_code is not None:
            gender = 'male' if gender_code == 1 else 'female'
        else:
            gender = None
        
        # 性别过滤
        if require_gender != 'any' and gender != require_gender:
            return None
        
        # 获取人脸边界框
        bbox = largest_face.bbox  # [x1, y1, x2, y2]
        face_bbox = BoundingBox(
            x_min=max(0, bbox[0] / width),
            y_min=max(0, bbox[1] / height),
            width=min(1, (bbox[2] - bbox[0]) / width),
            height=min(1, (bbox[3] - bbox[1]) / height)
        )
        face_confidence = float(largest_face.det_score) if hasattr(largest_face, 'det_score') else None
        
        # 根据人脸推算 body
        body_bbox = self._estimate_body_from_face(face_bbox)
        body_confidence = face_confidence  # 使用人脸置信度作为 body 置信度
        
        return GenderDetectionResult(
            image_path=image_path,
            image_width=width,
            image_height=height,
            face_bbox=face_bbox,
            face_confidence=face_confidence,
            body_bbox=body_bbox,
            body_confidence=body_confidence,
            gender=gender,
            age=age
        )


# 全局检测器实例（延迟初始化）
_detector: Optional[GenderDetector] = None


def _get_detector() -> GenderDetector:
    """获取全局检测器实例"""
    global _detector
    if _detector is None:
        _detector = GenderDetector()
    return _detector


def detect_with_gender(
    image: str,
    require_gender: Literal['any', 'male', 'female'] = 'any'
) -> Optional[GenderDetectionResult]:
    """
    检测图片中的人脸和人体，根据性别进行过滤
    
    Args:
        image: 图片路径
        require_gender: 性别过滤条件
            - 'any': 返回所有检测到的人（包含 gender 字段表示性别）
            - 'male': 只返回男性的 face 和 body bounding box
            - 'female': 只返回女性的 face 和 body bounding box
            
    Returns:
        GenderDetectionResult 对象，包含:
            - face_bbox: 人脸边界框（归一化坐标）
            - body_bbox: 人体边界框（归一化坐标）
            - gender: 性别 ('male' 或 'female')
            - age: 年龄
        如果没有检测到符合条件的人，返回 None
    """
    detector = _get_detector()
    return detector.detect(image, require_gender)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        require_gender = sys.argv[2] if len(sys.argv) > 2 else 'any'
        
        result = detect_with_gender(image_path, require_gender)
        
        if result:
            print(f"检测结果: {result.to_dict()}")
        else:
            print(f"未检测到符合条件的人（require_gender={require_gender}）")
