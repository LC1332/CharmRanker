"""
gender_and_normalize 模块
提供带性别检测的人脸人体检测和图片归一化功能
"""

from .gender_detector import (
    detect_with_gender,
    GenderDetector,
    GenderDetectionResult,
    BoundingBox,
)

from .normalizer import (
    GenderNormalizer,
    NormalizeResult,
    normalize_with_gender,
)

__all__ = [
    # 检测
    'detect_with_gender',
    'GenderDetector',
    'GenderDetectionResult',
    'BoundingBox',
    # 归一化
    'GenderNormalizer',
    'NormalizeResult',
    'normalize_with_gender',
]
