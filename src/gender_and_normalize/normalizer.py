"""
GenderNormalizer - 基于性别检测的图片归一化模块
提供 .normalize(image_path) 接口进行图片裁剪
"""

import cv2
import numpy as np
import math
import yaml
from pathlib import Path
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

try:
    from .gender_detector import GenderDetector, BoundingBox
except ImportError:
    from gender_detector import GenderDetector, BoundingBox


@dataclass
class BBox:
    """边界框数据类（归一化坐标）"""
    x_min: float
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
    
    def get_center(self) -> Tuple[float, float]:
        """获取中心点（归一化坐标）"""
        return self.x_min + self.width / 2, self.y_min + self.height / 2
    
    def get_size(self) -> float:
        """获取边长的几何平均值"""
        return math.sqrt(self.width * self.height)


def load_config(config_path: Optional[str] = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class NormalizeResult:
    """归一化结果"""
    image: np.ndarray           # 裁剪后的图片 (BGR)
    gender: Optional[str]       # 性别 'male' 或 'female'
    age: Optional[int]          # 年龄
    face_bbox: Optional[BBox]   # 人脸边界框
    body_bbox: Optional[BBox]   # 人体边界框
    
    def save(self, output_path: str) -> bool:
        """保存图片"""
        return cv2.imwrite(str(output_path), self.image)


class GenderNormalizer:
    """
    基于性别检测的图片归一化处理器
    
    使用 InsightFace 进行人脸检测和性别分类，
    然后根据人脸位置推算人体区域进行裁剪。
    
    使用示例:
        normalizer = GenderNormalizer()
        result = normalizer.normalize("image.jpg")
        result.save("output.jpg")
        
        # 只处理女性
        result = normalizer.normalize("image.jpg", require_gender='female')
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 GenderNormalizer
        
        Args:
            config_path: 配置文件路径，默认使用 src/gender_and_normalize/config.yaml
        """
        self.config = load_config(config_path)
        
        # Face to Body 推算参数
        self.body_width_ratio = self.config['face_to_body']['width_ratio']
        self.body_height_ratio = self.config['face_to_body']['height_ratio']
        self.body_y_offset_ratio = self.config['face_to_body']['y_offset_ratio']
        self.body_x_offset_ratio = self.config['face_to_body']['x_offset_ratio']
        
        # 渲染参数
        self.render_body = self.config['render']['render_body']
        self.render_face = self.config['render']['render_face']
        self.body_bbox_color = tuple(self.config['render']['body_bbox_color'])
        self.face_bbox_color = tuple(self.config['render']['face_bbox_color'])
        self.line_thickness = self.config['render']['line_thickness']
        
        # 性别检测器（延迟初始化）
        self._detector: Optional[GenderDetector] = None
    
    @property
    def detector(self) -> GenderDetector:
        """获取检测器（延迟初始化）"""
        if self._detector is None:
            det_size = tuple(self.config['detection']['det_size'])
            det_thresh = self.config['detection']['det_thresh']
            self._detector = GenderDetector(det_size=det_size, det_thresh=det_thresh)
        return self._detector
    
    def _estimate_body_from_face(self, face_bbox: BBox) -> BBox:
        """根据 face bbox 推算 body bbox"""
        face_size = face_bbox.get_size()
        
        body_width = face_size * self.body_width_ratio
        body_height = face_size * self.body_height_ratio
        
        face_center_x, face_center_y = face_bbox.get_center()
        
        body_center_x = face_center_x + face_size * self.body_x_offset_ratio
        body_center_y = face_center_y + face_size * self.body_y_offset_ratio
        
        body_x_min = body_center_x - body_width / 2
        body_y_min = body_center_y - body_height / 2
        
        # 边界限制
        body_x_min = max(0, min(1 - body_width, body_x_min))
        body_y_min = max(0, min(1 - body_height, body_y_min))
        body_width = min(body_width, 1 - body_x_min)
        body_height = min(body_height, 1 - body_y_min)
        
        return BBox(
            x_min=body_x_min,
            y_min=body_y_min,
            width=body_width,
            height=body_height
        )
    
    def _crop_and_blend(
        self,
        image: np.ndarray,
        body_bbox: BBox,
        face_bbox: Optional[BBox] = None
    ) -> np.ndarray:
        """核心裁剪和混合函数"""
        img_height, img_width = image.shape[:2]
        
        bx, by, bw, bh = body_bbox.to_pixel(img_width, img_height)
        
        # 计算输出尺寸
        long_edge = max(bw, bh)
        padding_ratio = self.config['crop']['padding_ratio']
        output_size = int(long_edge * (1 + 2 * padding_ratio))
        
        # body_bbox 中心点
        body_center_x = bx + bw // 2
        body_center_y = by + bh // 2
        
        # 裁剪区域
        crop_x1 = body_center_x - output_size // 2
        crop_y1 = body_center_y - output_size // 2
        crop_x2 = crop_x1 + output_size
        crop_y2 = crop_y1 + output_size
        
        # 创建画布
        bg_color = self.config['crop']['background_color']
        canvas = np.full((output_size, output_size, 3), bg_color, dtype=np.uint8)
        
        # 计算有效复制区域
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img_width, crop_x2)
        src_y2 = min(img_height, crop_y2)
        
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
        # 高斯模糊
        blur_sigma_ratio = self.config['blend']['blur_sigma_ratio']
        sigma = int(long_edge * blur_sigma_ratio)
        if sigma < 1:
            sigma = 1
        kernel_size = sigma * 6 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma)
        
        # 创建遮罩
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        
        body_on_canvas_x = bx - crop_x1
        body_on_canvas_y = by - crop_y1
        
        mask_x1 = max(0, body_on_canvas_x)
        mask_y1 = max(0, body_on_canvas_y)
        mask_x2 = min(output_size, body_on_canvas_x + bw)
        mask_y2 = min(output_size, body_on_canvas_y + bh)
        
        if mask_x2 > mask_x1 and mask_y2 > mask_y1:
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1.0
        
        # 模糊遮罩边界
        blend_transition_ratio = self.config['blend'].get('transition_ratio', 0.05)
        transition_sigma = int(long_edge * blend_transition_ratio)
        if transition_sigma < 1:
            transition_sigma = 1
        transition_kernel = transition_sigma * 6 + 1
        if transition_kernel % 2 == 0:
            transition_kernel += 1
        
        mask = cv2.GaussianBlur(mask, (transition_kernel, transition_kernel), transition_sigma)
        
        # 混合
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (canvas * mask_3ch + blurred_canvas * (1 - mask_3ch)).astype(np.uint8)
        
        # 渲染 bbox（如果启用）
        if self.render_body:
            cv2.rectangle(
                result,
                (body_on_canvas_x, body_on_canvas_y),
                (body_on_canvas_x + bw, body_on_canvas_y + bh),
                self.body_bbox_color,
                self.line_thickness
            )
        
        if self.render_face and face_bbox is not None:
            fx, fy, fw, fh = face_bbox.to_pixel(img_width, img_height)
            face_on_canvas_x = fx - crop_x1
            face_on_canvas_y = fy - crop_y1
            cv2.rectangle(
                result,
                (face_on_canvas_x, face_on_canvas_y),
                (face_on_canvas_x + fw, face_on_canvas_y + fh),
                self.face_bbox_color,
                self.line_thickness
            )
        
        return result
    
    def normalize(
        self,
        image_path: str,
        require_gender: Literal['any', 'male', 'female'] = 'any'
    ) -> Optional[NormalizeResult]:
        """
        对图片进行归一化处理
        
        Args:
            image_path: 图片路径
            require_gender: 性别过滤
                - 'any': 任意性别
                - 'male': 只处理男性
                - 'female': 只处理女性
                
        Returns:
            NormalizeResult 对象，包含裁剪后的图片和检测信息
            如果没有检测到符合条件的人脸，返回 None
        """
        # 检测人脸和性别
        detection = self.detector.detect(image_path, require_gender=require_gender)
        
        if detection is None or detection.face_bbox is None:
            return None
        
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        # 转换 face_bbox 为 BBox
        face_bbox = BBox(
            x_min=float(detection.face_bbox.x_min),
            y_min=float(detection.face_bbox.y_min),
            width=float(detection.face_bbox.width),
            height=float(detection.face_bbox.height)
        )
        
        # 根据人脸推算 body
        body_bbox = self._estimate_body_from_face(face_bbox)
        
        # 裁剪
        cropped = self._crop_and_blend(image, body_bbox, face_bbox)
        
        return NormalizeResult(
            image=cropped,
            gender=detection.gender,
            age=detection.age,
            face_bbox=face_bbox,
            body_bbox=body_bbox
        )
    
    def set_render_body(self, enabled: bool):
        """设置是否渲染 body bbox"""
        self.render_body = enabled
    
    def set_render_face(self, enabled: bool):
        """设置是否渲染 face bbox"""
        self.render_face = enabled


# 便捷函数
def normalize_with_gender(
    image_path: str,
    require_gender: Literal['any', 'male', 'female'] = 'any',
    config_path: Optional[str] = None
) -> Optional[NormalizeResult]:
    """
    便捷函数：对图片进行归一化处理
    
    Args:
        image_path: 图片路径
        require_gender: 性别过滤
        config_path: 配置文件路径
        
    Returns:
        NormalizeResult 对象或 None
    """
    normalizer = GenderNormalizer(config_path)
    return normalizer.normalize(image_path, require_gender)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        require_gender = sys.argv[2] if len(sys.argv) > 2 else 'any'
        
        normalizer = GenderNormalizer()
        result = normalizer.normalize(image_path, require_gender)
        
        if result:
            output_path = "normalized_output.jpg"
            result.save(output_path)
            print(f"结果已保存: {output_path}")
            print(f"性别: {result.gender}, 年龄: {result.age}")
        else:
            print(f"未检测到符合条件的人脸（require_gender={require_gender}）")
