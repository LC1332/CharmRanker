"""
ImageNormalizer 模块
将图片中的人体裁剪并居中，对背景进行高斯模糊处理
"""

import cv2
import numpy as np
import yaml
import math
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import sys

# 添加 src 目录到 path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from detect.detector import FaceBodyDetector


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


class ImageNormalizer:
    """
    图片归一化处理器
    
    将图片中的人体裁剪并居中，对背景进行高斯模糊处理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 ImageNormalizer
        
        Args:
            config_path: 配置文件路径，默认使用 src/ImageNormalizer/config.yaml
        """
        self.config = load_config(config_path)
        self.detector = FaceBodyDetector()
        
        # Face to Body 推算参数
        self.body_width_ratio = self.config['face_to_body']['width_ratio']
        self.body_height_ratio = self.config['face_to_body']['height_ratio']
        self.body_y_offset_ratio = self.config['face_to_body']['y_offset_ratio']
        self.body_x_offset_ratio = self.config['face_to_body']['x_offset_ratio']
    
    def _estimate_body_from_face(
        self, 
        face_bbox: BBox, 
        img_width: int, 
        img_height: int
    ) -> BBox:
        """
        根据 face bbox 推算 body bbox
        
        Args:
            face_bbox: 人脸边界框
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            推算的人体边界框
        """
        # 计算 face_size（几何平均值）
        face_size = face_bbox.get_size()
        
        # 计算 body 尺寸
        body_width = face_size * self.body_width_ratio
        body_height = face_size * self.body_height_ratio
        
        # 计算 face 中心
        face_center_x, face_center_y = face_bbox.get_center()
        
        # 计算 body 中心
        body_center_x = face_center_x + face_size * self.body_x_offset_ratio
        body_center_y = face_center_y + face_size * self.body_y_offset_ratio
        
        # 计算 body bbox
        body_x_min = body_center_x - body_width / 2
        body_y_min = body_center_y - body_height / 2
        
        # 确保不超出图片边界（归一化坐标）
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
    
    def _get_center_square_bbox(self, img_width: int, img_height: int) -> BBox:
        """
        获取图片中心的最大正方形作为 body bbox
        
        Args:
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            中心正方形边界框
        """
        # 短边作为正方形边长
        short_edge = min(img_width, img_height)
        
        # 计算归一化坐标
        if img_width > img_height:
            # 宽图
            x_min = (img_width - short_edge) / (2 * img_width)
            y_min = 0.0
            width = short_edge / img_width
            height = 1.0
        else:
            # 高图或正方形
            x_min = 0.0
            y_min = (img_height - short_edge) / (2 * img_height)
            width = 1.0
            height = short_edge / img_height
        
        return BBox(x_min=x_min, y_min=y_min, width=width, height=height)
    
    def _crop_and_blend(
        self,
        image: np.ndarray,
        body_bbox: BBox
    ) -> np.ndarray:
        """
        核心裁剪和混合函数
        
        Args:
            image: 输入图片 (BGR 格式)
            body_bbox: body 边界框 (归一化坐标)
            
        Returns:
            处理后的正方形图片 (BGR 格式)
        """
        img_height, img_width = image.shape[:2]
        
        # 解析 body_bbox
        bx, by, bw, bh = body_bbox.to_pixel(img_width, img_height)
        
        # 计算输出尺寸
        long_edge = max(bw, bh)
        padding_ratio = self.config['crop']['padding_ratio']
        output_size = int(long_edge * (1 + 2 * padding_ratio))
        
        # 计算 body_bbox 中心点
        body_center_x = bx + bw // 2
        body_center_y = by + bh // 2
        
        # 计算裁剪区域（以 body 中心为中心，取 output_size 大小的区域）
        crop_x1 = body_center_x - output_size // 2
        crop_y1 = body_center_y - output_size // 2
        crop_x2 = crop_x1 + output_size
        crop_y2 = crop_y1 + output_size
        
        # 创建输出画布（白色背景）
        bg_color = self.config['crop']['background_color']
        canvas = np.full((output_size, output_size, 3), bg_color, dtype=np.uint8)
        
        # 计算源图片和目标画布的有效复制区域
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img_width, crop_x2)
        src_y2 = min(img_height, crop_y2)
        
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # 复制图片到画布
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
        # 高斯模糊处理
        blur_sigma_ratio = self.config['blend']['blur_sigma_ratio']
        sigma = int(long_edge * blur_sigma_ratio)
        if sigma < 1:
            sigma = 1
        # kernel_size 必须是奇数
        kernel_size = sigma * 6 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 对整个画布进行高斯模糊
        blurred_canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma)
        
        # 创建 body_bbox 的遮罩
        mask = np.zeros((output_size, output_size), dtype=np.float32)
        
        # 计算 body_bbox 在画布上的位置
        body_on_canvas_x = bx - crop_x1
        body_on_canvas_y = by - crop_y1
        
        # 在遮罩上绘制 body_bbox 区域（值为1）
        mask_x1 = max(0, body_on_canvas_x)
        mask_y1 = max(0, body_on_canvas_y)
        mask_x2 = min(output_size, body_on_canvas_x + bw)
        mask_y2 = min(output_size, body_on_canvas_y + bh)
        
        if mask_x2 > mask_x1 and mask_y2 > mask_y1:
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1.0
        
        # 对 mask 进行高斯模糊，实现渐变过渡效果
        blend_transition_ratio = self.config['blend'].get('transition_ratio', 0.05)
        transition_sigma = int(long_edge * blend_transition_ratio)
        if transition_sigma < 1:
            transition_sigma = 1
        transition_kernel = transition_sigma * 6 + 1
        if transition_kernel % 2 == 0:
            transition_kernel += 1
        
        # 模糊 mask 边界，实现 alpha blending 渐变效果
        mask = cv2.GaussianBlur(mask, (transition_kernel, transition_kernel), transition_sigma)
        
        # 混合：使用渐变 mask 实现平滑过渡
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (canvas * mask_3ch + blurred_canvas * (1 - mask_3ch)).astype(np.uint8)
        
        # 渲染边界框
        if_render_body = self.config['render'].get('if_render_body', False)
        
        if if_render_body:
            line_thickness = self.config['render']['line_thickness']
            body_color = tuple(self.config['render']['body_bbox_color'])
            cv2.rectangle(
                result,
                (body_on_canvas_x, body_on_canvas_y),
                (body_on_canvas_x + bw, body_on_canvas_y + bh),
                body_color,
                line_thickness
            )
        
        return result
    
    def normalize(
        self,
        image_path: str,
        body_bbox: Optional[Dict] = None,
        face_bbox: Optional[Dict] = None
    ) -> np.ndarray:
        """
        对图片进行归一化处理
        
        处理优先级:
        1. 如果提供了 body_bbox，直接使用
        2. 否则进行检测，如果检测到 body，使用检测结果
        3. 如果没有检测到 body 但检测到 face（或提供了 face_bbox），根据 face 推算 body
        4. 如果都没有，取图片中心最大正方形作为 body
        
        Args:
            image_path: 图片路径
            body_bbox: 可选的 body 边界框，格式 {"x": px, "y": py, "width": pw, "height": ph} (像素坐标)
            face_bbox: 可选的 face 边界框，格式 {"x": px, "y": py, "width": pw, "height": ph} (像素坐标)
            
        Returns:
            处理后的正方形图片 (BGR 格式)
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        img_height, img_width = image.shape[:2]
        
        # 处理输入的 bbox（像素坐标转归一化坐标）
        final_body_bbox = None
        
        if body_bbox is not None:
            # 使用提供的 body_bbox（像素坐标）
            final_body_bbox = BBox(
                x_min=body_bbox['x'] / img_width,
                y_min=body_bbox['y'] / img_height,
                width=body_bbox['width'] / img_width,
                height=body_bbox['height'] / img_height
            )
        else:
            # 进行检测
            detection_result = self.detector.detect(str(image_path))
            
            if detection_result.body_bbox is not None:
                # 检测到了 body
                final_body_bbox = BBox(
                    x_min=detection_result.body_bbox.x_min,
                    y_min=detection_result.body_bbox.y_min,
                    width=detection_result.body_bbox.width,
                    height=detection_result.body_bbox.height
                )
            elif face_bbox is not None:
                # 提供了 face_bbox，根据 face 推算 body（像素坐标）
                face_bbox_normalized = BBox(
                    x_min=face_bbox['x'] / img_width,
                    y_min=face_bbox['y'] / img_height,
                    width=face_bbox['width'] / img_width,
                    height=face_bbox['height'] / img_height
                )
                final_body_bbox = self._estimate_body_from_face(
                    face_bbox_normalized, img_width, img_height
                )
            elif detection_result.face_bbox is not None:
                # 检测到了 face，根据 face 推算 body
                face_bbox_detected = BBox(
                    x_min=detection_result.face_bbox.x_min,
                    y_min=detection_result.face_bbox.y_min,
                    width=detection_result.face_bbox.width,
                    height=detection_result.face_bbox.height
                )
                final_body_bbox = self._estimate_body_from_face(
                    face_bbox_detected, img_width, img_height
                )
            else:
                # 都没有，取图片中心最大正方形
                final_body_bbox = self._get_center_square_bbox(img_width, img_height)
        
        # 执行裁剪和混合
        return self._crop_and_blend(image, final_body_bbox)


# 便捷函数
def normalize_image(
    image_path: str,
    body_bbox: Optional[Dict] = None,
    face_bbox: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> np.ndarray:
    """
    便捷函数：对图片进行归一化处理
    
    Args:
        image_path: 图片路径
        body_bbox: 可选的 body 边界框
        face_bbox: 可选的 face 边界框
        config_path: 配置文件路径
        
    Returns:
        处理后的正方形图片 (BGR 格式)
    """
    normalizer = ImageNormalizer(config_path)
    return normalizer.normalize(image_path, body_bbox, face_bbox)


if __name__ == "__main__":
    # 简单测试
    import sys
    
    if len(sys.argv) > 1:
        normalizer = ImageNormalizer()
        result = normalizer.normalize(sys.argv[1])
        print(f"输出尺寸: {result.shape}")
        
        # 保存结果
        output_path = "test_output.jpg"
        cv2.imwrite(output_path, result)
        print(f"结果已保存到: {output_path}")

