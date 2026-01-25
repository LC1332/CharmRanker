"""
Crop and Blend 核心功能模块
将检测到的人体裁剪并居中，对背景进行高斯模糊处理
"""

import cv2
import numpy as np
import yaml
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


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
    
    def get_short_edge(self, img_width: int, img_height: int) -> int:
        """获取短边像素长度"""
        _, _, w, h = self.to_pixel(img_width, img_height)
        return min(w, h)
    
    def get_long_edge(self, img_width: int, img_height: int) -> int:
        """获取长边像素长度"""
        _, _, w, h = self.to_pixel(img_width, img_height)
        return max(w, h)


def load_config(config_path: Optional[str] = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_valid_records(config: dict) -> List[Dict]:
    """
    加载并筛选有效的检测记录
    
    Args:
        config: 配置字典
        
    Returns:
        符合条件的检测记录列表
    """
    project_root = Path(__file__).parent.parent.parent
    local_data_dir = project_root / config['data']['local_data_dir']
    
    valid_records = []
    min_short_edge = config['filter']['min_body_short_edge']
    
    for jsonl_file in config['data']['jsonl_files']:
        jsonl_path = local_data_dir / jsonl_file
        if not jsonl_path.exists():
            print(f"警告: 文件不存在 {jsonl_path}")
            continue
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # 检查是否有 body_bbox
                if 'body_bbox' not in record or record['body_bbox'] is None:
                    continue
                
                # 计算 body_bbox 短边
                body_bbox = BBox(**record['body_bbox'])
                img_width = record.get('image_width', 0)
                img_height = record.get('image_height', 0)
                
                if img_width == 0 or img_height == 0:
                    continue
                
                short_edge = body_bbox.get_short_edge(img_width, img_height)
                if short_edge < min_short_edge:
                    continue
                
                # 添加本地数据目录信息
                record['_local_data_dir'] = str(local_data_dir)
                valid_records.append(record)
    
    return valid_records


def crop_and_blend(
    image_path: str,
    body_bbox: Dict,
    face_bbox: Optional[Dict] = None,
    config: Optional[dict] = None
) -> np.ndarray:
    """
    核心裁剪和混合函数
    
    Args:
        image_path: 图片完整路径
        body_bbox: body 边界框 (mediapipe 格式，归一化坐标)
        face_bbox: face 边界框 (backup_face_bbox，可选)
        config: 配置字典
        
    Returns:
        处理后的正方形图片 (BGR 格式)
    """
    if config is None:
        config = load_config()
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # 解析 body_bbox
    body = BBox(**body_bbox)
    bx, by, bw, bh = body.to_pixel(img_width, img_height)
    
    # 计算输出尺寸
    long_edge = max(bw, bh)
    padding_ratio = config['crop']['padding_ratio']
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
    bg_color = config['crop']['background_color']
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
    blur_sigma_ratio = config['blend']['blur_sigma_ratio']
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
    blend_transition_ratio = config['blend'].get('transition_ratio', 0.05)
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
    line_thickness = config['render']['line_thickness']
    if_render_body = config['render'].get('if_render_body', True)
    if_render_face = config['render'].get('if_render_face', True)
    
    # 先渲染 face_bbox（绿色细线）
    if if_render_face and face_bbox is not None:
        face = BBox(**face_bbox)
        fx, fy, fw, fh = face.to_pixel(img_width, img_height)
        
        # 转换到画布坐标
        face_on_canvas_x = fx - crop_x1
        face_on_canvas_y = fy - crop_y1
        
        face_color = tuple(config['render']['face_bbox_color'])
        cv2.rectangle(
            result,
            (face_on_canvas_x, face_on_canvas_y),
            (face_on_canvas_x + fw, face_on_canvas_y + fh),
            face_color,
            line_thickness
        )
    
    # 再渲染 body_bbox（红色细线）
    if if_render_body:
        body_color = tuple(config['render']['body_bbox_color'])
        cv2.rectangle(
            result,
            (body_on_canvas_x, body_on_canvas_y),
            (body_on_canvas_x + bw, body_on_canvas_y + bh),
            body_color,
            line_thickness
        )
    
    return result


def process_record(record: Dict, config: Optional[dict] = None) -> np.ndarray:
    """
    处理单条检测记录
    
    Args:
        record: 检测记录字典
        config: 配置字典
        
    Returns:
        处理后的图片
    """
    if config is None:
        config = load_config()
    
    local_data_dir = record.get('_local_data_dir')
    if local_data_dir is None:
        project_root = Path(__file__).parent.parent.parent
        local_data_dir = project_root / config['data']['local_data_dir']
    
    image_path = Path(local_data_dir) / record['image_path']
    body_bbox = record['body_bbox']
    face_bbox = record.get('backup_face_bbox')
    
    return crop_and_blend(str(image_path), body_bbox, face_bbox, config)


def get_random_samples(n: int = 5, config: Optional[dict] = None) -> List[Tuple[Dict, np.ndarray]]:
    """
    随机获取 n 个样本并处理
    
    Args:
        n: 样本数量
        config: 配置字典
        
    Returns:
        (记录, 处理后图片) 的列表
    """
    if config is None:
        config = load_config()
    
    valid_records = load_valid_records(config)
    
    if len(valid_records) < n:
        samples = valid_records
    else:
        samples = random.sample(valid_records, n)
    
    results = []
    for record in samples:
        try:
            image = process_record(record, config)
            results.append((record, image))
        except Exception as e:
            print(f"处理失败: {record.get('image_path', 'unknown')}, 错误: {e}")
    
    return results


if __name__ == "__main__":
    # 简单测试
    config = load_config()
    print(f"已加载配置")
    
    valid_records = load_valid_records(config)
    print(f"有效记录数: {len(valid_records)}")
    
    if valid_records:
        # 随机测试一条记录
        record = random.choice(valid_records)
        print(f"测试图片: {record['image_path']}")
        
        result = process_record(record, config)
        print(f"输出尺寸: {result.shape}")

