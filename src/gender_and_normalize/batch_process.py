"""
批量处理脚本
从 celeb_female/celeb_male 文件夹中处理图片，找出最大的女性/男性人脸，
裁剪并保存到 univer_female_celeb/univer_male_celeb，同时生成 jsonl 文件
"""

import os
import sys
import json
import random
import string
from pathlib import Path
from typing import Optional, Literal
from tqdm import tqdm

# 添加 src 目录到 path
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from gender_and_normalize.normalizer import GenderNormalizer, BBox


def generate_random_filename(length: int = 10) -> str:
    """生成随机文件名"""
    return ''.join(random.choices(string.ascii_lowercase, k=length)) + '.jpg'


def process_batch(
    input_dir: str,
    output_dir: str,
    output_jsonl: str,
    require_gender: Literal['male', 'female'],
    render_body: bool = True,
    render_face: bool = False
):
    """
    批量处理图片
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        output_jsonl: 输出 jsonl 文件路径
        require_gender: 要求的性别 ('male' 或 'female')
        render_body: 是否渲染 body bbox
        render_face: 是否渲染 face bbox
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件（包括 .jpg_xx 这样的特殊命名）
    def is_image_file(filename: str) -> bool:
        """判断是否为图片文件，支持 .jpg_xx 这样的命名"""
        name_lower = filename.lower()
        # 支持标准扩展名和 .jpg_xx 这样的命名
        return any(ext in name_lower for ext in ['.jpg', '.jpeg', '.png', '.webp'])
    
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and is_image_file(f.name)
    ]
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"要求性别: {require_gender}")
    print(f"输出目录: {output_path}")
    print(f"输出 JSONL: {output_jsonl}")
    
    # 初始化 normalizer
    normalizer = GenderNormalizer()
    normalizer.set_render_body(render_body)
    normalizer.set_render_face(render_face)
    
    # 处理计数
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 打开 jsonl 文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for image_file in tqdm(image_files, desc=f"处理 {require_gender}"):
            try:
                # 进行归一化处理
                result = normalizer.normalize(str(image_file), require_gender=require_gender)
                
                if result is None:
                    skip_count += 1
                    continue
                
                # 生成随机文件名
                output_filename = generate_random_filename()
                output_file = output_path / output_filename
                
                # 确保文件名不重复
                while output_file.exists():
                    output_filename = generate_random_filename()
                    output_file = output_path / output_filename
                
                # 保存图片
                result.save(str(output_file))
                
                # 获取输出图片尺寸
                output_size = result.image.shape[0]  # 正方形，高=宽
                
                # 计算 transformed bbox（在输出图片中的位置）
                # 这需要根据裁剪逻辑计算
                img_height, img_width = result.image.shape[:2]
                
                # 读取原图尺寸
                import cv2
                original_img = cv2.imread(str(image_file))
                orig_height, orig_width = original_img.shape[:2]
                
                # 构建 jsonl 记录
                record = {
                    "output_filename": output_filename,
                    "output_size": output_size,
                    "transformed_body_bbox": None,  # 需要计算
                    "transformed_face_bbox": None,  # 需要计算
                    "original_path": str(image_file.relative_to(input_path.parent.parent)),
                    "original_body_bbox": {
                        "x_min": float(result.body_bbox.x_min) if result.body_bbox else None,
                        "y_min": float(result.body_bbox.y_min) if result.body_bbox else None,
                        "width": float(result.body_bbox.width) if result.body_bbox else None,
                        "height": float(result.body_bbox.height) if result.body_bbox else None,
                    } if result.body_bbox else None,
                    "original_face_bbox": {
                        "x_min": float(result.face_bbox.x_min) if result.face_bbox else None,
                        "y_min": float(result.face_bbox.y_min) if result.face_bbox else None,
                        "width": float(result.face_bbox.width) if result.face_bbox else None,
                        "height": float(result.face_bbox.height) if result.face_bbox else None,
                    } if result.face_bbox else None,
                    "original_image_width": orig_width,
                    "original_image_height": orig_height,
                    "gender": result.gender,
                    "age": result.age,
                }
                
                # 计算 transformed bbox
                if result.body_bbox:
                    bx, by, bw, bh = result.body_bbox.to_pixel(orig_width, orig_height)
                    long_edge = max(bw, bh)
                    padding_ratio = normalizer.config['crop']['padding_ratio']
                    crop_output_size = int(long_edge * (1 + 2 * padding_ratio))
                    
                    body_center_x = bx + bw // 2
                    body_center_y = by + bh // 2
                    
                    crop_x1 = body_center_x - crop_output_size // 2
                    crop_y1 = body_center_y - crop_output_size // 2
                    
                    # body 在输出图片中的位置
                    body_on_canvas_x = bx - crop_x1
                    body_on_canvas_y = by - crop_y1
                    
                    record["transformed_body_bbox"] = {
                        "x": body_on_canvas_x,
                        "y": body_on_canvas_y,
                        "width": bw,
                        "height": bh
                    }
                    
                    # face 在输出图片中的位置
                    if result.face_bbox:
                        fx, fy, fw, fh = result.face_bbox.to_pixel(orig_width, orig_height)
                        face_on_canvas_x = fx - crop_x1
                        face_on_canvas_y = fy - crop_y1
                        
                        record["transformed_face_bbox"] = {
                            "x": face_on_canvas_x,
                            "y": face_on_canvas_y,
                            "width": fw,
                            "height": fh
                        }
                
                # 写入 jsonl
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                success_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"\n处理 {image_file.name} 时出错: {e}")
    
    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  跳过 (未检测到匹配性别): {skip_count}")
    print(f"  错误: {error_count}")


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    local_data = project_root / "local_data"
    
    # 输入输出路径
    celeb_female_dir = local_data / "claw_by_minimax" / "celeb_female"
    celeb_male_dir = local_data / "claw_by_minimax" / "celeb_male"
    
    output_female_dir = local_data / "univer_female_celeb"
    output_male_dir = local_data / "univer_male_celeb"
    
    output_female_jsonl = local_data / "univer_female_celeb.jsonl"
    output_male_jsonl = local_data / "univer_male_celeb.jsonl"
    
    print("=" * 60)
    print("批量处理 celeb_female -> univer_female_celeb")
    print("=" * 60)
    process_batch(
        input_dir=str(celeb_female_dir),
        output_dir=str(output_female_dir),
        output_jsonl=str(output_female_jsonl),
        require_gender='female',
        render_body=True,
        render_face=False
    )
    
    print("\n" + "=" * 60)
    print("批量处理 celeb_male -> univer_male_celeb")
    print("=" * 60)
    process_batch(
        input_dir=str(celeb_male_dir),
        output_dir=str(output_male_dir),
        output_jsonl=str(output_male_jsonl),
        require_gender='male',
        render_body=True,
        render_face=False
    )


if __name__ == "__main__":
    main()
