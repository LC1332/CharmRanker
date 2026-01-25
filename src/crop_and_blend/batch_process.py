"""
批量处理脚本
将所有符合要求的图片进行 crop_and_blend 处理后保存
"""

import cv2
import json
import hashlib
import string
from pathlib import Path
from tqdm import tqdm
from crop_and_blend import load_config, load_valid_records, process_record


def generate_filename_hash(original_path: str, length: int = 10) -> str:
    """
    根据原始文件路径生成固定长度的哈希文件名
    
    使用 SHA256 哈希后转换为只包含英文字母的字符串
    保证相同的原始路径生成相同的哈希值，与渲染设置无关
    
    Args:
        original_path: 原始文件路径
        length: 输出长度（默认10位）
        
    Returns:
        10位英文字母字符串
    """
    # 只使用文件路径进行 hash，不包含任何渲染配置
    hash_bytes = hashlib.sha256(original_path.encode('utf-8')).digest()
    
    # 将字节转换为只包含英文字母的字符串
    letters = string.ascii_lowercase
    result = []
    for byte in hash_bytes:
        if len(result) >= length:
            break
        result.append(letters[byte % 26])
    
    return ''.join(result)


def batch_process(config=None):
    """
    批量处理所有符合要求的图片
    
    Args:
        config: 配置字典，如果为None则加载默认配置
    """
    if config is None:
        config = load_config()
    
    # 获取输出目录
    project_root = Path(__file__).parent.parent.parent
    local_data_dir = project_root / config['data']['local_data_dir']
    target_folder = config['output']['target_folder']
    output_dir = local_data_dir / target_folder
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备日志文件
    log_path = output_dir / "crop_log.jsonl"
    
    # 加载有效记录
    print("正在加载数据记录...")
    valid_records = load_valid_records(config)
    total = len(valid_records)
    print(f"共找到 {total} 条有效记录")
    
    if total == 0:
        print("没有找到符合条件的记录，退出。")
        return
    
    # 显示配置信息
    print(f"\n配置信息:")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 渲染 Body: {config['render'].get('if_render_body', True)}")
    print(f"  - 渲染 Face: {config['render'].get('if_render_face', True)}")
    print(f"  - 最小短边: {config['filter']['min_body_short_edge']}px")
    print()
    
    # 批量处理
    success_count = 0
    fail_count = 0
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for record in tqdm(valid_records, desc="处理进度", unit="张"):
            original_path = record.get('image_path', '')
            
            try:
                # 处理图片
                result_image = process_record(record, config)
                
                # 生成输出文件名
                hashed_name = generate_filename_hash(original_path)
                output_filename = f"{hashed_name}.jpg"
                output_path = output_dir / output_filename
                
                # 保存图片
                cv2.imwrite(str(output_path), result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 计算变换后的 bbox 位置（在输出图片坐标系中）
                img_width = record.get('image_width', 0)
                img_height = record.get('image_height', 0)
                body_bbox = record['body_bbox']
                face_bbox = record.get('backup_face_bbox')
                
                # 计算输出尺寸和偏移量（与 crop_and_blend 中一致）
                bx = int(body_bbox['x_min'] * img_width)
                by = int(body_bbox['y_min'] * img_height)
                bw = int(body_bbox['width'] * img_width)
                bh = int(body_bbox['height'] * img_height)
                
                long_edge = max(bw, bh)
                padding_ratio = config['crop']['padding_ratio']
                output_size = int(long_edge * (1 + 2 * padding_ratio))
                
                body_center_x = bx + bw // 2
                body_center_y = by + bh // 2
                crop_x1 = body_center_x - output_size // 2
                crop_y1 = body_center_y - output_size // 2
                
                # 变换后的 body bbox 位置
                transformed_body_bbox = {
                    "x": bx - crop_x1,
                    "y": by - crop_y1,
                    "width": bw,
                    "height": bh
                }
                
                # 变换后的 face bbox 位置
                transformed_face_bbox = None
                if face_bbox is not None:
                    fx = int(face_bbox['x_min'] * img_width)
                    fy = int(face_bbox['y_min'] * img_height)
                    fw = int(face_bbox['width'] * img_width)
                    fh = int(face_bbox['height'] * img_height)
                    transformed_face_bbox = {
                        "x": fx - crop_x1,
                        "y": fy - crop_y1,
                        "width": fw,
                        "height": fh
                    }
                
                # 记录日志
                log_entry = {
                    "output_filename": output_filename,
                    "output_size": output_size,
                    "transformed_body_bbox": transformed_body_bbox,
                    "transformed_face_bbox": transformed_face_bbox,
                    "original_path": original_path,
                    "original_body_bbox": body_bbox,
                    "original_face_bbox": face_bbox,
                    "original_image_width": img_width,
                    "original_image_height": img_height
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                tqdm.write(f"处理失败: {original_path}, 错误: {e}")
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"  - 成功: {success_count} 张")
    print(f"  - 失败: {fail_count} 张")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 日志文件: {log_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理 crop_and_blend')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        config = load_config(args.config)
    
    batch_process(config)

