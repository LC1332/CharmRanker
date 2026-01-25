"""
批量检测脚本
对 local_data/Album_A 和 local_data/univer-light 下的所有图片进行人脸和人体检测
结果保存为 JSONL 格式
"""

import json
import sys
from pathlib import Path
from typing import List, Generator

from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from detector import FaceBodyDetector, DetectionResult


# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP'}


def find_all_images(folder_path: Path) -> Generator[Path, None, None]:
    """
    递归查找文件夹下所有图片
    
    Args:
        folder_path: 文件夹路径
        
    Yields:
        图片路径
    """
    if not folder_path.exists():
        print(f"警告: 文件夹不存在 {folder_path}")
        return
    
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in IMAGE_EXTENSIONS:
            yield file_path


def process_folder(
    folder_path: Path,
    output_path: Path,
    detector: FaceBodyDetector,
    show_progress: bool = True
) -> int:
    """
    处理单个文件夹下的所有图片
    
    Args:
        folder_path: 输入文件夹路径
        output_path: 输出 JSONL 文件路径
        detector: 检测器实例
        show_progress: 是否显示进度条
        
    Returns:
        处理的图片数量
    """
    # 收集所有图片路径
    print(f"\n正在扫描文件夹: {folder_path}")
    image_paths = list(find_all_images(folder_path))
    
    if not image_paths:
        print(f"警告: 在 {folder_path} 中未找到图片")
        return 0
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理并保存结果
    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        iterator = tqdm(image_paths, desc=f"处理 {folder_path.name}", disable=not show_progress)
        
        for image_path in iterator:
            try:
                result = detector.detect(str(image_path))
                # 将路径转换为相对于 local_data 的相对路径
                result.image_path = str(image_path.relative_to(folder_path.parent))
                
                # 写入 JSONL
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                # 记录错误但继续处理
                error_result = DetectionResult(
                    image_path=str(image_path),
                    image_width=0,
                    image_height=0,
                    error=str(e)
                )
                f.write(json.dumps(error_result.to_dict(), ensure_ascii=False) + '\n')
                processed_count += 1
    
    print(f"结果已保存到: {output_path}")
    return processed_count


def main():
    """主函数"""
    # 项目路径
    project_root = Path(__file__).parent.parent.parent
    local_data = project_root / "local_data"
    
    # 要处理的文件夹
    folders_to_process = [
        # ("Album_A", local_data / "Album_A"),
        ("univer-light", local_data / "univer-light"),
    ]
    
    # 初始化检测器
    print("正在初始化检测器...")
    detector = FaceBodyDetector()
    print("检测器初始化完成")
    
    # 处理每个文件夹
    total_processed = 0
    for folder_name, folder_path in folders_to_process:
        output_path = local_data / f"{folder_name}_detect_result.jsonl"
        count = process_folder(folder_path, output_path, detector)
        total_processed += count
    
    print(f"\n全部处理完成! 共处理 {total_processed} 张图片")


if __name__ == "__main__":
    main()

