"""
测试脚本：随机输出5张 crop_and_blend 结果到可视化目录
"""

import cv2
from pathlib import Path
from crop_and_blend import load_config, get_random_samples


def main():
    """生成5张测试样本"""
    # 加载配置
    config = load_config()
    
    # 设置输出目录
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "local_data" / "visualization" / "crop_and_blend_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 获取5个随机样本
    samples = get_random_samples(n=5, config=config)
    
    print(f"成功处理 {len(samples)} 个样本")
    
    # 保存结果
    for i, (record, image) in enumerate(samples):
        output_path = output_dir / f"sample_{i+1}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"已保存: {output_path}")
        print(f"  原图: {record['image_path']}")
        print(f"  尺寸: {image.shape[1]}x{image.shape[0]}")
        if 'backup_face_bbox' in record and record['backup_face_bbox']:
            print(f"  包含 face_bbox: 是")
        else:
            print(f"  包含 face_bbox: 否")
        print()


if __name__ == "__main__":
    main()

