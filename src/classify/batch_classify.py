"""
批量分类处理脚本 - 对 crop 后的图片进行批量属性分类

功能：
- 读取 crop_log_with_face_and_body.jsonl 中的图片记录
- 对每张图片进行性别、亚洲人等属性分类
- 支持断点续传（跳过已处理的图片）
- 支持失败重试（每张图最多重试1次）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

from tqdm import tqdm

from classify import classify_gender_and_asian

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"

INPUT_JSONL = LOCAL_DATA_DIR / "crop_log_with_face_and_body.jsonl"
CROP_OUTPUT_DIR = LOCAL_DATA_DIR / "crop_output"
OUTPUT_JSONL = LOCAL_DATA_DIR / "crop_classify.jsonl"
FAILED_JSONL = LOCAL_DATA_DIR / "fail_to_classify.jsonl"


def load_processed_filenames() -> Set[str]:
    """加载已处理的文件名集合，用于断点续传"""
    processed = set()
    
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "output_filename" in record:
                        processed.add(record["output_filename"])
                except json.JSONDecodeError:
                    continue
    
    print(f"📋 已处理的图片数量: {len(processed)}")
    return processed


def load_input_records() -> list:
    """加载输入的 jsonl 记录"""
    records = []
    
    if not INPUT_JSONL.exists():
        print(f"❌ 输入文件不存在: {INPUT_JSONL}")
        sys.exit(1)
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"📂 总共读取的记录数量: {len(records)}")
    return records


def process_single_image(record: dict, max_retries: int = 1) -> Tuple[Optional[dict], Optional[str]]:
    """
    处理单张图片，支持重试
    
    Returns:
        (成功结果, None) 或 (None, 错误信息)
    """
    output_filename = record.get("output_filename")
    if not output_filename:
        return None, "缺少 output_filename 字段"
    
    image_path = CROP_OUTPUT_DIR / output_filename
    
    if not image_path.exists():
        return None, f"图片文件不存在: {image_path}"
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # 调用分类函数
            classify_result = classify_gender_and_asian(image_path)
            
            # 合并原始记录和分类结果
            result = record.copy()
            result["classify_result"] = classify_result
            
            return result, None
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                continue  # 重试
    
    return None, last_error


def append_to_jsonl(filepath: Path, record: dict):
    """追加一条记录到 jsonl 文件"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """主函数 - 批量处理所有图片"""
    print("=" * 60)
    print("🔍 批量人物属性分类")
    print("=" * 60)
    
    # 加载已处理的文件名
    processed_filenames = load_processed_filenames()
    
    # 加载输入记录
    all_records = load_input_records()
    
    # 过滤出待处理的记录
    pending_records = [
        r for r in all_records
        if r.get("output_filename") not in processed_filenames
    ]
    
    print(f"⏳ 待处理的图片数量: {len(pending_records)}")
    
    if not pending_records:
        print("✅ 所有图片都已处理完成!")
        return
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 批量处理
    for record in tqdm(pending_records, desc="分类处理", unit="张"):
        result, error = process_single_image(record, max_retries=1)
        
        if result is not None:
            # 成功 - 追加到输出文件
            append_to_jsonl(OUTPUT_JSONL, result)
            success_count += 1
        else:
            # 失败 - 追加到失败文件
            failed_record = record.copy()
            failed_record["error"] = error
            append_to_jsonl(FAILED_JSONL, failed_record)
            fail_count += 1
    
    # 打印统计
    print("\n" + "=" * 60)
    print("📊 处理统计")
    print("=" * 60)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📄 输出文件: {OUTPUT_JSONL}")
    if fail_count > 0:
        print(f"📄 失败记录: {FAILED_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()

