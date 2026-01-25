"""
Gemini 批量分类脚本

功能：
- 使用并发请求批量处理所有未分类的数据
- 支持断点续传（跳过已处理的图片）
- 显示进度条
- 自动处理错误和重试
"""

from __future__ import annotations

import json
import os
import sys
import time
import base64
import requests
from pathlib import Path
from typing import Set, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 加载环境变量
load_dotenv()

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"

INPUT_JSONL = LOCAL_DATA_DIR / "crop_log_with_face_and_body.jsonl"
CROP_OUTPUT_DIR = LOCAL_DATA_DIR / "crop_output"
OUTPUT_JSONL = LOCAL_DATA_DIR / "crop_classify.jsonl"
FAILED_JSONL = LOCAL_DATA_DIR / "fail_to_classify.jsonl"

# API 配置
LUMOS_API = os.getenv("LUMOS_API")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")
GEMINI_MODEL = "gemini-3-flash-preview"

# 并发配置
MAX_WORKERS = 10  # 并发数
BATCH_SIZE = 100  # 每批处理数量，用于定期保存进度
REQUEST_TIMEOUT = 120  # 请求超时时间（秒）

# 分类 Prompt
CLASSIFICATION_PROMPT = """Based on the input image, analyze the person highlighted by the RED bounding box and determine their attributes.

Please output a JSON object with the following fields:

- "analysis": A brief analysis describing whether the red box clearly identifies a primary subject, whether the person appears to be Asian, their gender, and any other relevant observations.

- "gender": Output "male" for male, "female" for female. If it's a false detection or cannot be determined with confidence, output "unpredictable".

- "if_asian": Output "yes" if the person appears to be Asian (East Asian, Southeast Asian, etc.), output "no" if they appear to be non-Asian, output "uncertain" if it cannot be determined.

- "if_ambiguous": Whether the red bounding box clearly identifies exactly one person. Output "no" if the box primarily contains one person (even if other people partially appear at the edges). Output "yes" if the bounding box clearly contains two or more complete persons.

- "if_correct_face": Whether the GREEN face bounding box (if present) belongs to the person highlighted by the red box. Output "yes" if it matches, "no" if it doesn't match, "no_face_box" if there is no green face box visible.

- "if_frontal": Whether the person is facing the camera (frontal or side view where facial features are visible). Output "yes" if the face and facial features (eyes, nose, mouth) can be seen. Output "no" if the person is facing away or the face is not visible.

- "false_alarm": Whether the red box is a false detection (no person inside). Output "yes" if the red box does NOT contain any person (false alarm). Output "no" if there IS a person inside the red box.

Output ONLY the JSON object, no additional text or markdown formatting."""

# 用于线程安全的文件写入
write_lock = Lock()


def load_processed_filenames() -> Set[str]:
    """加载已处理的文件名集合（包括成功和失败的）"""
    processed = set()
    
    # 加载成功的
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
    
    # 加载失败的（避免重复处理）
    if FAILED_JSONL.exists():
        with open(FAILED_JSONL, "r", encoding="utf-8") as f:
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
    
    return records


def load_image_as_base64(image_path: Path) -> str:
    """加载图片并转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    """解析 JSON 响应"""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    return json.loads(text)


def append_to_jsonl(filepath: Path, record: dict):
    """线程安全地追加记录到 JSONL 文件"""
    with write_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_single_image(record: dict, max_retries: int = 2) -> Tuple[dict, Optional[dict], Optional[str]]:
    """
    分类单张图片，支持重试
    
    Returns:
        (原始记录, 分类结果, 错误信息)
    """
    output_filename = record["output_filename"]
    image_path = CROP_OUTPUT_DIR / output_filename
    
    if not image_path.exists():
        return record, None, f"图片不存在: {image_path}"
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # 加载图片
            image_base64 = load_image_as_base64(image_path)
            ext = image_path.suffix.lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            
            # 构建请求
            url = f"{GEMINI_BASE_URL}/v1/models/{GEMINI_MODEL}:generateContent"
            headers = {
                "Authorization": f"Bearer {LUMOS_API}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                        {"text": CLASSIFICATION_PROMPT}
                    ]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            result_data = response.json()
            result_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
            classify_result = parse_json_response(result_text)
            
            return record, classify_result, None
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1)  # 短暂等待后重试
                continue
    
    return record, None, last_error


def run_batch_classify(pending_records: list, max_workers: int = MAX_WORKERS):
    """
    批量并发分类
    """
    total = len(pending_records)
    success_count = 0
    fail_count = 0
    
    print(f"\n🚀 开始批量分类 (并发数: {max_workers}, 总数: {total})")
    print("=" * 60)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(classify_single_image, record): i 
            for i, record in enumerate(pending_records)
        }
        
        # 处理完成的任务
        for future in as_completed(futures):
            idx = futures[future]
            record, classify_result, error = future.result()
            output_filename = record["output_filename"]
            
            if classify_result is not None:
                # 成功 - 写入结果文件
                result_record = record.copy()
                result_record["classify_result"] = classify_result
                append_to_jsonl(OUTPUT_JSONL, result_record)
                success_count += 1
            else:
                # 失败 - 写入失败文件
                failed_record = record.copy()
                failed_record["error"] = error
                append_to_jsonl(FAILED_JSONL, failed_record)
                fail_count += 1
            
            # 显示进度
            completed = success_count + fail_count
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            
            # 每处理一定数量显示进度
            if completed % 10 == 0 or completed == total:
                print(f"\r⏳ 进度: {completed}/{total} ({100*completed/total:.1f}%) | "
                      f"✓ {success_count} ✗ {fail_count} | "
                      f"速度: {rate:.1f}/s | ETA: {eta:.0f}s", end="", flush=True)
    
    print()  # 换行
    
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"📊 处理完成!")
    print(f"   ✓ 成功: {success_count}")
    print(f"   ✗ 失败: {fail_count}")
    print(f"   ⏱️ 耗时: {elapsed_total:.1f}s")
    print(f"   📈 速度: {(success_count + fail_count) / elapsed_total:.1f} 张/秒")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Gemini 批量分类")
    print("=" * 60)
    
    # 检查 API KEY
    if not LUMOS_API:
        print("❌ 请在 .env 文件中设置 LUMOS_API")
        sys.exit(1)
    
    print(f"✓ API 已配置")
    print(f"✓ 并发数: {MAX_WORKERS}")
    
    # 加载数据
    print("\n📂 加载数据...")
    processed_filenames = load_processed_filenames()
    print(f"   已处理: {len(processed_filenames)} 条")
    
    all_records = load_input_records()
    print(f"   总记录: {len(all_records)} 条")
    
    # 过滤出待处理的记录
    pending_records = [
        r for r in all_records
        if r.get("output_filename") not in processed_filenames
    ]
    print(f"   待处理: {len(pending_records)} 条")
    
    if not pending_records:
        print("\n✅ 所有图片都已处理完成!")
        return
    
    # 确认开始
    print(f"\n⚠️ 即将处理 {len(pending_records)} 条记录")
    print("   按 Ctrl+C 可随时中断（已处理的结果会保存）")
    
    try:
        # 开始批量处理
        run_batch_classify(pending_records, max_workers=MAX_WORKERS)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，已处理的结果已保存")
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print(f"📄 结果文件: {OUTPUT_JSONL}")
    if FAILED_JSONL.exists():
        print(f"📄 失败记录: {FAILED_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
