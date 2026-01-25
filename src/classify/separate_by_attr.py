"""
根据属性过滤数据并按性别分离

过滤条件:
- if_asian = "yes"
- if_ambiguous = "no"
- if_frontal = "yes"

将符合条件的图片按gender分别保存到:
- local_data/univer_male
- local_data/univer_female

同时生成对应的jsonl文件:
- local_data/univer_male.jsonl
- local_data/univer_female.jsonl
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    # 设置路径
    project_root = Path(__file__).resolve().parent.parent.parent
    input_jsonl = project_root / "local_data" / "crop_classify.jsonl"
    source_dir = project_root / "local_data" / "crop_body_only"
    
    male_dir = project_root / "local_data" / "univer_male"
    female_dir = project_root / "local_data" / "univer_female"
    male_jsonl = project_root / "local_data" / "univer_male.jsonl"
    female_jsonl = project_root / "local_data" / "univer_female.jsonl"
    
    # 创建输出目录
    male_dir.mkdir(parents=True, exist_ok=True)
    female_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计计数
    stats = {
        "total": 0,
        "filtered_out": 0,
        "male": 0,
        "female": 0,
        "missing_file": 0,
        "duplicate": 0,
    }
    
    # 已处理的文件名集合（去重）
    processed_files = set()
    
    # 读取所有数据
    print(f"读取: {input_jsonl}")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"总共 {len(lines)} 条记录")
    
    # 打开输出文件
    with open(male_jsonl, "w", encoding="utf-8") as f_male, \
         open(female_jsonl, "w", encoding="utf-8") as f_female:
        
        for line in tqdm(lines, desc="处理中"):
            stats["total"] += 1
            
            data = json.loads(line.strip())
            classify_result = data.get("classify_result", {})
            
            # 处理 classify_result 为列表的情况
            if isinstance(classify_result, list):
                if len(classify_result) > 0:
                    classify_result = classify_result[0]
                else:
                    classify_result = {}
            
            # 检查过滤条件
            if_asian = classify_result.get("if_asian", "")
            if_ambiguous = classify_result.get("if_ambiguous", "")
            if_frontal = classify_result.get("if_frontal", "")
            
            if if_asian != "yes" or if_ambiguous != "no" or if_frontal != "yes":
                stats["filtered_out"] += 1
                continue
            
            # 获取性别和文件名
            gender = classify_result.get("gender", "")
            output_filename = data.get("output_filename", "")
            
            if not output_filename:
                stats["filtered_out"] += 1
                continue
            
            # 跳过已处理的文件（去重）
            if output_filename in processed_files:
                stats["duplicate"] += 1
                continue
            
            # 检查源文件是否存在
            source_file = source_dir / output_filename
            if not source_file.exists():
                stats["missing_file"] += 1
                continue
            
            # 根据性别处理
            if gender == "male":
                dest_file = male_dir / output_filename
                shutil.copy2(source_file, dest_file)
                f_male.write(line)
                stats["male"] += 1
                processed_files.add(output_filename)
            elif gender == "female":
                dest_file = female_dir / output_filename
                shutil.copy2(source_file, dest_file)
                f_female.write(line)
                stats["female"] += 1
                processed_files.add(output_filename)
            else:
                stats["filtered_out"] += 1
    
    # 打印统计信息
    print("\n===== 处理完成 =====")
    print(f"总记录数: {stats['total']}")
    print(f"过滤掉的: {stats['filtered_out']}")
    print(f"重复记录: {stats['duplicate']}")
    print(f"缺失文件: {stats['missing_file']}")
    print(f"男性数量: {stats['male']} -> {male_dir}")
    print(f"女性数量: {stats['female']} -> {female_dir}")
    print(f"男性JSONL: {male_jsonl}")
    print(f"女性JSONL: {female_jsonl}")


if __name__ == "__main__":
    main()

