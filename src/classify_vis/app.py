"""
分类结果可视化应用
- Tab 1: 展示各category的边缘分布
- Tab 2: 筛选功能 + 随机展示图片
"""

import json
import random
from pathlib import Path
from collections import Counter
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# 配置路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "local_data" / "crop_classify.jsonl"
CROP_OUTPUT_DIR = BASE_DIR / "local_data" / "crop_output"

# 全局数据存储
data_records = []
category_fields = ["gender", "if_asian", "if_ambiguous", "if_correct_face", "if_frontal", "false_alarm"]


def load_data():
    """加载JSONL数据"""
    global data_records
    data_records = []
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                classify_result = record.get("classify_result", {})
                
                # 处理classify_result可能是列表的情况
                if isinstance(classify_result, list) and len(classify_result) > 0:
                    classify_result = classify_result[0]
                
                if isinstance(classify_result, dict):
                    data_records.append({
                        "output_filename": record.get("output_filename"),
                        "original_path": record.get("original_path"),
                        **{field: classify_result.get(field, "unknown") for field in category_fields}
                    })
            except json.JSONDecodeError:
                continue
    
    print(f"加载了 {len(data_records)} 条记录")


def get_distribution():
    """计算各category的边缘分布"""
    distributions = {}
    
    for field in category_fields:
        counter = Counter(record.get(field, "unknown") for record in data_records)
        distributions[field] = dict(counter)
    
    return distributions


def filter_records(filters: dict):
    """根据筛选条件过滤记录"""
    filtered = data_records
    
    for field, value in filters.items():
        if value and value != "all":
            filtered = [r for r in filtered if r.get(field) == value]
    
    return filtered


@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/api/distributions")
def api_distributions():
    """获取分布数据"""
    distributions = get_distribution()
    return jsonify({
        "total": len(data_records),
        "distributions": distributions
    })


@app.route("/api/filter_options")
def api_filter_options():
    """获取所有可用的筛选选项"""
    options = {}
    for field in category_fields:
        values = set(record.get(field, "unknown") for record in data_records)
        options[field] = sorted(list(values))
    return jsonify(options)


@app.route("/api/random_samples")
def api_random_samples():
    """根据筛选条件随机返回5张图片"""
    filters = {}
    for field in category_fields:
        value = request.args.get(field)
        if value and value != "all":
            filters[field] = value
    
    filtered = filter_records(filters)
    
    # 随机抽取5张
    sample_size = min(5, len(filtered))
    samples = random.sample(filtered, sample_size) if filtered else []
    
    return jsonify({
        "total_filtered": len(filtered),
        "samples": samples
    })


@app.route("/images/<filename>")
def serve_image(filename):
    """提供图片服务"""
    return send_from_directory(CROP_OUTPUT_DIR, filename)


if __name__ == "__main__":
    print("正在加载数据...")
    load_data()
    print(f"数据加载完成，共 {len(data_records)} 条记录")
    app.run(debug=True, port=5002)

