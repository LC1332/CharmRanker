"""
动态可视化应用 - 查看动态比较结果

功能:
- Tab 1: 展示 Top 10 节点的照片，包括 mu 和 sigma
- Tab 2: 展示最近 5 个比较结果
- Tab 3: 展示 Anchor 节点（各分位数样本）
"""

import json
import yaml
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file
import sys

# 添加项目根目录到path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.EloManager.elo_manager import EloManager

app = Flask(__name__)

# 全局配置
config = None
result_folder = None
img_folder = None
elo_manager = None


def load_config():
    """加载配置文件"""
    global config, result_folder, img_folder
    config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    result_folder = BASE_DIR / config['result_folder']
    img_folder = BASE_DIR / config['img_folder']
    
    print(f"配置加载完成:")
    print(f"  结果文件夹: {result_folder}")
    print(f"  图片文件夹: {img_folder}")


def load_elo_manager():
    """加载 EloManager"""
    global elo_manager
    
    elo_config_path = result_folder / "elo_config.yaml"
    if elo_config_path.exists():
        elo_manager = EloManager(str(elo_config_path))
    else:
        elo_manager = EloManager()
    
    elo_manager.load(str(result_folder))
    print(f"EloManager 加载完成，共 {len(elo_manager.nodes)} 个节点")


def get_top_nodes(n: int = 10):
    """
    获取 Top N 节点
    
    Returns:
        list of dict，包含节点信息和分数
    """
    nodes_path = result_folder / "nodes.json"
    
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # 按 mu 降序排序
    sorted_nodes = sorted(nodes, key=lambda x: x['score']['mu'], reverse=True)
    
    # 取 Top N
    top_nodes = sorted_nodes[:n]
    
    # 处理图片路径
    result = []
    for node in top_nodes:
        # 从 image_path 中提取文件名
        image_path = Path(node['image_path'])
        filename = image_path.name
        
        result.append({
            'name': node['name'],
            'filename': filename,
            'mu': round(node['score']['mu'], 2),
            'sigma': round(node['score']['sigma'], 2)
        })
    
    return result


def get_recent_comparisons(n: int = 5):
    """
    获取最近 N 个比较结果
    
    Returns:
        list of dict，包含比较信息
    """
    response_logs_path = result_folder / "response_logs.jsonl"
    
    if not response_logs_path.exists():
        return []
    
    # 读取所有日志
    logs = []
    with open(response_logs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # 只取成功的比较
    successful_logs = [log for log in logs if log.get('success', False)]
    
    # 取最近 N 个
    recent_logs = successful_logs[-n:][::-1]  # 倒序，最新的在前
    
    # 处理结果
    result = []
    for log in recent_logs:
        triplet = log.get('triplet', [])
        image_paths = log.get('image_paths', [])
        parsed_response = log.get('parsed_response', {}) or {}
        
        # 获取最attractive和least attractive
        most_attractive = parsed_response.get('most_attractive', '')
        least_attractive = parsed_response.get('least_attractive', '')
        analysis = parsed_response.get('analysis', '')
        
        # 转换 A/B/C 到索引
        label_to_idx = {'A': 0, 'B': 1, 'C': 2}
        most_idx = label_to_idx.get(most_attractive, -1)
        least_idx = label_to_idx.get(least_attractive, -1)
        
        # 构造图片信息
        images = []
        for i, (name, path) in enumerate(zip(triplet, image_paths)):
            filename = Path(path).name
            label = chr(ord('A') + i)
            is_most = (i == most_idx)
            is_least = (i == least_idx)
            
            images.append({
                'name': name,
                'filename': filename,
                'label': label,
                'is_most_attractive': is_most,
                'is_least_attractive': is_least
            })
        
        result.append({
            'timestamp': log.get('timestamp', ''),
            'images': images,
            'analysis': analysis,
            'most_attractive_label': most_attractive,
            'least_attractive_label': least_attractive
        })
    
    return result


def get_anchors():
    """
    获取各分位数的 Anchor 节点
    
    Returns:
        list of dict，包含 anchor 信息
    """
    global elo_manager
    
    # 重新加载以获取最新数据
    load_elo_manager()
    
    anchors = elo_manager.get_anchors()
    
    result = []
    for anchor in anchors:
        image_path = Path(anchor.get('image_path', ''))
        filename = image_path.name
        
        result.append({
            'name': anchor.get('name', ''),
            'filename': filename,
            'percentile': round(anchor.get('percentile', 0), 1),
            'mu': round(anchor.get('mu', 0), 2),
            'sigma': round(anchor.get('sigma', 0), 2),
            'compare_count': anchor.get('compare_count', 0)
        })
    
    # 按分位数排序
    result.sort(key=lambda x: x['percentile'])
    
    return result


@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/api/top_nodes")
def api_top_nodes():
    """获取 Top 10 节点"""
    try:
        nodes = get_top_nodes(10)
        return jsonify({
            'success': True,
            'nodes': nodes
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route("/api/recent_comparisons")
def api_recent_comparisons():
    """获取最近 5 个比较结果"""
    try:
        comparisons = get_recent_comparisons(5)
        return jsonify({
            'success': True,
            'comparisons': comparisons
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route("/api/anchors")
def api_anchors():
    """获取 Anchor 节点"""
    try:
        anchors = get_anchors()
        return jsonify({
            'success': True,
            'anchors': anchors
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route("/images/<filename>")
def serve_image(filename):
    """提供图片服务"""
    image_path = img_folder / filename
    if image_path.exists():
        return send_file(str(image_path))
    else:
        return "Image not found", 404


if __name__ == "__main__":
    print("正在加载配置...")
    load_config()
    
    print("正在加载 EloManager...")
    load_elo_manager()
    
    port = config.get('server', {}).get('port', 5015)
    debug = config.get('server', {}).get('debug', True)
    
    print(f"\n启动服务器，端口: {port}")
    app.run(debug=debug, port=port)
