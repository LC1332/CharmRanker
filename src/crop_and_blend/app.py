"""
Crop and Blend 可视化应用
每次点击按钮可随机显示5张处理后的图片
"""

import cv2
import io
import base64
from flask import Flask, render_template_string, jsonify
from crop_and_blend import load_config, get_random_samples, load_valid_records

app = Flask(__name__)

# 全局配置和缓存
_config = None
_total_records = 0


def get_config():
    """获取配置（懒加载）"""
    global _config, _total_records
    if _config is None:
        _config = load_config()
        _total_records = len(load_valid_records(_config))
    return _config


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop & Blend 可视化</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #8892b0;
            font-size: 1.1rem;
        }
        
        .stats {
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stats span {
            color: #00d9ff;
            font-weight: 500;
        }
        
        .controls {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .btn-refresh {
            background: linear-gradient(135deg, #00d9ff 0%, #00ff88 100%);
            color: #1a1a2e;
            border: none;
            padding: 16px 48px;
            font-size: 1.2rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 217, 255, 0.3);
        }
        
        .btn-refresh:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 217, 255, 0.5);
        }
        
        .btn-refresh:active {
            transform: translateY(0);
        }
        
        .btn-refresh:disabled {
            background: #555;
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
        }
        
        .image-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            border-color: rgba(0, 217, 255, 0.3);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .image-wrapper {
            position: relative;
            width: 100%;
            padding-top: 100%;
            background: #0a0a0f;
        }
        
        .image-wrapper img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .image-info {
            padding: 15px;
        }
        
        .image-path {
            font-size: 0.8rem;
            color: #8892b0;
            word-break: break-all;
            line-height: 1.4;
        }
        
        .image-meta {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.75rem;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .meta-item .label {
            color: #666;
        }
        
        .meta-item .value {
            color: #00d9ff;
            font-weight: 500;
        }
        
        .has-face {
            color: #00ff88 !important;
        }
        
        .no-face {
            color: #ff6b6b !important;
        }
        
        .loading {
            text-align: center;
            padding: 60px;
            color: #8892b0;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 217, 255, 0.1);
            border-top-color: #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }
        
        .legend-color {
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }
        
        .legend-color.red {
            background: #ff0000;
        }
        
        .legend-color.green {
            background: #00ff00;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Crop & Blend 可视化</h1>
            <p class="subtitle">人体裁剪与背景模糊处理效果预览</p>
            <div class="stats">
                有效记录数: <span id="totalCount">{{ total }}</span>
            </div>
        </header>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color red"></div>
                <span>Body BBox (红色)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color green"></div>
                <span>Face BBox (绿色)</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-refresh" onclick="loadRandomSamples()">
                🎲 随机加载 5 张
            </button>
        </div>
        
        <div id="gallery" class="gallery">
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>点击上方按钮加载图片</p>
            </div>
        </div>
    </div>
    
    <script>
        async function loadRandomSamples() {
            const btn = document.querySelector('.btn-refresh');
            const gallery = document.getElementById('gallery');
            
            btn.disabled = true;
            btn.textContent = '⏳ 加载中...';
            
            gallery.innerHTML = `
                <div class="loading" style="grid-column: 1 / -1;">
                    <div class="loading-spinner"></div>
                    <p>正在处理图片...</p>
                </div>
            `;
            
            try {
                const response = await fetch('/api/random_samples');
                const data = await response.json();
                
                if (data.error) {
                    gallery.innerHTML = `<div class="loading" style="grid-column: 1 / -1;"><p>错误: ${data.error}</p></div>`;
                    return;
                }
                
                gallery.innerHTML = '';
                
                data.samples.forEach((sample, index) => {
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    
                    const hasFace = sample.has_face;
                    const faceClass = hasFace ? 'has-face' : 'no-face';
                    const faceText = hasFace ? '有' : '无';
                    
                    card.innerHTML = `
                        <div class="image-wrapper">
                            <img src="data:image/jpeg;base64,${sample.image}" alt="Sample ${index + 1}">
                        </div>
                        <div class="image-info">
                            <div class="image-path">${sample.path}</div>
                            <div class="image-meta">
                                <div class="meta-item">
                                    <span class="label">尺寸:</span>
                                    <span class="value">${sample.size}</span>
                                </div>
                                <div class="meta-item">
                                    <span class="label">Face:</span>
                                    <span class="value ${faceClass}">${faceText}</span>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    gallery.appendChild(card);
                });
                
            } catch (err) {
                gallery.innerHTML = `<div class="loading" style="grid-column: 1 / -1;"><p>请求失败: ${err.message}</p></div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = '🎲 随机加载 5 张';
            }
        }
        
        // 页面加载完成后自动加载一次
        window.addEventListener('load', loadRandomSamples);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页"""
    get_config()  # 确保配置已加载
    return render_template_string(HTML_TEMPLATE, total=_total_records)


@app.route('/api/random_samples')
def api_random_samples():
    """获取5个随机样本"""
    try:
        config = get_config()
        samples = get_random_samples(n=5, config=config)
        
        result = []
        for record, image in samples:
            # 将图片编码为 base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result.append({
                'path': record.get('image_path', 'unknown'),
                'image': img_base64,
                'size': f"{image.shape[1]}x{image.shape[0]}",
                'has_face': 'backup_face_bbox' in record and record['backup_face_bbox'] is not None
            })
        
        return jsonify({'samples': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/stats')
def api_stats():
    """获取统计信息"""
    config = get_config()
    return jsonify({
        'total_records': _total_records,
        'min_short_edge': config['filter']['min_body_short_edge']
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop and Blend 可视化应用')
    parser.add_argument('--port', type=int, default=5002, help='服务端口 (默认: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务主机 (默认: 127.0.0.1)')
    
    args = parser.parse_args()
    
    # 预加载配置
    config = get_config()
    print(f"\n有效记录数: {_total_records}")
    print(f"最小短边阈值: {config['filter']['min_body_short_edge']}px")
    print(f"\n启动服务器: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host=args.host, port=args.port, debug=False)

