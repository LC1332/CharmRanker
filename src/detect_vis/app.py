"""
检测结果交互式可视化应用
允许用户一张一张查看检测结果，用不同颜色渲染人脸和人体框
"""

import json
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, Response, request
from typing import Optional

# target_folder = "Album_A"
target_folder = "univer-light"

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA = PROJECT_ROOT / "local_data"

app = Flask(__name__, template_folder="templates")


class DetectionVisualizer:
    """检测结果可视化器"""
    
    def __init__(self, jsonl_path: str, image_base_dir: str):
        """
        初始化可视化器
        
        Args:
            jsonl_path: 检测结果 JSONL 文件路径
            image_base_dir: 图片基础目录
        """
        self.jsonl_path = Path(jsonl_path)
        self.image_base_dir = Path(image_base_dir)
        self.results = []
        self._load_results()
    
    def _load_results(self):
        """加载检测结果"""
        if not self.jsonl_path.exists():
            print(f"警告: 检测结果文件不存在: {self.jsonl_path}")
            return
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e}")
        
        print(f"已加载 {len(self.results)} 条检测结果")
    
    def get_total_count(self) -> int:
        """获取总数"""
        return len(self.results)
    
    def get_result(self, index: int) -> Optional[dict]:
        """获取指定索引的检测结果"""
        if 0 <= index < len(self.results):
            return self.results[index]
        return None
    
    def get_image_path(self, index: int) -> Optional[Path]:
        """获取图片完整路径"""
        result = self.get_result(index)
        if result and 'image_path' in result:
            return self.image_base_dir / result['image_path']
        return None
    
    def draw_detection(self, image: np.ndarray, result: dict, 
                       draw_face: bool = True, draw_body: bool = True,
                       draw_keypoints: bool = True, draw_backup_face: bool = True) -> np.ndarray:
        """
        在图片上绘制检测结果
        
        Args:
            image: BGR 格式图片
            result: 检测结果字典
            draw_face: 是否绘制人脸框
            draw_body: 是否绘制人体框
            draw_keypoints: 是否绘制人脸关键点
            draw_backup_face: 是否绘制备选人脸框
            
        Returns:
            绘制后的图片
        """
        image = image.copy()
        height, width = image.shape[:2]
        
        # 绘制人脸框（青色 - Cyan）
        if draw_face and 'face_bbox' in result and result['face_bbox']:
            bbox = result['face_bbox']
            x = int(bbox['x_min'] * width)
            y = int(bbox['y_min'] * height)
            w = int(bbox['width'] * width)
            h = int(bbox['height'] * height)
            
            # 青色框，线宽3
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)
            
            # 绘制置信度标签
            if 'face_confidence' in result and result['face_confidence']:
                label = f"Face: {result['face_confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (x, y - label_size[1] - 10), 
                             (x + label_size[0], y), (255, 255, 0), -1)
                cv2.putText(image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 绘制人脸关键点（蓝色）
        if draw_keypoints and 'face_keypoints' in result and result['face_keypoints']:
            kp = result['face_keypoints']
            keypoint_names = {
                'right_eye': 'RE',
                'left_eye': 'LE', 
                'nose_tip': 'N',
                'mouth_center': 'M',
                'right_ear_tragion': 'RET',
                'left_ear_tragion': 'LET'
            }
            
            for key, name in keypoint_names.items():
                if key in kp:
                    px, py = kp[key]
                    x = int(px * width)
                    y = int(py * height)
                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(image, name, (x + 8, y - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 绘制人体框（橙色 - Orange）
        if draw_body and 'body_bbox' in result and result['body_bbox']:
            bbox = result['body_bbox']
            x = int(bbox['x_min'] * width)
            y = int(bbox['y_min'] * height)
            w = int(bbox['width'] * width)
            h = int(bbox['height'] * height)
            
            # 橙色框，线宽3
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 3)
            
            # 绘制置信度标签
            if 'body_confidence' in result and result['body_confidence']:
                label = f"Body: {result['body_confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 标签放在框下方
                label_y = y + h + label_size[1] + 10
                if label_y > height:
                    label_y = y - 5
                cv2.rectangle(image, (x, label_y - label_size[1] - 5), 
                             (x + label_size[0], label_y + 5), (0, 165, 255), -1)
                cv2.putText(image, label, (x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 绘制备选人脸框（洋红色 - Magenta）
        if draw_backup_face and 'backup_face_bbox' in result and result['backup_face_bbox']:
            bbox = result['backup_face_bbox']
            x = int(bbox['x_min'] * width)
            y = int(bbox['y_min'] * height)
            w = int(bbox['width'] * width)
            h = int(bbox['height'] * height)
            
            # 洋红色框，线宽3，虚线效果通过绘制多个小矩形模拟
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
            
            # 绘制置信度标签
            if 'backup_face_confidence' in result and result['backup_face_confidence']:
                label = f"Backup: {result['backup_face_confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 标签放在框右侧
                label_x = x + w + 5
                label_y = y + label_size[1] + 5
                if label_x + label_size[0] > width:
                    label_x = x - label_size[0] - 5
                cv2.rectangle(image, (label_x, label_y - label_size[1] - 5), 
                             (label_x + label_size[0], label_y + 5), (255, 0, 255), -1)
                cv2.putText(image, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制备选人脸关键点（洋红色）
        if draw_backup_face and draw_keypoints and 'backup_face_keypoints' in result and result['backup_face_keypoints']:
            kp = result['backup_face_keypoints']
            keypoint_names = {
                'right_eye': 'RE',
                'left_eye': 'LE', 
                'nose_tip': 'N',
                'mouth_center': 'M',
                'right_ear_tragion': 'RET',
                'left_ear_tragion': 'LET'
            }
            
            for key, name in keypoint_names.items():
                if key in kp:
                    px, py = kp[key]
                    x = int(px * width)
                    y = int(py * height)
                    cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
                    cv2.putText(image, name, (x + 8, y - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return image
    
    def get_visualized_image(self, index: int, 
                             draw_face: bool = True, 
                             draw_body: bool = True,
                             draw_keypoints: bool = True,
                             draw_backup_face: bool = True) -> Optional[np.ndarray]:
        """
        获取可视化后的图片
        
        Args:
            index: 图片索引
            draw_face: 是否绘制人脸框
            draw_body: 是否绘制人体框
            draw_keypoints: 是否绘制人脸关键点
            draw_backup_face: 是否绘制备选人脸框
            
        Returns:
            可视化后的图片（BGR格式），如果失败返回 None
        """
        result = self.get_result(index)
        if result is None:
            return None
        
        image_path = self.get_image_path(index)
        print(f"image_path: {image_path}")
        if image_path is None or not image_path.exists():
            print(f"图片不存在: {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        return self.draw_detection(image, result, draw_face, draw_body, draw_keypoints, draw_backup_face)


# 全局可视化器实例
visualizer: Optional[DetectionVisualizer] = None


def get_folder_param() -> str:
    """从请求参数获取 folder，默认使用全局 target_folder"""
    return request.args.get('folder', target_folder)


def get_visualizer(folder: str) -> DetectionVisualizer:
    """获取或创建可视化器实例"""
    global visualizer
    
    jsonl_path = LOCAL_DATA / f"{folder}_detect_result.jsonl"
    # image_path 字段已经包含了 target_folder 前缀，所以这里直接用 LOCAL_DATA
    image_base_dir = LOCAL_DATA
    
    if visualizer is None or visualizer.jsonl_path != jsonl_path:
        visualizer = DetectionVisualizer(str(jsonl_path), str(image_base_dir))
    
    return visualizer


@app.route('/')
def index():
    """主页"""
    folder = get_folder_param()
    vis = get_visualizer(folder)
    return render_template('viewer.html', 
                         total=vis.get_total_count(),
                         target_folder=folder)


@app.route('/api/info')
def get_info():
    """获取当前数据集信息"""
    folder = get_folder_param()
    vis = get_visualizer(folder)
    return jsonify({
        'total': vis.get_total_count(),
        'target_folder': folder
    })


@app.route('/api/result/<int:index>')
def get_result(index: int):
    """获取指定索引的检测结果"""
    folder = get_folder_param()
    vis = get_visualizer(folder)
    result = vis.get_result(index)
    
    if result is None:
        return jsonify({'error': '索引超出范围'}), 404
    
    return jsonify({
        'index': index,
        'total': vis.get_total_count(),
        'result': result
    })


@app.route('/api/image/<int:index>')
def get_image(index: int):
    """获取可视化后的图片"""
    folder = get_folder_param()
    draw_face = request.args.get('face', 'true').lower() == 'true'
    draw_body = request.args.get('body', 'true').lower() == 'true'
    draw_keypoints = request.args.get('keypoints', 'true').lower() == 'true'
    draw_backup_face = request.args.get('backup_face', 'true').lower() == 'true'
    
    vis = get_visualizer(folder)
    image = vis.get_visualized_image(index, draw_face, draw_body, draw_keypoints, draw_backup_face)
    
    if image is None:
        # 返回一个占位图
        placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Image not found", (150, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    # 编码为 JPEG
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/original/<int:index>')
def get_original_image(index: int):
    """获取原始图片（不带标注）"""
    folder = get_folder_param()
    vis = get_visualizer(folder)
    
    image_path = vis.get_image_path(index)
    if image_path is None or not image_path.exists():
        return jsonify({'error': '图片不存在'}), 404
    
    with open(image_path, 'rb') as f:
        return Response(f.read(), mimetype='image/jpeg')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检测结果交互式可视化')
    parser.add_argument('--folder', type=str, default= 'Album_A',
                       help='目标文件夹名称 (默认: Album_A)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务端口 (默认: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='服务主机 (默认: 127.0.0.1)')
    
    args = parser.parse_args()
    
    # 更新全局默认 folder
    target_folder = args.folder
    
    # 预加载可视化器 
    vis = get_visualizer(args.folder)
    print(f"\n数据集: {args.folder}")
    print(f"检测结果数量: {vis.get_total_count()}")
    print(f"\n启动服务器: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务器\n")
    
    app.run(host=args.host, port=args.port, debug=False)

