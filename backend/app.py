"""
雾天图像目标检测Web应用后端
"""
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime
from dehaze_api import DehazeDetector

# ==================== 初始化 ====================
app = Flask(__name__, static_folder=None)
CORS(app)

# 配置文件
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'models/yolov8n.pt'

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化检测器
print("🌫️ 初始化雾天目标检测系统...")
detector = DehazeDetector(MODEL_PATH)

# ==================== 路由 ====================

@app.route('/')
def index():
    """前端页面（使用 mist-frontend 构建产物）"""
    return send_from_directory('../mist-frontend/dist', 'index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """提供前端构建后的静态资源"""
    return send_from_directory('../mist-frontend/dist/assets', filename)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """上传并处理图像"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})

    try:
        # 读取图像为numpy数组
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_np is None:
            return jsonify({'success': False, 'error': '无法读取图像'})

        print(f"📸 处理图片: {file.filename}, 尺寸: {image_np.shape}")

        # 处理图像 - 传递numpy数组和文件名
        result = detector.process(image_np, file.filename)

        return jsonify({
            'success': True,
            'original': f'/api/image/{result["original_filename"]}',
            'dehazed': f'/api/image/{result["dehazed_filename"]}',
            'detected': f'/api/image/{result["detected_filename"]}',
            'num_objects': result['num_objects'],
            'output_dir': result['output_dir'],
            'timestamp': result['timestamp']
        })

    except Exception as e:
        print(f"❌ 处理出错: {str(e)}")  # 添加详细错误日志
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detections/<timestamp>/<basename>')
def get_detections(timestamp, basename):
    """读取并返回检测结果TXT为结构化数据

    参数:
        timestamp: 处理输出的时间戳目录，例如 20251208_181942
        basename: 原始文件的基础名（不含扩展名），例如 school夕阳马路(1) final
    返回:
        { success: true, list: [ { index, label, confidence } ] }
    """
    try:
        txt_path = os.path.join(OUTPUT_FOLDER, timestamp, f"detection_results_{basename}.txt")
        if not os.path.exists(txt_path):
            return jsonify({'success': True, 'list': []})

        detections = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 匹配形如: "1. car: 置信度 0.86"
                if "." in line and "置信度" in line:
                    try:
                        left, conf_part = line.split(':', 1)
                        idx_str, label = left.split('.', 1)
                        idx = int(idx_str.strip())
                        label = label.strip()
                        # 提取数值部分
                        conf_str = conf_part.split('置信度', 1)[1].strip()
                        confidence = float(conf_str)
                        detections.append({
                            'index': idx,
                            'label': label,
                            'confidence': confidence
                        })
                    except Exception:
                        # 跳过无法解析的行
                        continue

        return jsonify({'success': True, 'list': detections})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/<path:filename>')
def get_image(filename):
    """获取处理后的图像"""
    # 检查文件在哪个目录
    possible_paths = [
        os.path.join(OUTPUT_FOLDER, filename),
        os.path.join(UPLOAD_FOLDER, filename)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return send_file(path)

    return jsonify({'success': False, 'error': '文件不存在'}), 404

@app.route('/<path:path>')
def spa_fallback(path):
    """前端单页应用路由回退"""
    root = '../mist-frontend/dist'
    full = os.path.join(root, path)
    if os.path.exists(full):
        return send_from_directory(root, path)
    return send_from_directory(root, 'index.html')

@app.route('/api/history')
def get_history():
    """获取处理历史"""
    try:
        history = []
        if os.path.exists(OUTPUT_FOLDER):
            # 获取按时间排序的文件夹
            folders = sorted([f for f in os.listdir(OUTPUT_FOLDER)
                            if os.path.isdir(os.path.join(OUTPUT_FOLDER, f))],
                           reverse=True)

            for folder in folders[:10]:  # 最近10条
                folder_path = os.path.join(OUTPUT_FOLDER, folder)
                # 查找检测结果图
                for file in os.listdir(folder_path):
                    if file.startswith('3_detection_'):
                        history.append({
                            'id': folder,
                            'time': folder,
                            'image': f'/api/image/{folder}/{file}',
                            'preview': file.replace('3_detection_', '原始: ')
                        })
                        break

        return jsonify({'success': True, 'history': history})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test')
def test():
    """测试接口"""
    return jsonify({
        'success': True,
        'message': '雾天目标检测系统API运行正常',
        'version': '1.0.0'
    })

# ==================== 启动应用 ====================

if __name__ == '__main__':
    print("🚀 服务器启动在: http://localhost:5000")
    print(f"📁 上传目录: {UPLOAD_FOLDER}/")
    print(f"📁 输出目录: {OUTPUT_FOLDER}/")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
