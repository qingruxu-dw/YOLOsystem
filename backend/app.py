"""
雾天图像目标检测Web应用后端
"""
from flask import Flask, request, jsonify, send_file, send_from_directory, g
from flask_cors import CORS
import os
import cv2
import json
import numpy as np
from datetime import datetime
import sqlite3
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from dehaze_api import DehazeDetector
from fusion_api import FusionDetector

# ==================== 初始化 ====================
app = Flask(__name__, static_folder=None)
CORS(app)

# 配置文件
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'models/yolo11n.pt'
AUTH_TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 7

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

def get_db():
    db = getattr(g, '_db', None)
    if db is None:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        g._db = db
    return db

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, '_db', None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    try:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                mode TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                processed_filename TEXT,
                dehazed_filename TEXT,
                detected_filename TEXT,
                output_dir TEXT,
                params TEXT,
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.commit()
    finally:
        db.close()

def get_token_serializer():
    return URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='auth-token-v1')

def create_auth_token(user_id):
    return get_token_serializer().dumps({'uid': int(user_id)})

def verify_auth_token(token):
    try:
        data = get_token_serializer().loads(token, max_age=AUTH_TOKEN_MAX_AGE_SECONDS)
        uid = int(data.get('uid'))
        return uid
    except (BadSignature, SignatureExpired, ValueError, TypeError):
        return None

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        token = ''
        if auth.lower().startswith('bearer '):
            token = auth[7:].strip()
        uid = verify_auth_token(token) if token else None
        if not uid:
            return jsonify({'success': False, 'error': '未登录或登录已过期'}), 401
        db = get_db()
        row = db.execute('SELECT id, username, created_at FROM users WHERE id = ?', (uid,)).fetchone()
        if row is None:
            return jsonify({'success': False, 'error': '用户不存在'}), 401
        g.current_user = {'id': row['id'], 'username': row['username'], 'created_at': row['created_at']}
        return fn(*args, **kwargs)
    return wrapper

init_db()

# 初始化检测器
print("🌫️ 初始化雾天目标检测系统...")
# 默认检测器 (单模态)
default_detector = DehazeDetector(MODEL_PATH)
# 融合检测器 (多模态)
fusion_detector = FusionDetector(MODEL_PATH)

# ==================== 路由 ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''

    if len(username) < 3 or len(username) > 32:
        return jsonify({'success': False, 'error': '用户名长度需在 3-32 之间'}), 400
    if len(password) < 6 or len(password) > 128:
        return jsonify({'success': False, 'error': '密码长度需在 6-128 之间'}), 400

    password_hash = generate_password_hash(password)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db = get_db()
    try:
        cur = db.execute(
            'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
            (username, password_hash, created_at),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': '用户名已存在'}), 409

    user_id = cur.lastrowid
    token = create_auth_token(user_id)
    return jsonify({
        'success': True,
        'token': token,
        'expires_in': AUTH_TOKEN_MAX_AGE_SECONDS,
        'user': {'id': user_id, 'username': username, 'created_at': created_at},
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''

    if not username or not password:
        return jsonify({'success': False, 'error': '用户名或密码不能为空'}), 400

    db = get_db()
    row = db.execute(
        'SELECT id, username, password_hash, created_at FROM users WHERE username = ?',
        (username,),
    ).fetchone()
    if row is None:
        return jsonify({'success': False, 'error': '用户名或密码错误'}), 401
    if not check_password_hash(row['password_hash'], password):
        return jsonify({'success': False, 'error': '用户名或密码错误'}), 401

    token = create_auth_token(row['id'])
    return jsonify({
        'success': True,
        'token': token,
        'expires_in': AUTH_TOKEN_MAX_AGE_SECONDS,
        'user': {'id': row['id'], 'username': row['username'], 'created_at': row['created_at']},
    })

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def me():
    return jsonify({'success': True, 'user': g.current_user})

@app.route('/api/auth/ping', methods=['GET'])
@require_auth
def auth_ping():
    return jsonify({'success': True, 'user': g.current_user})

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

        # 获取处理模式
        mode = request.form.get('mode', 'normal')
        text_prompt = request.form.get('text_prompt', '')
        
        # 获取去雾强度 (默认为 0.95)
        try:
            dehaze_strength = float(request.form.get('dehaze_strength', 0.95))
        except ValueError:
            dehaze_strength = 0.95
            
        print(f"🔄 处理模式: {mode}, 文本引导: {text_prompt}, 去雾强度: {dehaze_strength}")

        if mode in ['fusion', 'multimodal']:
            # 使用多模态融合检测
            result = fusion_detector.process(image_np, file.filename, text_prompt=text_prompt)
        elif mode == 'baseline':
            # 使用基准检测 (直接在有雾图上检测)
            result = default_detector.process_baseline(image_np, file.filename)
        else:
            # 使用默认单模态检测 (去雾后检测)
            # 将前端传入的 strength 作为 omega 参数
            result = default_detector.process(image_np, file.filename, omega=dehaze_strength)

        # === 记录历史 ===
        try:
            auth = request.headers.get('Authorization', '')
            user_id = None
            if auth.lower().startswith('bearer '):
                token = auth[7:].strip()
                uid = verify_auth_token(token)
                if uid:
                    user_id = uid
            
            params = {
                'dehaze_strength': dehaze_strength,
                'text_prompt': text_prompt
            }

            db = get_db()
            db.execute(
                '''
                INSERT INTO history (
                    user_id, mode, original_filename, processed_filename, 
                    dehazed_filename, detected_filename, output_dir, params, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id, mode, file.filename, 
                    result.get('original_filename', ''), 
                    result.get('dehazed_filename', ''), 
                    result.get('detected_filename', ''), 
                    result.get('output_dir', ''), 
                    json.dumps(params), 
                    result.get('timestamp', '')
                )
            )
            db.commit()
        except Exception as e:
            print(f"⚠️ 记录历史失败: {str(e)}")
            # 不影响主流程

        return jsonify({
            'success': True,
            'original': f'/api/image/{result["original_filename"]}',
            'dehazed': f'/api/image/{result["dehazed_filename"]}',
            'detected': f'/api/image/{result["detected_filename"]}',
            'num_objects': result['num_objects'],
            'output_dir': result['output_dir'],
            'timestamp': result['timestamp'],
            'latency': result.get('latency', 0)
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
        mode = request.args.get('mode')
        filename = request.args.get('filename')
        
        # Determine user (optional)
        auth = request.headers.get('Authorization', '')
        user_id = None
        if auth.lower().startswith('bearer '):
            token = auth[7:].strip()
            user_id = verify_auth_token(token)
            
        db = get_db()
        
        # Comparison Mode: Get unique filenames or specific file details
        if mode == 'comparison':
            if filename:
                # Detail view: Get all 3 modes for this file
                rows = db.execute('''
                    SELECT * FROM history 
                    WHERE original_filename = ? AND mode IN ('baseline', 'normal', 'basic', 'multimodal', 'fusion')
                    ORDER BY created_at DESC
                ''', (filename,)).fetchall()
                
                # Group by mode, take latest
                results = {}
                for row in rows:
                    m = row['mode']
                    if m == 'fusion': m = 'multimodal'
                    if m == 'normal': m = 'basic'
                    if m not in results:
                        # Parse params safely
                        try:
                            params = json.loads(row['params']) if row['params'] else {}
                        except:
                            params = {}

                        results[m] = {
                            'id': row['id'],
                            'mode': row['mode'],
                            'original_filename': row['original_filename'],
                            'original': f'/api/image/{row["processed_filename"]}' if row["processed_filename"] else '',
                            'dehazed': f'/api/image/{row["dehazed_filename"]}' if row["dehazed_filename"] else '',
                            'detected': f'/api/image/{row["detected_filename"]}' if row["detected_filename"] else '',
                            'timestamp': row['timestamp'],
                            'params': params,
                            'output_dir': row['output_dir']
                        }
                        
                return jsonify({'success': True, 'comparison': results})
            else:
                # List view: Get unique filenames that have comparison data
                query = "SELECT DISTINCT original_filename, MAX(created_at) as last_time FROM history WHERE mode IN ('baseline', 'normal', 'basic', 'multimodal', 'fusion')"
                params = []
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                query += " GROUP BY original_filename ORDER BY last_time DESC LIMIT 20"
                
                rows = db.execute(query, params).fetchall()
                history = [{'filename': r['original_filename'], 'time': r['last_time']} for r in rows]
                return jsonify({'success': True, 'history': history})

        # Standard Mode (Basic / Multimodal)
        query = "SELECT * FROM history WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if mode:
            if mode == 'basic':
                query += " AND mode IN ('normal', 'basic')"
            elif mode == 'multimodal':
                query += " AND mode IN ('multimodal', 'fusion')"
            else:
                query += " AND mode = ?"
                params.append(mode)
        
        query += " ORDER BY created_at DESC LIMIT 20"
        
        rows = db.execute(query, params).fetchall()
        
        history = []
        for row in rows:
            # Parse params safely
            try:
                params = json.loads(row['params']) if row['params'] else {}
            except:
                params = {}

            history.append({
                'id': row['id'],
                'mode': row['mode'],
                'original_filename': row['original_filename'],
                'time': row['created_at'],
                'original': f'/api/image/{row["processed_filename"]}' if row["processed_filename"] else '',
                'dehazed': f'/api/image/{row["dehazed_filename"]}' if row["dehazed_filename"] else '',
                'detected': f'/api/image/{row["detected_filename"]}' if row["detected_filename"] else '',
                'timestamp': row['timestamp'],
                'params': params,
                'output_dir': row['output_dir']
            })

        return jsonify({'success': True, 'history': history})

    except Exception as e:
        print(f"Error in get_history: {e}")
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
