import cv2
import numpy as np
import os
import torch
import sys
from datetime import datetime
from typing import Tuple, Optional

# ==========================================
# 0. Path Configuration for Multibackend
# ==========================================
# 确保使用绝对路径，避免环境差异
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
multibackend_path = os.path.normpath(os.path.join(project_root, 'multibackend'))

if multibackend_path not in sys.path:
    sys.path.append(multibackend_path)

print(f"📍 [Fusion] Project Root: {project_root}")
print(f"📍 [Fusion] Multibackend Path: {multibackend_path}")

# Now we can import from multibackend
try:
    from inference_fusion import FusionInference
    print("✅ [Fusion] Successfully imported FusionInference from multibackend")
except ImportError as e:
    print(f"❌ [Fusion] Failed to import FusionInference: {e}")
    # Fallback import if needed
    try:
        from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo
        from yolosystem.coa_adapter import CoADehazer
    except ImportError:
        pass

# ==========================================
# 2. Fusion Detector (Calling InferenceInference directly)
# ==========================================

class FusionDetector:
    def __init__(self, model_path=None):
        """初始化融合检测器"""
        # 默认权重路径
        fusion_checkpoint = os.path.normpath(os.path.join(multibackend_path, 'stage2_best.pth'))
        
        if model_path is None or (isinstance(model_path, str) and model_path.endswith('.pt')):
            model_path = fusion_checkpoint
            
        print(f"🚀 [Fusion] 正在调用高级融合推理引擎: {model_path}")
        
        try:
            # 直接使用 inference_fusion.py 中的类
            self.engine = FusionInference(
                checkpoint_path=model_path,
                model_size='n'
            )
            print(f"✅ [Fusion] 引擎加载成功")
        except Exception as e:
            print(f"❌ [Fusion] 引擎加载失败: {e}")
            raise e

    def create_output_folders(self):
        """创建输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = "output"
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, timestamp

    def process(self, image_np, filename, fusion_weight=0.7, text_prompt=''):
        """
        处理单张图像：调用 FusionInference 引擎
        """
        import time
        start_time = time.time()
        
        print(f"� [Fusion] 开始处理图像: {filename}, mode: fusion, prompt: {text_prompt}")
        
        # 1. 准备目录
        output_dir, timestamp = self.create_output_folders()
        base_name = os.path.splitext(filename)[0]
        
        # 2. 保存临时原图供推理引擎读取
        temp_input_path = os.path.join(output_dir, f"temp_{filename}")
        cv2.imwrite(temp_input_path, image_np)
        print(f"💾 [Fusion] 临时原图已保存: {temp_input_path}")

        # 3. 执行检测
        detected_path = os.path.join(output_dir, f"3_fusion_detection_{base_name}.jpg")
        
        try:
            print(f"🔍 [Fusion] 正在调用引擎检测...")
            detections = self.engine.detect(
                img_path=temp_input_path,
                conf_threshold=0.25,
                save_path=detected_path,
                dehaze_prompt=text_prompt
            )
            num_objects = len(detections)
            print(f"✨ [Fusion] 检测完成，发现 {num_objects} 个目标")
        except Exception as e:
            print(f"❌ [Fusion] 推理过程出错: {e}")
            import traceback
            traceback.print_exc()
            raise e

        # 4. 再次获取去雾图 (用于前端展示)
        print(f"🌫️ [Fusion] 正在生成展示用去雾图...")
        if text_prompt:
            dehazed = self.engine.dehazer.process_opencv(image_np, prompt=text_prompt)
        else:
            dehazed = self.engine.dehazer.process_opencv(image_np)
            
        dehazed_path = os.path.join(output_dir, f"2_dehazed_{base_name}.jpg")
        cv2.imwrite(dehazed_path, dehazed)

        # 5. 准备正式路径
        original_path = os.path.join(output_dir, f"1_original_{base_name}.jpg")
        if os.path.exists(original_path):
            os.remove(original_path)
        os.rename(temp_input_path, original_path)

        # 6. 生成对比图
        print(f"📊 [Fusion] 正在生成对比图...")
        comparison_path = os.path.join(output_dir, f"4_comparison_{base_name}.jpg")
        h, w = 400, 600
        vis_orig = cv2.resize(image_np, (w, h))
        vis_dehz = cv2.resize(dehazed, (w, h))
        
        detected_vis = cv2.imread(detected_path)
        if detected_vis is not None:
            vis_det = cv2.resize(detected_vis, (w, h))
        else:
            vis_det = np.zeros((h, w, 3), dtype=np.uint8)
        
        cv2.putText(vis_orig, "Original Foggy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_dehz, "Dehazed (CoA)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_det, f"Fusion Det ({num_objects})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        comparison = np.hstack([vis_orig, vis_dehz, vis_det])
        cv2.imwrite(comparison_path, comparison)

        # 7. 保存结果文本
        if num_objects > 0:
            txt_path = os.path.join(output_dir, f"detection_results_{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 高级特征融合检测结果 (Direct Call) ===\n")
                f.write(f"图像: {filename}\n")
                if text_prompt:
                    f.write(f"Prompt: {text_prompt}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"检测到目标数量: {num_objects}\n")
                f.write("-" * 40 + "\n")
                for i, det in enumerate(detections):
                    cls = int(det['class'])
                    conf = float(det['confidence'])
                    name = self.engine.class_names[cls] if cls < len(self.engine.class_names) else f"ID:{cls}"
                    f.write(f"{i + 1}. {name}: 置信度 {conf:.2f}\n")

        # 8. 路径映射
        original_filename = f"{timestamp}/1_original_{base_name}.jpg"
        dehazed_filename = f"{timestamp}/2_dehazed_{base_name}.jpg"
        detected_filename = f"{timestamp}/3_fusion_detection_{base_name}.jpg"

        latency = (time.time() - start_time) * 1000
        print(f"🏁 [Fusion] 处理完毕，耗时: {latency:.2f}ms")
        
        return {
            'output_dir': output_dir,
            'timestamp': timestamp,
            'original_filename': original_filename,
            'dehazed_filename': dehazed_filename,
            'detected_filename': detected_filename,
            'num_objects': num_objects,
            'latency': latency
        }
