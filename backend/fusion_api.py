import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from datetime import datetime
from typing import Tuple, Optional

# ==========================================
# 1. Dehazing Module (Copied from multibackend)
# ==========================================

class DehazingModule:
    """
    图像去雾模块
    使用暗通道先验(Dark Channel Prior)算法 + 导向滤波
    """
    
    def __init__(self, omega: float = 0.95, t0: float = 0.1, radius: int = 15, eps: float = 0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.eps = eps
    
    def get_dark_channel(self, img: np.ndarray, size: int = 15) -> np.ndarray:
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def estimate_atmospheric_light(self, img: np.ndarray, dark_channel: np.ndarray, 
                                   percent: float = 0.001) -> np.ndarray:
        h, w = dark_channel.shape
        num_pixels = int(h * w * percent)
        dark_vec = dark_channel.reshape(h * w)
        img_vec = img.reshape(h * w, 3)
        indices = np.argsort(dark_vec)[-num_pixels:]
        atmospheric_light = np.max(img_vec[indices], axis=0)
        return atmospheric_light
    
    def estimate_transmission(self, img: np.ndarray, atmospheric_light: np.ndarray, 
                             size: int = 15) -> np.ndarray:
        normalized_img = img.astype(np.float64) / atmospheric_light
        dark_channel = self.get_dark_channel(normalized_img, size)
        transmission = 1 - self.omega * dark_channel
        return transmission
    
    def guided_filter(self, guide: np.ndarray, src: np.ndarray, 
                     radius: int, eps: float) -> np.ndarray:
        mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        
        mean_guide_guide = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_guide = mean_guide_guide - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        return mean_a * guide + mean_b
    
    def recover_image(self, img: np.ndarray, transmission: np.ndarray, 
                     atmospheric_light: np.ndarray) -> np.ndarray:
        transmission = np.maximum(transmission, self.t0)
        recovered = np.empty_like(img, dtype=np.float64)
        for i in range(3):
            recovered[:, :, i] = (img[:, :, i].astype(np.float64) - atmospheric_light[i]) / transmission + atmospheric_light[i]
        recovered = np.clip(recovered, 0, 255)
        return recovered.astype(np.uint8)
    
    def dehaze(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        img_float = img.astype(np.float64)
        dark_channel = self.get_dark_channel(img_float)
        atmospheric_light = self.estimate_atmospheric_light(img_float, dark_channel)
        transmission = self.estimate_transmission(img_float, atmospheric_light)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission_refined = self.guided_filter(gray, transmission, self.radius, self.eps)
        
        dehazed = self.recover_image(img_float, transmission_refined, atmospheric_light)
        
        return dehazed, {
            'dark_channel': dark_channel,
            'atmospheric_light': atmospheric_light,
            'transmission_refined': transmission_refined
        }

# ==========================================
# 2. Fusion Detector (Adapted for Web)
# ==========================================

class FusionDetector:
    def __init__(self, model_path='models/yolo11n.pt'):
        """初始化融合检测器"""
        self.model = YOLO(model_path)
        self.dehazer = DehazingModule()
        print(f"✅ [Fusion] 融合模型加载完成: {model_path}")

    def create_output_folders(self):
        """创建输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = "output"
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, timestamp

    def process(self, image_np, filename, fusion_weight=0.7, text_prompt=''):
        """
        处理单张图像：去雾 -> 融合 -> 检测
        """
        import time
        start_time = time.time()
        
        if text_prompt:
            print(f"📝 [Fusion] 收到文本引导: {text_prompt}")
        # 1. 准备目录
        output_dir, timestamp = self.create_output_folders()
        base_name = os.path.splitext(filename)[0]

        # 2. 去雾
        dehazed, _ = self.dehazer.dehaze(image_np)

        # 3. 图像融合
        # img_fusion = img * (1 - w) + dehazed * w
        image_float = image_np.astype(np.float32)
        dehazed_float = dehazed.astype(np.float32)
        img_fusion = (image_float * (1 - fusion_weight) + dehazed_float * fusion_weight)
        img_fusion = np.clip(img_fusion, 0, 255).astype(np.uint8)

        # 4. 目标检测 (在融合图上进行)
        results = self.model(img_fusion)
        
        # 绘制检测结果
        detected_vis = results[0].plot()
        num_objects = len(results[0].boxes)

        # 5. 生成文件名和路径
        # 为了兼容前端，文件名保持特定的前缀格式，或者直接用清晰的命名
        # 这里我们复用 dehaze_api 的命名习惯，但内容不同
        # 1_original: 原图
        # 2_dehazed: 去雾图
        # 3_detection: 融合检测结果图
        
        original_filename = f"{timestamp}/1_original_{base_name}.jpg"
        dehazed_filename = f"{timestamp}/2_dehazed_{base_name}.jpg"
        detected_filename = f"{timestamp}/3_fusion_detection_{base_name}.jpg" # 名字稍微改下以示区别
        comparison_filename = f"{timestamp}/4_comparison_{base_name}.jpg"

        original_path = os.path.join(output_dir, f"1_original_{base_name}.jpg")
        dehazed_path = os.path.join(output_dir, f"2_dehazed_{base_name}.jpg")
        detected_path = os.path.join(output_dir, f"3_fusion_detection_{base_name}.jpg")
        comparison_path = os.path.join(output_dir, f"4_comparison_{base_name}.jpg")

        # 6. 保存图像
        cv2.imwrite(original_path, image_np)
        cv2.imwrite(dehazed_path, dehazed)
        cv2.imwrite(detected_path, detected_vis)

        # 保存对比图 (可选，方便调试或查看)
        # 简单的水平拼接：原图 | 去雾 | 融合检测
        h, w = 400, 600
        vis_orig = cv2.resize(image_np, (w, h))
        vis_dehz = cv2.resize(dehazed, (w, h))
        vis_det = cv2.resize(detected_vis, (w, h))
        
        # 添加文字
        cv2.putText(vis_orig, "Original Foggy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_dehz, "Dehazed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_det, f"Fusion Detection ({num_objects})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        comparison = np.hstack([vis_orig, vis_dehz, vis_det])
        cv2.imwrite(comparison_path, comparison)

        # 7. 保存结果文本
        if num_objects > 0:
            txt_path = os.path.join(output_dir, f"detection_results_{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 融合检测结果 (Fusion Weight: {fusion_weight}) ===\n")
                f.write(f"图像: {filename}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"检测到目标数量: {num_objects}\n")
                f.write("-" * 40 + "\n")
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    name = results[0].names[cls]
                    f.write(f"{i + 1}. {name}: 置信度 {conf:.2f}\n")

        latency = (time.time() - start_time) * 1000  # 转为毫秒
        return {
            'output_dir': output_dir,
            'timestamp': timestamp,
            'original_filename': original_filename,
            'dehazed_filename': dehazed_filename,
            'detected_filename': detected_filename,
            'num_objects': num_objects,
            'latency': latency
        }
