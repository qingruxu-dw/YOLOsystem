# backend/dehaze_api.py
import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime


class DehazeDetector:
    def __init__(self, model_path='models/yolov8n.pt'):
        """初始化去雾检测器"""
        self.model = YOLO(model_path)
        print(f"✅ 模型加载完成: {model_path}")

    def create_output_folders(self):
        """创建输出目录结构"""
        # 使用当前时间戳作为文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建输出目录结构
        base_output_dir = "output"
        output_dir = os.path.join(base_output_dir, timestamp)

        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"📁 输出目录: {output_dir}")
        return output_dir, timestamp

    def dark_channel_dehaze(self, image, window_size=15, omega=0.95, t0=0.1):
        """暗通道先验去雾"""
        img = image.astype(np.float32) / 255.0
        dark = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark = cv2.erode(dark, kernel)
        atmospheric_light = np.percentile(dark, 99.9)
        transmission = 1 - omega * dark
        transmission = np.clip(transmission, t0, 1)

        result = np.zeros_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atmospheric_light) / transmission + atmospheric_light

        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)

    def process(self, image_np, filename, window_size=15, omega=0.95, t0=0.1):
        """
        处理单张图像

        参数:
            image_np: 原始图像 (numpy数组，BGR格式)
            filename: 原始文件名
        返回:
            处理结果字典
        """
        # 创建输出目录
        output_dir, timestamp = self.create_output_folders()

        # 获取基础文件名
        base_name = os.path.splitext(filename)[0]

        # 去雾处理
        dehazed = self.dark_channel_dehaze(image_np, window_size, omega, t0)

        # 目标检测
        results = self.model(dehazed)
        detected = results[0].plot()
        num_objects = len(results[0].boxes)

        # 生成文件名
        original_filename = f"{timestamp}/1_original_{base_name}.jpg"
        dehazed_filename = f"{timestamp}/2_dehazed_{base_name}.jpg"
        detected_filename = f"{timestamp}/3_detection_{base_name}.jpg"

        # 完整路径
        original_path = os.path.join(output_dir, f"1_original_{base_name}.jpg")
        dehazed_path = os.path.join(output_dir, f"2_dehazed_{base_name}.jpg")
        detected_path = os.path.join(output_dir, f"3_detection_{base_name}.jpg")

        # 保存图像
        cv2.imwrite(original_path, image_np)  # 直接保存numpy数组
        cv2.imwrite(dehazed_path, dehazed)
        cv2.imwrite(detected_path, detected)

        # 保存检测结果文本
        if num_objects > 0:
            txt_path = os.path.join(output_dir, f"detection_results_{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 检测结果 ===\n")
                f.write(f"图像: {filename}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"检测到目标数量: {num_objects}\n")
                f.write("-" * 40 + "\n")

                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    name = results[0].names[cls]
                    f.write(f"{i + 1}. {name}: 置信度 {conf:.2f}\n")

        return {
            'output_dir': output_dir,
            'timestamp': timestamp,
            'original_filename': original_filename,
            'dehazed_filename': dehazed_filename,
            'detected_filename': detected_filename,
            'num_objects': num_objects
        }