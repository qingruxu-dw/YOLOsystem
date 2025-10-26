"""
YOLO Detection Module
基于Ultralytics YOLO的目标检测模块
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


class YOLODetector:
    """
    YOLO目标检测器封装类
    支持YOLOv8系列模型
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型路径或模型名称 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: 运行设备 ("cpu" 或 "cuda")
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        
        # 将模型移到指定设备
        if device == "cuda":
            self.model.to("cuda")
    
    def detect(self, 
               img: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               classes: Optional[List[int]] = None,
               max_det: int = 300) -> List[Dict]:
        """
        对图像进行目标检测
        
        Args:
            img: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
            classes: 需要检测的类别列表，None表示检测所有类别
            max_det: 最大检测数量
            
        Returns:
            检测结果列表，每个元素包含:
                - bbox: [x1, y1, x2, y2]
                - confidence: 置信度
                - class_id: 类别ID
                - class_name: 类别名称
        """
        # 进行检测
        results = self.model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            max_det=max_det,
            verbose=False
        )
        
        # 解析结果
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i]
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度和类别
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, 
                       img: np.ndarray, 
                       detections: List[Dict],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            img: 输入图像
            detections: 检测结果列表
            color: 边界框颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            绘制了检测框的图像
        """
        img_copy = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制文本背景
            cv2.rectangle(
                img_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img_copy
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'num_classes': len(self.model.names),
            'class_names': self.model.names
        }
