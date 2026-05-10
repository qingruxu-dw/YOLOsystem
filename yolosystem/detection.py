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


class MultiModelDetector:
    """
    多模型目标检测器
    支持同时使用多个YOLO模型进行检测并对比结果
    """

    def __init__(self,
                 model_configs: List[Dict[str, str]],
                 device: str = "cpu"):
        """
        初始化多模型检测器

        Args:
            model_configs: 模型配置列表，每个配置包含:
                - name: 模型名称（用于标识）
                - model_path: 模型路径或模型名称
            device: 运行设备 ("cpu" 或 "cuda")

        Example:
            configs = [
                {"name": "YOLOv8", "model_path": "yolov8n.pt"},
                {"name": "YOLOv11", "model_path": "yolo11n.pt"}
            ]
            detector = MultiModelDetector(configs, device="cpu")
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.device = device
        self.detectors = {}

        # 初始化所有检测器
        for config in model_configs:
            name = config['name']
            model_path = config['model_path']
            self.detectors[name] = YOLODetector(model_path=model_path, device=device)

    def detect_all(self,
                   img: np.ndarray,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   classes: Optional[List[int]] = None,
                   max_det: int = 300) -> Dict[str, List[Dict]]:
        """
        使用所有模型进行检测

        Args:
            img: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
            classes: 需要检测的类别列表
            max_det: 最大检测数量

        Returns:
            字典，key为模型名称，value为对应的检测结果列表
        """
        results = {}

        for name, detector in self.detectors.items():
            detections = detector.detect(
                img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                classes=classes,
                max_det=max_det
            )
            results[name] = detections

        return results

    def draw_detections_comparison(self,
                                   img: np.ndarray,
                                   all_detections: Dict[str, List[Dict]],
                                   colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
                                   thickness: int = 2) -> Dict[str, np.ndarray]:
        """
        为每个模型分别绘制检测结果

        Args:
            img: 输入图像
            all_detections: 所有模型的检测结果
            colors: 每个模型的颜色配置，默认为绿色和蓝色
            thickness: 线条粗细

        Returns:
            字典，key为模型名称，value为绘制了检测框的图像
        """
        if colors is None:
            # 默认颜色配置
            default_colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 165, 255),  # 橙色
                (0, 255, 255),  # 黄色
                (255, 0, 255),  # 品红
            ]
            colors = {}
            for idx, name in enumerate(self.detectors.keys()):
                colors[name] = default_colors[idx % len(default_colors)]

        detection_images = {}

        for name, detections in all_detections.items():
            color = colors.get(name, (0, 255, 0))
            detector = self.detectors[name]
            detection_img = detector.draw_detections(img, detections, color=color, thickness=thickness)
            detection_images[name] = detection_img

        return detection_images

    def create_side_by_side_comparison(self,
                                       detection_images: Dict[str, np.ndarray],
                                       add_labels: bool = True) -> np.ndarray:
        """
        创建并排对比图像

        Args:
            detection_images: 各模型的检测结果图像
            add_labels: 是否添加模型名称标签

        Returns:
            并排对比的图像
        """
        if not detection_images:
            raise ValueError("No detection images provided")

        images = list(detection_images.values())
        names = list(detection_images.keys())

        # 确保所有图像尺寸一致
        height = images[0].shape[0]
        width = images[0].shape[1]

        # 创建并排图像
        comparison = np.hstack(images)

        # 添加标签
        if add_labels:
            comparison_copy = comparison.copy()
            current_x = 0

            for i, name in enumerate(names):
                # 在每个图像顶部添加模型名称
                label_position = (current_x + 10, 30)

                # 绘制文本背景
                (text_width, text_height), baseline = cv2.getTextSize(
                    name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )
                cv2.rectangle(
                    comparison_copy,
                    (current_x + 5, 5),
                    (current_x + 15 + text_width, 40 + baseline),
                    (0, 0, 0),
                    -1
                )

                # 绘制文本
                cv2.putText(
                    comparison_copy,
                    name,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                current_x += width

            return comparison_copy

        return comparison

    def get_comparison_statistics(self, all_detections: Dict[str, List[Dict]]) -> Dict:
        """
        获取多模型检测的对比统计信息

        Args:
            all_detections: 所有模型的检测结果

        Returns:
            对比统计信息字典
        """
        stats = {}

        for name, detections in all_detections.items():
            # 统计各类别的数量
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            stats[name] = {
                'total_detections': len(detections),
                'class_counts': class_counts,
                'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
            }

        return stats

    def get_models_info(self) -> Dict[str, Dict]:
        """
        获取所有模型的信息

        Returns:
            所有模型的信息字典
        """
        info = {}
        for name, detector in self.detectors.items():
            info[name] = detector.get_model_info()
        return info
