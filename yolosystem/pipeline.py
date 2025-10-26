"""
Pipeline Module
整合去雾和目标检测的完整流程
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from .dehazing import DehazingModule
from .detection import YOLODetector


class DehazingDetectionPipeline:
    """
    去雾和目标检测流水线
    整合图像去雾和YOLO目标检测功能
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化流水线
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 加载配置
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 默认配置
            self.config = {
                'dehazing': {
                    'enabled': True,
                    'omega': 0.95,
                    't0': 0.1,
                    'radius': 15,
                    'eps': 0.001
                },
                'detection': {
                    'model': 'yolov8n.pt',
                    'conf_threshold': 0.25,
                    'iou_threshold': 0.45,
                    'max_det': 300,
                    'classes': None,
                    'device': 'cpu'
                },
                'pipeline': {
                    'save_dehazed': True,
                    'save_detections': True,
                    'output_dir': 'outputs',
                    'show_results': False
                }
            }
        
        # 初始化去雾模块
        self.dehazing_enabled = self.config['dehazing']['enabled']
        if self.dehazing_enabled:
            self.dehazer = DehazingModule(
                omega=self.config['dehazing']['omega'],
                t0=self.config['dehazing']['t0'],
                radius=self.config['dehazing']['radius'],
                eps=self.config['dehazing']['eps']
            )
        else:
            self.dehazer = None
        
        # 初始化检测模块
        self.detector = YOLODetector(
            model_path=self.config['detection']['model'],
            device=self.config['detection']['device']
        )
        
        # 创建输出目录
        self.output_dir = Path(self.config['pipeline']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image(self, img: np.ndarray, save_path: Optional[str] = None) -> Dict:
        """
        处理单张图像
        
        Args:
            img: 输入图像 (BGR格式)
            save_path: 保存路径（不包含扩展名），如果为None则不保存
            
        Returns:
            处理结果字典，包含:
                - original: 原始图像
                - dehazed: 去雾图像（如果启用）
                - detections: 检测结果
                - detection_img: 带检测框的图像
        """
        results = {'original': img}
        
        # 1. 去雾处理
        if self.dehazing_enabled and self.dehazer is not None:
            dehazed_img = self.dehazer.process(img)
            results['dehazed'] = dehazed_img
            detection_input = dehazed_img
        else:
            detection_input = img
        
        # 2. 目标检测
        detections = self.detector.detect(
            detection_input,
            conf_threshold=self.config['detection']['conf_threshold'],
            iou_threshold=self.config['detection']['iou_threshold'],
            classes=self.config['detection']['classes'],
            max_det=self.config['detection']['max_det']
        )
        results['detections'] = detections
        
        # 3. 绘制检测结果
        detection_img = self.detector.draw_detections(detection_input, detections)
        results['detection_img'] = detection_img
        
        # 4. 保存结果
        if save_path:
            save_path = Path(save_path)
            
            if self.config['pipeline']['save_dehazed'] and self.dehazing_enabled and 'dehazed' in results:
                dehazed_path = save_path.parent / f"{save_path.stem}_dehazed{save_path.suffix}"
                cv2.imwrite(str(dehazed_path), results['dehazed'])
            
            if self.config['pipeline']['save_detections']:
                detection_path = save_path.parent / f"{save_path.stem}_detection{save_path.suffix}"
                cv2.imwrite(str(detection_path), detection_img)
        
        # 5. 显示结果（如果启用）
        if self.config['pipeline']['show_results']:
            self._show_results(img, results)
        
        return results
    
    def process_image_file(self, image_path: str) -> Dict:
        """
        从文件读取并处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理结果字典
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image from {image_path}")
        
        # 准备保存路径
        image_name = Path(image_path).stem
        save_path = self.output_dir / f"{image_name}.jpg"
        
        return self.process_image(img, str(save_path))
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径，如果为None则自动生成
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 准备输出
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = self.output_dir / f"{video_name}_processed.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            results = self.process_image(frame)
            processed_frame = results['detection_img']
            
            # 写入输出视频
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"Video processing complete. Saved to: {output_path}")
    
    def _show_results(self, original: np.ndarray, results: Dict) -> None:
        """
        显示处理结果
        
        Args:
            original: 原始图像
            results: 处理结果
        """
        # 显示原始图像
        cv2.imshow('Original', original)
        
        # 显示去雾图像
        if 'dehazed' in results:
            cv2.imshow('Dehazed', results['dehazed'])
        
        # 显示检测结果
        cv2.imshow('Detection', results['detection_img'])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_statistics(self, results: Dict) -> Dict:
        """
        获取检测统计信息
        
        Args:
            results: 处理结果
            
        Returns:
            统计信息字典
        """
        detections = results['detections']
        
        # 统计各类别的数量
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats = {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
        }
        
        return stats
