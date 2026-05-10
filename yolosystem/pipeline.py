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
from .detection import YOLODetector, MultiModelDetector
from .fusion import FusionDetector, ImageQualityAssessment


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


class MultiModelDetectionPipeline:
    """
    多模型去雾和目标检测流水线
    支持同时使用多个YOLO模型进行检测并对比结果
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化多模型流水线

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
                'multi_detection': {
                    'enabled': True,
                    'models': [
                        {'name': 'YOLOv8', 'model_path': 'yolov8n.pt'},
                        {'name': 'YOLOv11', 'model_path': 'yolo11n.pt'}
                    ],
                    'conf_threshold': 0.25,
                    'iou_threshold': 0.45,
                    'max_det': 300,
                    'classes': None,
                    'device': 'cpu'
                },
                'pipeline': {
                    'save_dehazed': True,
                    'save_detections': True,
                    'save_comparison': True,
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

        # 初始化多模型检测模块
        self.multi_detector = MultiModelDetector(
            model_configs=self.config['multi_detection']['models'],
            device=self.config['multi_detection']['device']
        )

        # 创建输出目录
        self.output_dir = Path(self.config['pipeline']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, img: np.ndarray, save_path: Optional[str] = None) -> Dict:
        """
        使用多个模型处理单张图像

        Args:
            img: 输入图像 (BGR格式)
            save_path: 保存路径（不包含扩展名），如果为None则不保存

        Returns:
            处理结果字典，包含:
                - original: 原始图像
                - dehazed: 去雾图像（如果启用）
                - all_detections: 所有模型的检测结果
                - detection_images: 各模型的检测可视化图像
                - comparison_image: 并排对比图像
        """
        results = {'original': img}

        # 1. 去雾处理
        if self.dehazing_enabled and self.dehazer is not None:
            dehazed_img = self.dehazer.process(img)
            results['dehazed'] = dehazed_img
            detection_input = dehazed_img
        else:
            detection_input = img

        # 2. 多模型目标检测
        all_detections = self.multi_detector.detect_all(
            detection_input,
            conf_threshold=self.config['multi_detection']['conf_threshold'],
            iou_threshold=self.config['multi_detection']['iou_threshold'],
            classes=self.config['multi_detection']['classes'],
            max_det=self.config['multi_detection']['max_det']
        )
        results['all_detections'] = all_detections

        # 3. 绘制各模型的检测结果
        detection_images = self.multi_detector.draw_detections_comparison(
            detection_input,
            all_detections
        )
        results['detection_images'] = detection_images

        # 4. 创建并排对比图像
        comparison_image = self.multi_detector.create_side_by_side_comparison(
            detection_images,
            add_labels=True
        )
        results['comparison_image'] = comparison_image

        # 5. 保存结果
        if save_path:
            save_path = Path(save_path)

            if self.config['pipeline']['save_dehazed'] and self.dehazing_enabled and 'dehazed' in results:
                dehazed_path = save_path.parent / f"{save_path.stem}_dehazed{save_path.suffix}"
                cv2.imwrite(str(dehazed_path), results['dehazed'])

            if self.config['pipeline']['save_detections']:
                # 保存各模型的检测结果
                for model_name, detection_img in detection_images.items():
                    detection_path = save_path.parent / f"{save_path.stem}_detection_{model_name}{save_path.suffix}"
                    cv2.imwrite(str(detection_path), detection_img)

            if self.config['pipeline']['save_comparison']:
                # 保存并排对比图像
                comparison_path = save_path.parent / f"{save_path.stem}_comparison{save_path.suffix}"
                cv2.imwrite(str(comparison_path), comparison_image)

        # 6. 显示结果（如果启用）
        if self.config['pipeline']['show_results']:
            self._show_results(img, results)

        return results

    def process_image_file(self, image_path: str) -> Dict:
        """
        从文件读取并使用多模型处理图像

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

    def _show_results(self, original: np.ndarray, results: Dict) -> None:
        """
        显示多模型处理结果

        Args:
            original: 原始图像
            results: 处理结果
        """
        # 显示原始图像
        cv2.imshow('Original', original)

        # 显示去雾图像
        if 'dehazed' in results:
            cv2.imshow('Dehazed', results['dehazed'])

        # 显示各模型的检测结果
        for model_name, detection_img in results['detection_images'].items():
            cv2.imshow(f'Detection - {model_name}', detection_img)

        # 显示并排对比图像
        cv2.imshow('Comparison', results['comparison_image'])

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_statistics(self, results: Dict) -> Dict:
        """
        获取多模型检测统计信息

        Args:
            results: 处理结果

        Returns:
            统计信息字典
        """
        return self.multi_detector.get_comparison_statistics(results['all_detections'])

    def print_statistics(self, stats: Dict) -> None:
        """
        打印统计信息

        Args:
            stats: 统计信息字典
        """
        print("\n" + "="*60)
        print("多模型检测统计信息")
        print("="*60)

        for model_name, model_stats in stats.items():
            print(f"\n【{model_name}】")
            print(f"  总检测数: {model_stats['total_detections']}")
            print(f"  平均置信度: {model_stats['average_confidence']:.3f}")
            print(f"  类别分布:")
            for class_name, count in model_stats['class_counts'].items():
                print(f"    - {class_name}: {count}")

        print("="*60 + "\n")


class FusionDetectionPipeline:
    """
    融合检测流水线
    使用双路检测+智能融合策略，真正结合去雾和目标检测
    支持多种融合策略：adaptive（自适应）、confidence（置信度）、quality（质量）、both（保留两者）
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化融合检测流水线

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
                'fusion': {
                    'strategy': 'adaptive',  # adaptive, confidence, quality, both
                    'iou_threshold': 0.5
                },
                'pipeline': {
                    'save_original_detection': True,
                    'save_dehazed_detection': True,
                    'save_fused_detection': True,
                    'save_comparison': True,
                    'output_dir': 'outputs',
                    'show_results': False
                }
            }

        # 初始化去雾模块
        self.dehazer = DehazingModule(
            omega=self.config['dehazing']['omega'],
            t0=self.config['dehazing']['t0'],
            radius=self.config['dehazing']['radius'],
            eps=self.config['dehazing']['eps']
        )

        # 初始化检测模块
        self.detector = YOLODetector(
            model_path=self.config['detection']['model'],
            device=self.config['detection']['device']
        )

        # 初始化融合检测器
        self.fusion_detector = FusionDetector(
            detector=self.detector,
            dehazer=self.dehazer,
            fusion_strategy=self.config['fusion']['strategy'],
            iou_threshold=self.config['fusion']['iou_threshold']
        )

        # 创建输出目录
        self.output_dir = Path(self.config['pipeline']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, img: np.ndarray, save_path: Optional[str] = None) -> Dict:
        """
        使用融合策略处理单张图像

        Args:
            img: 输入图像 (BGR格式)
            save_path: 保存路径（不包含扩展名），如果为None则不保存

        Returns:
            处理结果字典，包含:
                - original: 原始图像
                - dehazed: 去雾图像
                - original_detections: 原图检测结果
                - dehazed_detections: 去雾图检测结果
                - fused_detections: 融合后的检测结果
                - original_quality: 原图质量评估
                - dehazed_quality: 去雾图质量评估
                - original_detection_img: 原图检测可视化
                - dehazed_detection_img: 去雾图检测可视化
                - fused_detection_img: 融合检测可视化
                - comparison_img: 三路对比图
        """
        results = {'original': img}

        # 1. 执行融合检测
        fusion_results = self.fusion_detector.detect_with_fusion(
            img,
            conf_threshold=self.config['detection']['conf_threshold'],
            iou_threshold=self.config['detection']['iou_threshold'],
            classes=self.config['detection']['classes'],
            max_det=self.config['detection']['max_det']
        )

        # 2. 整合结果
        results.update(fusion_results)

        # 3. 绘制检测结果
        original_detection_img = self.detector.draw_detections(
            img, fusion_results['original_detections'],
            color=(0, 255, 0), thickness=2
        )
        results['original_detection_img'] = original_detection_img

        dehazed_detection_img = self.detector.draw_detections(
            fusion_results['dehazed_image'], fusion_results['dehazed_detections'],
            color=(255, 0, 0), thickness=2
        )
        results['dehazed_detection_img'] = dehazed_detection_img

        # 绘制融合结果（根据来源使用不同颜色）
        fused_detection_img = self._draw_fused_detections(
            img, fusion_results['dehazed_image'], fusion_results['fused_detections']
        )
        results['fused_detection_img'] = fused_detection_img

        # 4. 创建三路对比图
        comparison_img = self._create_comparison_image(
            original_detection_img,
            dehazed_detection_img,
            fused_detection_img
        )
        results['comparison_img'] = comparison_img

        # 5. 保存结果
        if save_path:
            save_path = Path(save_path)

            if self.config['pipeline']['save_original_detection']:
                orig_path = save_path.parent / f"{save_path.stem}_original_detection{save_path.suffix}"
                cv2.imwrite(str(orig_path), original_detection_img)

            if self.config['pipeline']['save_dehazed_detection']:
                dehazed_path = save_path.parent / f"{save_path.stem}_dehazed_detection{save_path.suffix}"
                cv2.imwrite(str(dehazed_path), dehazed_detection_img)

            if self.config['pipeline']['save_fused_detection']:
                fused_path = save_path.parent / f"{save_path.stem}_fused_detection{save_path.suffix}"
                cv2.imwrite(str(fused_path), fused_detection_img)

            if self.config['pipeline']['save_comparison']:
                comparison_path = save_path.parent / f"{save_path.stem}_comparison{save_path.suffix}"
                cv2.imwrite(str(comparison_path), comparison_img)

        # 6. 显示结果（如果启用）
        if self.config['pipeline']['show_results']:
            self._show_results(results)

        return results

    def process_image_file(self, image_path: str) -> Dict:
        """
        从文件读取并使用融合策略处理图像

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

    def _draw_fused_detections(self,
                               original_img: np.ndarray,
                               dehazed_img: np.ndarray,
                               fused_detections: List[Dict]) -> np.ndarray:
        """
        绘制融合检测结果（根据来源选择不同颜色）

        Args:
            original_img: 原始图像
            dehazed_img: 去雾图像
            fused_detections: 融合后的检测结果

        Returns:
            绘制了检测框的图像
        """
        # 使用去雾图作为基础（因为融合后的结果通常质量更好）
        img_copy = dehazed_img.copy()

        for det in fused_detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 根据来源选择颜色
            if det.get('source') == 'original':
                color = (0, 255, 0)  # 绿色表示来自原图
                label_suffix = " (Orig)"
            elif det.get('source') == 'dehazed':
                color = (255, 0, 0)  # 蓝色表示来自去雾图
                label_suffix = " (Dehz)"
            else:
                color = (0, 165, 255)  # 橙色表示未知来源
                label_suffix = ""

            # 绘制边界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            label = f"{det['class_name']}: {det['confidence']:.2f}{label_suffix}"

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

    def _create_comparison_image(self,
                                original_detection: np.ndarray,
                                dehazed_detection: np.ndarray,
                                fused_detection: np.ndarray) -> np.ndarray:
        """
        创建三路对比图像

        Args:
            original_detection: 原图检测结果
            dehazed_detection: 去雾图检测结果
            fused_detection: 融合检测结果

        Returns:
            并排对比的图像
        """
        # 确保所有图像尺寸一致
        height = original_detection.shape[0]
        width = original_detection.shape[1]

        # 创建并排图像
        comparison = np.hstack([original_detection, dehazed_detection, fused_detection])

        # 添加标签
        labels = ['Original Detection', 'Dehazed Detection', 'Fused Detection']
        current_x = 0

        for label in labels:
            # 在每个图像顶部添加标签
            label_position = (current_x + 10, 30)

            # 绘制文本背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            cv2.rectangle(
                comparison,
                (current_x + 5, 5),
                (current_x + 15 + text_width, 40 + baseline),
                (0, 0, 0),
                -1
            )

            # 绘制文本
            cv2.putText(
                comparison,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            current_x += width

        return comparison

    def _show_results(self, results: Dict) -> None:
        """
        显示处理结果

        Args:
            results: 处理结果
        """
        # 显示原始图像
        cv2.imshow('Original', results['original'])

        # 显示去雾图像
        cv2.imshow('Dehazed', results['dehazed_image'])

        # 显示各路检测结果
        cv2.imshow('Original Detection', results['original_detection_img'])
        cv2.imshow('Dehazed Detection', results['dehazed_detection_img'])
        cv2.imshow('Fused Detection', results['fused_detection_img'])

        # 显示对比图
        cv2.imshow('Comparison', results['comparison_img'])

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_statistics(self, results: Dict) -> Dict:
        """
        获取详细的检测统计信息

        Args:
            results: 处理结果

        Returns:
            统计信息字典
        """
        def get_detection_stats(detections: List[Dict]) -> Dict:
            if not detections:
                return {
                    'total_detections': 0,
                    'class_counts': {},
                    'average_confidence': 0.0
                }

            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            return {
                'total_detections': len(detections),
                'class_counts': class_counts,
                'average_confidence': np.mean([d['confidence'] for d in detections])
            }

        stats = {
            'original': get_detection_stats(results['original_detections']),
            'dehazed': get_detection_stats(results['dehazed_detections']),
            'fused': get_detection_stats(results['fused_detections']),
            'quality': {
                'original': results['original_quality'],
                'dehazed': results['dehazed_quality']
            },
            'fusion_info': {
                'strategy': self.config['fusion']['strategy'],
                'quality_improvement': {
                    'sharpness': (results['dehazed_quality']['sharpness'] /
                                 (results['original_quality']['sharpness'] + 1e-6)),
                    'contrast': (results['dehazed_quality']['contrast'] /
                                (results['original_quality']['contrast'] + 1e-6)),
                    'entropy': (results['dehazed_quality']['entropy'] /
                               (results['original_quality']['entropy'] + 1e-6))
                }
            }
        }

        # 统计融合结果的来源分布
        if results['fused_detections']:
            source_counts = {}
            for det in results['fused_detections']:
                source = det.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            stats['fusion_info']['source_distribution'] = source_counts

        return stats

    def print_statistics(self, stats: Dict) -> None:
        """
        打印统计信息

        Args:
            stats: 统计信息字典
        """
        print("\n" + "="*80)
        print("融合检测统计信息")
        print("="*80)

        # 打印质量评估
        print("\n【图像质量评估】")
        print(f"原图:")
        for key, value in stats['quality']['original'].items():
            print(f"  {key}: {value:.3f}")

        print(f"\n去雾图:")
        for key, value in stats['quality']['dehazed'].items():
            print(f"  {key}: {value:.3f}")

        print(f"\n质量改善比例:")
        for key, value in stats['fusion_info']['quality_improvement'].items():
            print(f"  {key}: {value:.3f}x")

        # 打印检测统计
        print("\n【检测结果统计】")
        for phase in ['original', 'dehazed', 'fused']:
            print(f"\n{phase.capitalize()}:")
            phase_stats = stats[phase]
            print(f"  总检测数: {phase_stats['total_detections']}")
            print(f"  平均置信度: {phase_stats['average_confidence']:.3f}")
            if phase_stats['class_counts']:
                print(f"  类别分布:")
                for class_name, count in phase_stats['class_counts'].items():
                    print(f"    - {class_name}: {count}")

        # 打印融合信息
        print(f"\n【融合策略】: {stats['fusion_info']['strategy']}")
        if 'source_distribution' in stats['fusion_info']:
            print(f"来源分布:")
            for source, count in stats['fusion_info']['source_distribution'].items():
                print(f"  - {source}: {count}")

        print("="*80 + "\n")
