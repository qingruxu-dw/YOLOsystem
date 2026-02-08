"""
Fusion Detection Module
实现双路检测+智能融合策略，真正结合去雾和目标检测
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from .detection import YOLODetector
from .dehazing import DehazingModule


class ImageQualityAssessment:
    """
    图像质量评估工具
    用于评估图像的清晰度、对比度等指标
    """

    @staticmethod
    def calculate_sharpness(img: np.ndarray) -> float:
        """
        计算图像清晰度（使用Laplacian方差）

        Args:
            img: 输入图像

        Returns:
            清晰度分数（越大越清晰）
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        return float(sharpness)

    @staticmethod
    def calculate_contrast(img: np.ndarray) -> float:
        """
        计算图像对比度（使用标准差）

        Args:
            img: 输入图像

        Returns:
            对比度分数
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        contrast = gray.std()

        return float(contrast)

    @staticmethod
    def calculate_brightness(img: np.ndarray) -> float:
        """
        计算图像亮度（平均值）

        Args:
            img: 输入图像

        Returns:
            亮度值
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        brightness = gray.mean()

        return float(brightness)

    @staticmethod
    def calculate_entropy(img: np.ndarray) -> float:
        """
        计算图像信息熵（信息量）

        Args:
            img: 输入图像

        Returns:
            信息熵值
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # 计算熵
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    @staticmethod
    def assess_image_quality(img: np.ndarray) -> Dict[str, float]:
        """
        综合评估图像质量

        Args:
            img: 输入图像

        Returns:
            包含各项质量指标的字典
        """
        return {
            'sharpness': ImageQualityAssessment.calculate_sharpness(img),
            'contrast': ImageQualityAssessment.calculate_contrast(img),
            'brightness': ImageQualityAssessment.calculate_brightness(img),
            'entropy': ImageQualityAssessment.calculate_entropy(img)
        }


class FusionDetector:
    """
    融合检测器
    结合原图和去雾图的检测结果，实现智能融合
    """

    def __init__(self,
                 detector: YOLODetector,
                 dehazer: Optional[DehazingModule] = None,
                 fusion_strategy: str = 'adaptive',
                 iou_threshold: float = 0.5):
        """
        初始化融合检测器

        Args:
            detector: YOLO检测器实例
            dehazer: 去雾模块实例，如果为None则会创建默认的
            fusion_strategy: 融合策略 ('adaptive', 'confidence', 'quality', 'both')
            iou_threshold: 用于匹配检测框的IoU阈值
        """
        self.detector = detector
        self.dehazer = dehazer if dehazer is not None else DehazingModule()
        self.fusion_strategy = fusion_strategy
        self.iou_threshold = iou_threshold
        self.quality_assessor = ImageQualityAssessment()

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的IoU

        Args:
            box1, box2: 边界框 [x1, y1, x2, y2]

        Returns:
            IoU值
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou

    def match_detections(self,
                        detections1: List[Dict],
                        detections2: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匹配两组检测结果

        Args:
            detections1: 第一组检测结果
            detections2: 第二组检测结果

        Returns:
            - matched_pairs: 匹配的索引对列表 [(idx1, idx2), ...]
            - unmatched1: 第一组中未匹配的索引
            - unmatched2: 第二组中未匹配的索引
        """
        matched_pairs = []
        unmatched1 = list(range(len(detections1)))
        unmatched2 = list(range(len(detections2)))

        # 计算所有检测框对的IoU
        iou_matrix = np.zeros((len(detections1), len(detections2)))
        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                # 只匹配相同类别的检测
                if det1['class_id'] == det2['class_id']:
                    iou_matrix[i, j] = self.calculate_iou(det1['bbox'], det2['bbox'])

        # 贪心匹配：每次选择IoU最大的配对
        while iou_matrix.size > 0:
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            # 找到最大IoU的位置
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            i, j = max_idx

            # 转换为原始索引
            orig_i = unmatched1[i]
            orig_j = unmatched2[j]

            matched_pairs.append((orig_i, orig_j))

            # 从未匹配列表中移除
            unmatched1.remove(orig_i)
            unmatched2.remove(orig_j)

            # 从矩阵中移除已匹配的行和列
            iou_matrix = np.delete(iou_matrix, i, axis=0)
            iou_matrix = np.delete(iou_matrix, j, axis=1)

        return matched_pairs, unmatched1, unmatched2

    def fuse_detections_adaptive(self,
                                 original_detections: List[Dict],
                                 dehazed_detections: List[Dict],
                                 original_quality: Dict[str, float],
                                 dehazed_quality: Dict[str, float]) -> List[Dict]:
        """
        自适应融合策略：根据图像质量和检测置信度综合决策

        Args:
            original_detections: 原图检测结果
            dehazed_detections: 去雾图检测结果
            original_quality: 原图质量指标
            dehazed_quality: 去雾图质量指标

        Returns:
            融合后的检测结果
        """
        matched_pairs, unmatched_orig, unmatched_dehazed = self.match_detections(
            original_detections, dehazed_detections
        )

        fused_detections = []

        # 计算质量权重（归一化）
        quality_weight_dehazed = self._calculate_quality_weight(original_quality, dehazed_quality)
        quality_weight_original = 1.0 - quality_weight_dehazed

        # 处理匹配的检测对
        for idx_orig, idx_dehazed in matched_pairs:
            det_orig = original_detections[idx_orig]
            det_dehazed = dehazed_detections[idx_dehazed]

            # 置信度加权
            conf_orig = det_orig['confidence']
            conf_dehazed = det_dehazed['confidence']

            # 综合权重 = 质量权重 * 置信度
            weight_orig = quality_weight_original * conf_orig
            weight_dehazed = quality_weight_dehazed * conf_dehazed

            # 选择权重更高的检测结果
            if weight_orig > weight_dehazed:
                fused_det = det_orig.copy()
                fused_det['source'] = 'original'
                fused_det['fusion_weight'] = weight_orig
            else:
                fused_det = det_dehazed.copy()
                fused_det['source'] = 'dehazed'
                fused_det['fusion_weight'] = weight_dehazed

            # 记录融合信息
            fused_det['matched'] = True
            fused_det['original_conf'] = conf_orig
            fused_det['dehazed_conf'] = conf_dehazed

            fused_detections.append(fused_det)

        # 添加原图中未匹配的检测（如果质量权重足够高）
        for idx in unmatched_orig:
            det = original_detections[idx].copy()
            det['source'] = 'original'
            det['matched'] = False
            det['fusion_weight'] = quality_weight_original * det['confidence']

            # 只保留高置信度或高质量权重的检测
            if det['confidence'] > 0.4 or quality_weight_original > 0.6:
                fused_detections.append(det)

        # 添加去雾图中未匹配的检测（如果质量权重足够高）
        for idx in unmatched_dehazed:
            det = dehazed_detections[idx].copy()
            det['source'] = 'dehazed'
            det['matched'] = False
            det['fusion_weight'] = quality_weight_dehazed * det['confidence']

            # 只保留高置信度或高质量权重的检测
            if det['confidence'] > 0.4 or quality_weight_dehazed > 0.6:
                fused_detections.append(det)

        return fused_detections

    def fuse_detections_confidence(self,
                                   original_detections: List[Dict],
                                   dehazed_detections: List[Dict]) -> List[Dict]:
        """
        基于置信度的融合策略：始终选择置信度更高的结果

        Args:
            original_detections: 原图检测结果
            dehazed_detections: 去雾图检测结果

        Returns:
            融合后的检测结果
        """
        matched_pairs, unmatched_orig, unmatched_dehazed = self.match_detections(
            original_detections, dehazed_detections
        )

        fused_detections = []

        # 处理匹配的检测对：选择置信度更高的
        for idx_orig, idx_dehazed in matched_pairs:
            det_orig = original_detections[idx_orig]
            det_dehazed = dehazed_detections[idx_dehazed]

            if det_orig['confidence'] > det_dehazed['confidence']:
                fused_det = det_orig.copy()
                fused_det['source'] = 'original'
            else:
                fused_det = det_dehazed.copy()
                fused_det['source'] = 'dehazed'

            fused_det['matched'] = True
            fused_detections.append(fused_det)

        # 添加未匹配的检测
        for idx in unmatched_orig:
            det = original_detections[idx].copy()
            det['source'] = 'original'
            det['matched'] = False
            fused_detections.append(det)

        for idx in unmatched_dehazed:
            det = dehazed_detections[idx].copy()
            det['source'] = 'dehazed'
            det['matched'] = False
            fused_detections.append(det)

        return fused_detections

    def fuse_detections_quality(self,
                                original_detections: List[Dict],
                                dehazed_detections: List[Dict],
                                original_quality: Dict[str, float],
                                dehazed_quality: Dict[str, float]) -> List[Dict]:
        """
        基于图像质量的融合策略：优先选择质量更好的图像的检测结果

        Args:
            original_detections: 原图检测结果
            dehazed_detections: 去雾图检测结果
            original_quality: 原图质量指标
            dehazed_quality: 去雾图质量指标

        Returns:
            融合后的检测结果
        """
        quality_weight_dehazed = self._calculate_quality_weight(original_quality, dehazed_quality)

        # 如果去雾显著提升质量，优先使用去雾图的结果
        if quality_weight_dehazed > 0.6:
            return dehazed_detections
        # 如果去雾没有明显改善，使用原图结果
        elif quality_weight_dehazed < 0.4:
            return original_detections
        # 质量相近，使用置信度融合
        else:
            return self.fuse_detections_confidence(original_detections, dehazed_detections)

    def fuse_detections_both(self,
                            original_detections: List[Dict],
                            dehazed_detections: List[Dict]) -> List[Dict]:
        """
        保留两者结果的融合策略：通过NMS去除重复

        Args:
            original_detections: 原图检测结果
            dehazed_detections: 去雾图检测结果

        Returns:
            融合后的检测结果
        """
        # 合并所有检测
        all_detections = []

        for det in original_detections:
            det_copy = det.copy()
            det_copy['source'] = 'original'
            all_detections.append(det_copy)

        for det in dehazed_detections:
            det_copy = det.copy()
            det_copy['source'] = 'dehazed'
            all_detections.append(det_copy)

        if not all_detections:
            return []

        # 按类别分组进行NMS
        fused_detections = []
        classes = set([det['class_id'] for det in all_detections])

        for cls_id in classes:
            cls_detections = [det for det in all_detections if det['class_id'] == cls_id]

            # 应用NMS
            nms_result = self._apply_nms(cls_detections, self.iou_threshold)
            fused_detections.extend(nms_result)

        return fused_detections

    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        应用非极大值抑制(NMS)

        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值

        Returns:
            NMS后的检测结果
        """
        if not detections:
            return []

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            # 保留置信度最高的
            best = detections.pop(0)
            keep.append(best)

            # 移除与best重叠度高的检测
            detections = [
                det for det in detections
                if self.calculate_iou(best['bbox'], det['bbox']) < iou_threshold
            ]

        return keep

    def _calculate_quality_weight(self,
                                  original_quality: Dict[str, float],
                                  dehazed_quality: Dict[str, float]) -> float:
        """
        计算去雾图的质量权重（0-1之间）

        Args:
            original_quality: 原图质量指标
            dehazed_quality: 去雾图质量指标

        Returns:
            去雾图的权重（越大表示去雾图质量越好）
        """
        # 归一化各项指标的改善程度
        sharpness_ratio = dehazed_quality['sharpness'] / (original_quality['sharpness'] + 1e-6)
        contrast_ratio = dehazed_quality['contrast'] / (original_quality['contrast'] + 1e-6)
        entropy_ratio = dehazed_quality['entropy'] / (original_quality['entropy'] + 1e-6)

        # 加权平均（可以根据实际效果调整权重）
        weight = (sharpness_ratio * 0.4 + contrast_ratio * 0.4 + entropy_ratio * 0.2)

        # 归一化到0-1
        weight = weight / 2.0  # 假设最大改善为2倍
        weight = np.clip(weight, 0.0, 1.0)

        return float(weight)

    def detect_with_fusion(self,
                          img: np.ndarray,
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45,
                          classes: Optional[List[int]] = None,
                          max_det: int = 300) -> Dict:
        """
        执行双路检测并融合结果

        Args:
            img: 输入图像
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS的IoU阈值
            classes: 检测的类别列表
            max_det: 最大检测数量

        Returns:
            包含详细信息的结果字典:
                - original_detections: 原图检测结果
                - dehazed_detections: 去雾图检测结果
                - fused_detections: 融合后的检测结果
                - original_quality: 原图质量评估
                - dehazed_quality: 去雾图质量评估
                - dehazed_image: 去雾后的图像
        """
        # 1. 评估原图质量
        original_quality = self.quality_assessor.assess_image_quality(img)

        # 2. 原图检测
        original_detections = self.detector.detect(
            img,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            max_det=max_det
        )

        # 3. 去雾处理
        dehazed_img = self.dehazer.process(img)

        # 4. 评估去雾图质量
        dehazed_quality = self.quality_assessor.assess_image_quality(dehazed_img)

        # 5. 去雾图检测
        dehazed_detections = self.detector.detect(
            dehazed_img,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            max_det=max_det
        )

        # 6. 融合检测结果
        if self.fusion_strategy == 'adaptive':
            fused_detections = self.fuse_detections_adaptive(
                original_detections, dehazed_detections,
                original_quality, dehazed_quality
            )
        elif self.fusion_strategy == 'confidence':
            fused_detections = self.fuse_detections_confidence(
                original_detections, dehazed_detections
            )
        elif self.fusion_strategy == 'quality':
            fused_detections = self.fuse_detections_quality(
                original_detections, dehazed_detections,
                original_quality, dehazed_quality
            )
        elif self.fusion_strategy == 'both':
            fused_detections = self.fuse_detections_both(
                original_detections, dehazed_detections
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return {
            'original_detections': original_detections,
            'dehazed_detections': dehazed_detections,
            'fused_detections': fused_detections,
            'original_quality': original_quality,
            'dehazed_quality': dehazed_quality,
            'dehazed_image': dehazed_img
        }
