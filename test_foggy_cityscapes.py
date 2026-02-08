"""
在Foggy Cityscapes数据集上测试融合检测

Foggy Cityscapes特点：
- 真实的城市道路场景
- 3种雾浓度：β=0.005(轻雾), 0.01(中雾), 0.02(浓雾)
- 有对应的清晰图（用于训练去雾算法）
- 有COCO格式的标注

测试目标：
1. 证明融合在真实雾天场景的效果
2. 对比不同雾浓度下的性能
3. 分析去雾算法的局限性
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from ultralytics import YOLO
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo
from yolosystem.dehazing import DehazingModule


class FoggyCityscapesEvaluator:
    """Foggy Cityscapes评估器"""

    def __init__(self, fusion_checkpoint: str, model_size: str = 's'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 加载融合模型
        print("加载融合模型...")
        self.fusion_model = create_feature_fusion_yolo(
            model_size=model_size,
            num_classes=80,  # COCO类别
            fusion_type='learned',
            pretrained=True
        )
        checkpoint = torch.load(fusion_checkpoint, map_location=self.device)
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()

        # 加载YOLO
        self.yolo_model = YOLO(f'yolo11{model_size}.pt')

        # 去雾算法
        self.dehazer = DehazingModule()

    def test_on_foggy_cityscapes(self, data_dir: str, fog_level: str = '0.02'):
        """
        在Foggy Cityscapes上测试

        Args:
            data_dir: Foggy Cityscapes数据集路径
            fog_level: 雾浓度 ('0.005', '0.01', '0.02')
        """
        data_path = Path(data_dir)

        # 查找包含指定雾浓度的图像
        pattern = f'*_beta_{fog_level}.png'
        img_files = list(data_path.glob(f'**/{pattern}'))[:50]  # 测试50张

        if not img_files:
            print(f"❌ 未找到雾浓度为 {fog_level} 的图像")
            return
        print(f"\n测试图像数量: {len(img_files)}")
        print(f"雾浓度: β={fog_level}")

        results = {
            'foggy': {'detections': 0, 'confidences': []},
            'dehazed': {'detections': 0, 'confidences': []},
            'fusion': {'detections': 0, 'confidences': []}
        }

        for img_file in tqdm(img_files, desc="检测中"):
            # 读取有雾图像
            img_foggy = cv2.imread(str(img_file))
            img_foggy = cv2.cvtColor(img_foggy, cv2.COLOR_BGR2RGB)

            # 去雾（dehaze返回元组：(去雾图像, 中间结果)）
            img_dehazed, _ = self.dehazer.dehaze(img_foggy)

            # 三种检测
            det_foggy = self.detect_with_yolo(img_foggy)
            det_dehazed = self.detect_with_yolo(img_dehazed)
            det_fusion = self.detect_with_fusion(img_foggy, img_dehazed)

            # 统计
            results['foggy']['detections'] += len(det_foggy)
            results['dehazed']['detections'] += len(det_dehazed)
            results['fusion']['detections'] += len(det_fusion)

            if det_foggy:
                results['foggy']['confidences'].extend([d['confidence'] for d in det_foggy])
            if det_dehazed:
                results['dehazed']['confidences'].extend([d['confidence'] for d in det_dehazed])
            if det_fusion:
                results['fusion']['confidences'].extend([d['confidence'] for d in det_fusion])

        return results

    def detect_with_yolo(self, img):
        """使用YOLO检测"""
        img_resized = cv2.resize(img, (640, 640))
        yolo_results = self.yolo_model(img_resized, conf=0.25, verbose=False)
        return self.parse_results(yolo_results[0])

    def detect_with_fusion(self, img_foggy, img_dehazed):
        """使用融合模型检测"""
        # 预处理
        img_foggy_tensor = cv2.resize(img_foggy, (640, 640))
        img_dehazed_tensor = cv2.resize(img_dehazed, (640, 640))

        img_foggy_tensor = torch.from_numpy(img_foggy_tensor).float().permute(2, 0, 1) / 255.0
        img_dehazed_tensor = torch.from_numpy(img_dehazed_tensor).float().permute(2, 0, 1) / 255.0

        img_foggy_tensor = img_foggy_tensor.unsqueeze(0).to(self.device)
        img_dehazed_tensor = img_dehazed_tensor.unsqueeze(0).to(self.device)

        # 融合
        with torch.no_grad():
            fused_img = self.fusion_model.fusion_module(img_foggy_tensor, img_dehazed_tensor)
            fused_img_np = fused_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            fused_img_np = (fused_img_np * 255).clip(0, 255).astype(np.uint8)

        # 检测
        yolo_results = self.yolo_model(fused_img_np, conf=0.25, verbose=False)
        return self.parse_results(yolo_results[0])

    def parse_results(self, result):
        """解析YOLO结果"""
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls)
                })
        return detections

    def print_results(self, results, fog_level):
        """打印结果"""
        print("\n" + "="*60)
        print(f"Foggy Cityscapes 测试结果 (β={fog_level})")
        print("="*60)

        for method in ['foggy', 'dehazed', 'fusion']:
            data = results[method]
            total = data['detections']
            avg_conf = np.mean(data['confidences']) if data['confidences'] else 0

            print(f"\n{method.upper()}:")
            print(f"  总检测数: {total}")
            print(f"  平均置信度: {avg_conf:.3f}")

        # 性能提升
        foggy_total = results['foggy']['detections']
        fusion_total = results['fusion']['detections']

        print("\n" + "="*60)
        print("性能提升")
        print("="*60)
        print(f"融合 vs 有雾图: +{fusion_total - foggy_total} ({(fusion_total - foggy_total) / foggy_total * 100:.1f}%)")

        if fusion_total > foggy_total:
            print("✓ 融合有效提升了雾天检测性能！")
        else:
            print("⚠ 融合未能提升性能")


def main():
    parser = argparse.ArgumentParser(description='在Foggy Cityscapes上测试')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型检查点')
    parser.add_argument('--data-dir', type=str, required=True, help='Foggy Cityscapes路径')
    parser.add_argument('--fog-level', type=str, default='0.02',
                       choices=['0.005', '0.01', '0.02'], help='雾浓度')
    parser.add_argument('--model-size', type=str, default='s', help='模型大小')

    args = parser.parse_args()

    evaluator = FoggyCityscapesEvaluator(args.checkpoint, args.model_size)
    results = evaluator.test_on_foggy_cityscapes(args.data_dir, args.fog_level)
    evaluator.print_results(results, args.fog_level)


if __name__ == '__main__':
    main()
