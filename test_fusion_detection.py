"""
测试Feature-level Fusion的检测性能

对比三种方案：
1. 原图检测
2. 去雾图检测
3. 融合检测
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


class DetectionComparison:
    """检测性能对比"""

    def __init__(self, fusion_checkpoint: str, model_size: str = 's', conf_threshold: float = 0.25):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold

        print(f"使用设备: {self.device}")

        # 加载融合模型
        print("\n加载融合模型...")
        self.fusion_model = create_feature_fusion_yolo(
            model_size=model_size,
            num_classes=6,
            fusion_type='learned',
            pretrained=True
        )
        checkpoint = torch.load(fusion_checkpoint, map_location=self.device)
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()
        print(f"✓ 融合模型加载成功 (epoch {checkpoint['epoch']})")

        # 加载标准YOLO（用于对比）
        print("\n加载标准YOLO模型...")
        self.yolo_model = YOLO(f'yolo11{model_size}.pt')
        print("✓ 标准YOLO加载成功")

        # 类别名称
        self.class_names = ['car', 'motorcycle', 'person', 'truck', 'bus', 'bicycle']

    def detect_with_fusion(self, img_orig, img_dehz):
        """使用融合模型检测"""
        # 预处理
        img_orig_tensor = cv2.resize(img_orig, (640, 640))
        img_dehz_tensor = cv2.resize(img_dehz, (640, 640))

        img_orig_tensor = torch.from_numpy(img_orig_tensor).float().permute(2, 0, 1) / 255.0
        img_dehz_tensor = torch.from_numpy(img_dehz_tensor).float().permute(2, 0, 1) / 255.0

        img_orig_tensor = img_orig_tensor.unsqueeze(0).to(self.device)
        img_dehz_tensor = img_dehz_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            # 获取融合图像
            fused_img = self.fusion_model.fusion_module(img_orig_tensor, img_dehz_tensor)
            # 转换回numpy用于YOLO推理
            fused_img_np = fused_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            fused_img_np = (fused_img_np * 255).clip(0, 255).astype(np.uint8)

        # 使用YOLO检测融合图像
        results = self.yolo_model(fused_img_np, conf=self.conf_threshold, verbose=False)

        return self.parse_results(results[0])

    def detect_with_yolo(self, img):
        """使用标准YOLO检测"""
        img_resized = cv2.resize(img, (640, 640))
        results = self.yolo_model(img_resized, conf=self.conf_threshold, verbose=False)
        return self.parse_results(results[0])

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
                    'class': int(cls),
                    'class_name': self.class_names[cls] if cls < len(self.class_names) else 'unknown'
                })

        return detections

    def draw_detections(self, img, detections, title):
        """绘制检测框"""
        img_draw = img.copy()
        h, w = img.shape[:2]

        # 缩放比例（从640到原始尺寸）
        scale_x = w / 640
        scale_y = h / 640

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            # 绘制框
            color = (0, 255, 0)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img_draw, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 添加标题
        cv2.putText(img_draw, f"{title} ({len(detections)} objects)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img_draw

    def test_on_dataset(self, data_dir: str, output_dir: Path):
        """在数据集上测试"""
        data_path = Path(data_dir)
        val_orig_dir = data_path / 'val' / 'images_original'
        val_dehz_dir = data_path / 'val' / 'images_dehazed'

        img_files = list(val_orig_dir.glob('*.jpg'))

        print(f"\n测试图像数量: {len(img_files)}")

        results = {
            'original': {'total_detections': 0, 'avg_confidence': [], 'per_image': []},
            'dehazed': {'total_detections': 0, 'avg_confidence': [], 'per_image': []},
            'fusion': {'total_detections': 0, 'avg_confidence': [], 'per_image': []}
        }

        # 创建可视化目录
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_file in enumerate(tqdm(img_files, desc="检测中")):
            # 读取图像
            img_orig = cv2.imread(str(img_file))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            img_dehz_path = val_dehz_dir / img_file.name
            img_dehz = cv2.imread(str(img_dehz_path))
            img_dehz = cv2.cvtColor(img_dehz, cv2.COLOR_BGR2RGB)

            # 三种检测
            det_orig = self.detect_with_yolo(img_orig)
            det_dehz = self.detect_with_yolo(img_dehz)
            det_fusion = self.detect_with_fusion(img_orig, img_dehz)

            # 统计
            results['original']['total_detections'] += len(det_orig)
            results['dehazed']['total_detections'] += len(det_dehz)
            results['fusion']['total_detections'] += len(det_fusion)

            if det_orig:
                results['original']['avg_confidence'].extend([d['confidence'] for d in det_orig])
            if det_dehz:
                results['dehazed']['avg_confidence'].extend([d['confidence'] for d in det_dehz])
            if det_fusion:
                results['fusion']['avg_confidence'].extend([d['confidence'] for d in det_fusion])

            results['original']['per_image'].append(len(det_orig))
            results['dehazed']['per_image'].append(len(det_dehz))
            results['fusion']['per_image'].append(len(det_fusion))

            # 可视化前5张
            if idx < 5:
                img_orig_draw = self.draw_detections(img_orig, det_orig, 'Original')
                img_dehz_draw = self.draw_detections(img_dehz, det_dehz, 'Dehazed')
                img_fusion_draw = self.draw_detections(img_orig, det_fusion, 'Fusion')

                # 拼接
                combined = np.hstack([img_orig_draw, img_dehz_draw, img_fusion_draw])
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

                save_path = vis_dir / f'comparison_{idx}.jpg'
                cv2.imwrite(str(save_path), combined)

        return results

    def print_results(self, results):
        """打印结果"""
        print("\n" + "="*60)
        print("检测性能对比")
        print("="*60)

        for method in ['original', 'dehazed', 'fusion']:
            data = results[method]
            total = data['total_detections']
            avg_conf = np.mean(data['avg_confidence']) if data['avg_confidence'] else 0
            avg_per_img = np.mean(data['per_image'])

            print(f"\n{method.upper()}:")
            print(f"  总检测数: {total}")
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  平均每张图: {avg_per_img:.1f}")

        # 对比
        print("\n" + "="*60)
        print("性能提升")
        print("="*60)

        orig_total = results['original']['total_detections']
        dehz_total = results['dehazed']['total_detections']
        fusion_total = results['fusion']['total_detections']

        print(f"\n融合 vs 原图:")
        print(f"  检测数提升: {fusion_total - orig_total} ({(fusion_total - orig_total) / orig_total * 100:.1f}%)")

        print(f"\n融合 vs 去雾:")
        print(f"  检测数提升: {fusion_total - dehz_total} ({(fusion_total - dehz_total) / dehz_total * 100:.1f}%)")

        # 判断
        if fusion_total > max(orig_total, dehz_total):
            print("\n✓ 融合检测效果最好！")
        elif fusion_total > min(orig_total, dehz_total):
            print("\n⚠ 融合检测效果中等")
        else:
            print("\n✗ 融合检测效果不如单独检测")


def main():
    parser = argparse.ArgumentParser(description='测试Feature-level Fusion检测性能')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型检查点')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--model-size', type=str, default='s', help='模型大小')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--output-dir', type=str, default='detection_results', help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Feature-level Fusion 检测性能测试")
    print("="*60)

    # 创建对比器
    comparator = DetectionComparison(
        fusion_checkpoint=args.checkpoint,
        model_size=args.model_size,
        conf_threshold=args.conf_threshold
    )

    # 测试
    results = comparator.test_on_dataset(args.data_dir, output_dir)

    # 打印结果
    comparator.print_results(results)

    # 保存结果
    results_file = output_dir / 'detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {results_file}")
    print(f"可视化已保存: {output_dir / 'visualizations'}")


if __name__ == '__main__':
    main()
