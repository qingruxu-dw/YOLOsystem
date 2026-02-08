"""
Feature-level Fusion YOLO 评估脚本

用于评估训练好的Feature-level Fusion模型
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import Dict, List
import time

from yolosystem.feature_fusion_yolo import create_feature_fusion_yolo
from yolosystem import DehazingModule


class FeatureFusionEvaluator:
    """Feature-level Fusion模型评估器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: 模型权重路径
            device: 设备 ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = create_feature_fusion_yolo(model_size='n', num_classes=6)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 去雾模块
        self.dehazing = DehazingModule()

        # 类别名称
        self.class_names = ['car', 'motorcycle', 'person', 'truck', 'bus', 'bicycle']

    def preprocess_image(self, image: np.ndarray, img_size: int = 640) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: BGR图像
            img_size: 目标大小

        Returns:
            预处理后的tensor [1, 3, H, W]
        """
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (img_size, img_size))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = torch.from_numpy(image).unsqueeze(0).float()

        return image

    @torch.no_grad()
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """
        对单张图像进行预测

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值

        Returns:
            检测结果字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 去雾
        dehazed = self.dehazing.process(image)

        # 预处理
        img_orig = self.preprocess_image(image).to(self.device)
        img_dehz = self.preprocess_image(dehazed).to(self.device)

        # 推理
        start_time = time.time()
        outputs = self.model(img_orig, img_dehz)
        inference_time = time.time() - start_time

        # 后处理（这里需要根据实际输出格式实现）
        # 暂时返回占位符
        detections = []

        return {
            'detections': detections,
            'inference_time': inference_time,
            'image_path': image_path
        }

    def evaluate_dataset(self, data_dir: str, split: str = 'test') -> Dict:
        """
        在数据集上评估

        Args:
            data_dir: 数据集目录
            split: 'train', 'val', 'test'

        Returns:
            评估结果
        """
        print(f"\n评估 {split} 数据集...")

        # 图像目录
        image_dir = Path(data_dir) / split / 'images_original'
        label_dir = Path(data_dir) / split / 'labels'

        # 获取所有图像
        image_files = sorted(list(image_dir.glob('*.jpg')))
        print(f"找到 {len(image_files)} 张图像")

        # 统计信息
        total_detections = 0
        total_time = 0.0
        results = []

        # 逐张评估
        for img_path in tqdm(image_files, desc="评估中"):
            result = self.predict(str(img_path))

            total_detections += len(result['detections'])
            total_time += result['inference_time']
            results.append(result)

        # 计算指标
        avg_time = total_time / len(image_files)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        summary = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files),
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'fps': fps,
            'results': results
        }

        return summary

    def compare_with_baseline(self, data_dir: str, split: str = 'test') -> Dict:
        """
        与基线方法对比

        对比：
        1. Feature-level Fusion (本模型)
        2. 简单串联 (去雾 -> 检测)
        3. 仅原图检测
        4. 仅去雾图检测

        Args:
            data_dir: 数据集目录
            split: 数据集划分

        Returns:
            对比结果
        """
        print("\n" + "=" * 60)
        print("与基线方法对比")
        print("=" * 60)

        # 评估Feature-level Fusion
        print("\n1. Feature-level Fusion")
        fusion_results = self.evaluate_dataset(data_dir, split)

        # TODO: 实现其他基线方法的评估

        comparison = {
            'feature_fusion': fusion_results,
            # 'simple_cascade': cascade_results,
            # 'original_only': original_results,
            # 'dehazed_only': dehazed_results,
        }

        return comparison

    def visualize_results(self, image_path: str, output_path: str):
        """
        可视化检测结果

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        dehazed = self.dehazing.process(image)

        # 预测
        result = self.predict(image_path)

        # 绘制检测框
        vis_image = image.copy()
        for det in result['detections']:
            # 绘制边界框和标签
            # TODO: 实现可视化逻辑
            pass

        # 保存
        cv2.imwrite(output_path, vis_image)
        print(f"结果已保存: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Feature-level Fusion YOLO评估')

    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--data-dir', type=str, default='datasets/fusion_training',
                       help='数据集目录')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='评估哪个数据集')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出结果文件')
    parser.add_argument('--compare', action='store_true',
                       help='与基线方法对比')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("Feature-level Fusion YOLO 评估")
    print("=" * 60)

    # 创建评估器
    evaluator = FeatureFusionEvaluator(
        model_path=args.model,
        device=args.device
    )

    # 评估
    if args.compare:
        # 与基线对比
        results = evaluator.compare_with_baseline(args.data_dir, args.split)
    else:
        # 仅评估本模型
        results = evaluator.evaluate_dataset(args.data_dir, args.split)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总图像数: {results.get('total_images', 0)}")
    print(f"总检测数: {results.get('total_detections', 0)}")
    print(f"平均检测数/图: {results.get('avg_detections_per_image', 0):.2f}")
    print(f"平均推理时间: {results.get('avg_inference_time', 0):.4f}s")
    print(f"FPS: {results.get('fps', 0):.2f}")

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
