"""
Feature-level Fusion YOLO 训练脚本

用于在AutoDL等云服务器上训练Feature-level Fusion模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import time
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from yolosystem.feature_fusion_yolo import create_feature_fusion_yolo


class FusionDataset(Dataset):
    """
    Feature-level Fusion数据集

    返回配对的原图和去雾图
    """

    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 640):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train', 'val', 'test'
            img_size: 图像大小
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size

        # 图像和标注路径
        self.original_dir = self.data_dir / split / 'images_original'
        self.dehazed_dir = self.data_dir / split / 'images_dehazed'
        self.label_dir = self.data_dir / split / 'labels'

        # 获取所有图像文件
        self.image_files = sorted([f.stem for f in self.original_dir.glob('*.jpg')])

        print(f"加载 {split} 数据集: {len(self.image_files)} 对图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # 读取原图
        img_orig_path = self.original_dir / f"{img_name}.jpg"
        img_orig = cv2.imread(str(img_orig_path))
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        # 读取去雾图
        img_dehz_path = self.dehazed_dir / f"{img_name}.jpg"
        img_dehz = cv2.imread(str(img_dehz_path))
        img_dehz = cv2.cvtColor(img_dehz, cv2.COLOR_BGR2RGB)

        # 读取标注
        label_path = self.label_dir / f"{img_name}.txt"
        labels = self.load_labels(label_path, img_orig.shape[:2])

        # 预处理
        img_orig, img_dehz, labels = self.preprocess(img_orig, img_dehz, labels)

        return {
            'image_original': img_orig,
            'image_dehazed': img_dehz,
            'labels': labels,
            'img_name': img_name
        }

    def load_labels(self, label_path: Path, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        加载YOLO格式标注

        Returns:
            labels: [N, 5] (class, x_center, y_center, width, height)
        """
        if not label_path.exists():
            return np.zeros((0, 5))

        with open(label_path, 'r') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(x) for x in parts])

        return np.array(labels) if labels else np.zeros((0, 5))

    def preprocess(self, img_orig, img_dehz, labels):
        """
        预处理：resize + normalize
        """
        h, w = img_orig.shape[:2]

        # Resize
        img_orig = cv2.resize(img_orig, (self.img_size, self.img_size))
        img_dehz = cv2.resize(img_dehz, (self.img_size, self.img_size))

        # Normalize to [0, 1]
        img_orig = img_orig.astype(np.float32) / 255.0
        img_dehz = img_dehz.astype(np.float32) / 255.0

        # HWC -> CHW
        img_orig = np.transpose(img_orig, (2, 0, 1))
        img_dehz = np.transpose(img_dehz, (2, 0, 1))

        # Convert to tensor
        img_orig = torch.from_numpy(img_orig).float()
        img_dehz = torch.from_numpy(img_dehz).float()
        labels = torch.from_numpy(labels).float()

        return img_orig, img_dehz, labels


def collate_fn(batch):
    """自定义collate函数"""
    images_orig = torch.stack([item['image_original'] for item in batch])
    images_dehz = torch.stack([item['image_dehazed'] for item in batch])

    # 标注需要特殊处理（不同图像的目标数量不同）
    labels = [item['labels'] for item in batch]
    img_names = [item['img_name'] for item in batch]

    return {
        'images_original': images_orig,
        'images_dehazed': images_dehz,
        'labels': labels,
        'img_names': img_names
    }


class FeatureFusionTrainer:
    """Feature-level Fusion训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        print("\n创建Feature-level Fusion模型...")
        self.model = create_feature_fusion_yolo(
            model_size=args.model_size,
            num_classes=args.num_classes,
            pretrained=args.pretrained
        )
        self.model = self.model.to(self.device)

        # 创建数据加载器
        print("\n加载数据集...")
        self.train_loader = self.create_dataloader('train')
        self.val_loader = self.create_dataloader('val')

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # 损失函数（使用YOLOv11的损失）
        # 这里简化处理，实际需要使用完整的YOLO损失
        self.criterion = nn.MSELoss()  # 占位符

        # 训练状态
        self.start_epoch = 0
        self.best_map = 0.0

    def create_dataloader(self, split: str):
        """创建数据加载器"""
        dataset = FusionDataset(
            data_dir=self.args.data_dir,
            split=split,
            img_size=self.args.img_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.args.workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        return dataloader

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        total_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # 数据移到GPU
            images_orig = batch['images_original'].to(self.device)
            images_dehz = batch['images_dehazed'].to(self.device)
            labels = batch['labels']

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images_orig, images_dehz)

            # 计算损失（这里需要实现完整的YOLO损失）
            # 暂时使用占位符
            loss = self.criterion(outputs, outputs)  # 占位符

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 更新进度条
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()

        print("\n验证中...")
        # TODO: 实现完整的验证逻辑（mAP计算等）

        return 0.0  # 占位符

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
        }

        # 保存最新模型
        checkpoint_path = self.output_dir / 'last.pt'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳模型: {best_path}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"总epochs: {self.args.epochs}")
        print(f"批次大小: {self.args.batch_size}")
        print(f"学习率: {self.args.lr}")
        print(f"输出目录: {self.output_dir}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_map = self.validate(epoch)

                # 保存最佳模型
                is_best = val_map > self.best_map
                if is_best:
                    self.best_map = val_map

                self.save_checkpoint(epoch, is_best)

            # 更新学习率
            self.scheduler.step()

            # 打印统计
            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

        # 训练完成
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"总耗时: {total_time / 3600:.2f} 小时")
        print(f"最佳mAP: {self.best_map:.4f}")
        print(f"模型保存在: {self.output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Feature-level Fusion YOLO训练')

    # 数据集
    parser.add_argument('--data-dir', type=str, default='datasets/fusion_training',
                       help='数据集目录')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='类别数量')

    # 模型
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用预训练权重')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像大小')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='权重衰减')
    parser.add_argument('--workers', type=int, default=8,
                       help='数据加载线程数')
    parser.add_argument('--val-interval', type=int, default=5,
                       help='验证间隔（epochs）')

    # 输出
    parser.add_argument('--output-dir', type=str, default='runs/feature_fusion',
                       help='输出目录')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("Feature-level Fusion YOLO 训练")
    print("=" * 60)

    # 创建训练器
    trainer = FeatureFusionTrainer(args)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
