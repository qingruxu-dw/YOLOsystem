"""
Feature-level Fusion YOLO 训练脚本 - 两阶段训练版本

阶段1: 冻结YOLO，只训练融合模块
阶段2: 解冻YOLO，端到端微调
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import time
from datetime import datetime

# 尝试导入 YOLOv8 Loss (需要 ultralytics 包)
try:
    from ultralytics.utils.loss import v8DetectionLoss
    HAS_YOLO_LOSS = True
except ImportError:
    print("Warning: 无法导入 v8DetectionLoss, Stage 2 可能无法正常运行")
    HAS_YOLO_LOSS = False

from ultralytics import YOLO
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo


class FusionDataset(Dataset):
    """Feature-level Fusion数据集"""

    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size

        self.original_dir = self.data_dir / split / 'images_original'
        self.dehazed_dir = self.data_dir / split / 'images_dehazed'
        self.label_dir = self.data_dir / split / 'labels'

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
        labels = self.load_labels(label_path)

        # 预处理
        img_orig, img_dehz, labels = self.preprocess(img_orig, img_dehz, labels)

        return {
            'image_original': img_orig,
            'image_dehazed': img_dehz,
            'labels': labels,
            'img_name': img_name
        }

    def load_labels(self, label_path: Path) -> np.ndarray:
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
        # Resize
        img_orig = cv2.resize(img_orig, (self.img_size, self.img_size))
        img_dehz = cv2.resize(img_dehz, (self.img_size, self.img_size))

        # Normalize
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
    images_orig = torch.stack([item['image_original'] for item in batch])
    images_dehz = torch.stack([item['image_dehazed'] for item in batch])
    labels = [item['labels'] for item in batch]
    img_names = [item['img_name'] for item in batch]

    return {
        'images_original': images_orig,
        'images_dehazed': images_dehz,
        'labels': labels,
        'img_names': img_names
    }


class TwoStageTrainer:
    """两阶段训练器"""

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
            fusion_type=args.fusion_type,
            pretrained=args.pretrained
        )
        self.model = self.model.to(self.device)

        # 加载检查点（如果指定）
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            print(f"\n加载检查点: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 模型权重加载成功")
            else:
                self.model.load_state_dict(checkpoint) # 尝试直接加载
                print("✓ 模型权重加载成功 (即使是直接的全模型保存)")

        # 加载数据
        print("\n加载数据集...")
        self.train_loader = self.create_dataloader('train')
        self.val_loader = self.create_dataloader('val')

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def create_dataloader(self, split: str):
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

    def stage1_train(self):
        """
        阶段1: 冻结YOLO，只训练融合模块
        使用简单的一致性损失
        """
        print("\n" + "=" * 60)
        print("阶段1: 训练融合模块（YOLO冻结）")
        print("=" * 60)

        # 冻结YOLO参数
        for param in self.model.yolo.parameters():
            param.requires_grad = False

        # 只优化融合模块
        optimizer = torch.optim.Adam(
            self.model.fusion_module.parameters(),
            lr=self.args.stage1_lr
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.stage1_epochs
        )

        best_loss = float('inf')

        for epoch in range(self.args.stage1_epochs):
            # 训练
            train_loss = self.train_epoch_stage1(epoch, optimizer)

            # 学习率调度
            scheduler.step()

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, 'stage1_best.pth')

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'stage1_epoch_{epoch}.pth')

        print(f"\n阶段1训练完成！最佳损失: {best_loss:.4f}")

    def train_epoch_stage1(self, epoch: int, optimizer):
        """阶段1的训练epoch"""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch}/{self.args.stage1_epochs}")
        total_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            images_orig = batch['images_original'].to(self.device)
            images_dehz = batch['images_dehazed'].to(self.device)

            optimizer.zero_grad()

            # 获取融合后的图像
            fused_img = self.model.fusion_module(images_orig, images_dehz)

            # 简单的一致性损失：融合图像应该接近原图和去雾图
            # 这鼓励融合模块学习有意义的权重
            loss_orig = torch.mean((fused_img - images_orig) ** 2)
            loss_dehz = torch.mean((fused_img - images_dehz) ** 2)

            # 总损失：平衡两者
            loss = 0.5 * loss_orig + 0.5 * loss_dehz

            # 添加融合权重的正则化（鼓励权重不要太极端）
            if hasattr(self.model.fusion_module, 'weight_original'):
                w_orig = torch.sigmoid(self.model.fusion_module.weight_original)
                w_dehz = torch.sigmoid(self.model.fusion_module.weight_dehazed)
                # 鼓励权重接近0.5（平衡）
                reg_loss = torch.abs(w_orig - 0.5) + torch.abs(w_dehz - 0.5)
                loss = loss + 0.01 * reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'args': vars(self.args)
        }

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  保存检查点: {save_path}")

    def train(self):
        """完整训练流程"""
        start_time = time.time()

        # 阶段1
        self.stage1_train()

        # 保存最终模型
        self.save_checkpoint(self.args.stage1_epochs - 1, 'final_stage1.pth')

        elapsed = time.time() - start_time
        print(f"\n总训练时间: {elapsed / 3600:.2f} 小时")

    def stage2_train(self):
        """
        阶段2: 解冻YOLO，端到端微调
        使用YOLOv8的真实检测损失
        """
        print("\n" + "=" * 60)
        print("阶段2: 端到端微调（YOLO解冻）")
        print("=" * 60)

        # 解冻YOLO参数
        for param in self.model.yolo.parameters():
            param.requires_grad = True

        # 优化器：同时优化YOLO和融合模块
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.stage2_lr
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.stage2_epochs
        )

        best_loss = float('inf')

        for epoch in range(self.args.stage2_epochs):
            # 训练
            train_loss = self.train_epoch_stage2(epoch, optimizer)

            # 学习率调度
            scheduler.step()

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, 'stage2_best.pth')

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'stage2_epoch_{epoch}.pth')

        print(f"\n阶段2训练完成！最佳损失: {best_loss:.4f}")

    def train_epoch_stage2(self, epoch: int, optimizer):
        """阶段2的训练epoch"""
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Stage2 Epoch {epoch}/{self.args.stage2_epochs}")
        total_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            images_orig = batch['images_original'].to(self.device)
            images_dehz = batch['images_dehazed'].to(self.device)

            # 正确处理 targets: list[Tensor] -> Tensor [N, 6]
            # YOLO需要格式: [batch_idx, class_idx, x, y, w, h]
            targets_list = []
            for i, label in enumerate(batch['labels']):
                # label 是一个 Tensor [num_boxes, 5]
                if len(label) > 0:
                    # 创建 batch index 列
                    batch_idx_tensor = torch.full((len(label), 1), i)
                    # 拼接: [batch_idx, cls, x, y, w, h]
                    target = torch.cat((batch_idx_tensor, label), 1)
                    targets_list.append(target)

            if targets_list:
                targets = torch.cat(targets_list, 0).to(self.device)
            else:
                targets = torch.zeros((0, 6)).to(self.device)

            optimizer.zero_grad()
            
            # 1. 前向传播得到融合图像
            fused_imgs = self.model.fusion_module(images_orig, images_dehz)
            
            # 2. 将融合图像送入 YOLO 进行检测计算 Loss
            if HAS_YOLO_LOSS:
                try:
                    # 初始化 Loss (仅一次)
                    if not hasattr(self, 'loss_criterion'):
                        # v8DetectionLoss 需要 hyp 参数，通常在 model.args 中
                        # 如果 model.args 是 dict，我们需要把它转换成 object 或者确保 Loss 能处理
                        
                        # 确保我们使用的是正确的 Detect Head
                        # feature_fusion_yolo.py 中的 self.neck_head 是 nn.ModuleList
                        # 真正的 Detect 层在最后
                        
                        # 尝试手动注入 hyp 参数（因为我们的 model 是自定义的，可能丢失了 args）
                        m = self.model.yolo.model
                        if not hasattr(m, 'args'):
                            # 创建默认参数
                            m.args = {
                                'box': 7.5,
                                'cls': 0.5,
                                'dfl': 1.5,
                            }
                        elif isinstance(m.args, dict):
                            # 如果 args 是 dict，确保包含了 loss 需要的 box/cls/dfl
                            if 'box' not in m.args: m.args['box'] = 7.5
                            if 'cls' not in m.args: m.args['cls'] = 0.5
                            if 'dfl' not in m.args: m.args['dfl'] = 1.5
                        
                        # 有些版本的 Ultralytics Loss 需要 args 是 namespace 而不是 dict
                        # 我们可以创建一个简单的类来模拟 Namespace
                        class ArgsNamespace:
                            def __init__(self, d):
                                for k, v in d.items():
                                    setattr(self, k, v)
                        
                        if isinstance(m.args, dict):
                            m.args = ArgsNamespace(m.args)

                        self.loss_criterion = v8DetectionLoss(m)
                    
                    # YOLO模型前向传播
                    preds = self.model.yolo.model(fused_imgs)
                    
                    # 准备 Loss 输入
                    # targets 格式: [batch_idx, class_idx, x, y, w, h] (normalized)
                    batch_data = {
                        'batch_idx': targets[:, 0],
                        'cls': targets[:, 1].view(-1, 1),
                        'bboxes': targets[:, 2:],
                        'device': self.device,
                        'img': fused_imgs  # for calculating grid size
                    }
                    
                    # 计算 Loss
                    loss_result = self.loss_criterion(preds, batch_data)
                    
                    # 某些版本的 v8DetectionLoss 返回 (loss, loss_items)
                    # 某些版本可能只返回 loss
                    if isinstance(loss_result, tuple):
                        loss = loss_result[0]
                    else:
                        loss = loss_result
                        
                    # 确保 loss 是 scalar
                    if loss.numel() > 1:
                        loss = loss.sum()

                except Exception as e:
                    # 如果出错了 (可能是 API 版本变动)
                    # 退回到 dummy loss 以防 crash
                    if batch_idx == 0:
                        print(f"Loss calculation error: {e}")
                        import traceback
                        traceback.print_exc()
                        
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            else:
                 loss = torch.tensor(0.0, requires_grad=True).to(self.device)

            # 只有当 loss 有梯度时才后向传播
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def resume_training(self, checkpoint_path: str):
        """从检查点恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复历史记录
        self.history = checkpoint['history']

        # 恢复优化器状态
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        print(f"从 {checkpoint_path} 恢复训练，当前轮次: {checkpoint['epoch']}")


def parse_args():
    parser = argparse.ArgumentParser(description='Feature-level Fusion YOLO 两阶段训练')

    # 数据参数
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--num-classes', type=int, default=10, help='类别数量')

    # 模型参数
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO模型大小')
    parser.add_argument('--fusion-type', type=str, default='learned',
                        choices=['average', 'learned', 'attention'], help='融合类型')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练权重')

    # 阶段1训练参数
    parser.add_argument('--stage1-epochs', type=int, default=30, help='阶段1训练轮数')
    parser.add_argument('--stage1-lr', type=float, default=0.001, help='阶段1学习率')

    # 阶段2训练参数
    parser.add_argument('--do-stage2', action='store_true', help='是否执行阶段2训练')
    parser.add_argument('--stage2-epochs', type=int, default=50, help='阶段2训练轮数')
    parser.add_argument('--stage2-lr', type=float, default=0.0001, help='阶段2学习率')

    # 通用训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/two_stage_training',
                        help='输出目录')

    # 恢复训练参数
    parser.add_argument('--resume-checkpoint', type=str, default='', help='恢复训练的检查点路径')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature-level Fusion YOLO 两阶段训练")
    print("=" * 60)

    # 创建训练器
    trainer = TwoStageTrainer(args)

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"阶段1 epochs: {args.stage1_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"阶段1学习率: {args.stage1_lr}")
    print(f"输出目录: {args.output_dir}")

    # 如果指定了恢复检查点，则加载并恢复训练
    if args.resume_checkpoint:
        trainer.resume_training(args.resume_checkpoint)
    else:
        trainer.train()

    # 如果选择执行阶段2，则开始阶段2训练
    if args.do_stage2:
        trainer.stage2_train()

    print("\n训练完成！")


if __name__ == '__main__':
    main()
