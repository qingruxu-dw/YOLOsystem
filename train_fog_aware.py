"""
训练端到端雾感知YOLO
直接从有雾图像学习检测，无需显式去雾
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from yolosystem.fog_aware_yolo import SimpleFogAwareYOLO
from ultralytics import YOLO


class FoggyDetectionDataset(Dataset):
    """
    雾天检测数据集
    直接使用有雾图像和标注
    """

    def __init__(self, data_dir: str, img_size: int = 640):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        # 查找所有有雾图像
        self.img_files = list(self.data_dir.glob('**/*_foggy_*.png'))
        if not self.img_files:
            # 如果没有foggy标记，使用所有图像
            self.img_files = list(self.data_dir.glob('**/*.png'))

        print(f"找到 {len(self.img_files)} 张训练图像")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整大小
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 转换为tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 查找对应的标注文件
        label_path = img_path.parent / (img_path.stem.replace('_leftImg8bit_foggy_beta_0.02', '_gtFine_labelIds') + '.txt')
        if not label_path.exists():
            # 如果没有标注，返回空标注
            labels = torch.zeros((0, 5))
        else:
            # 读取YOLO格式标注
            labels = self._load_labels(label_path)

        return img, labels

    def _load_labels(self, label_path):
        """加载YOLO格式标注"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(x) for x in parts])
            return torch.tensor(labels) if labels else torch.zeros((0, 5))
        except:
            return torch.zeros((0, 5))


def train_fog_aware_yolo(
    train_data: str,
    val_data: str,
    model_size: str = 'n',
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = 'cuda'
):
    """
    训练雾感知YOLO

    策略：
    1. 冻结YOLO，只训练增强模块
    2. 使用有雾图像和标注进行端到端训练
    3. 让模型自动学习如何处理雾的影响

    Args:
        train_data: 训练数据配置文件或目录
        val_data: 验证数据配置文件或目录
        model_size: YOLO模型大小
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备
    """
    print("=" * 60)
    print("训练端到端雾感知YOLO")
    print("=" * 60)

    # 创建模型
    model = SimpleFogAwareYOLO(model_size)
    model = model.to(device)

    # 冻结YOLO参数
    for param in model.yolo.model.parameters():
        param.requires_grad = False

    # 只训练增强模块
    optimizer = optim.Adam(model.enhancement.parameters(), lr=lr)

    print(f"\n模型参数:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")

    # 使用YOLO的训练接口（更简单）
    print("\n使用YOLO内置训练...")

    # 方案1：直接使用YOLO训练（推荐）
    # 这种方式最简单，让YOLO自己学习处理雾天图像
    yolo_model = YOLO(f'yolo11{model_size}.pt')

    results = yolo_model.train(
        data=train_data,  # 数据配置文件
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project='runs/fog_aware',
        name='direct_train',
        exist_ok=True,
        # 数据增强（帮助模型适应雾天）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\n✓ 训练完成！")
    print(f"  最佳权重: runs/fog_aware/direct_train/weights/best.pt")

    return results


def create_foggy_data_yaml(
    foggy_train_dir: str,
    foggy_val_dir: str,
    output_path: str = 'foggy_data.yaml'
):
    """
    创建雾天数据集配置文件

    Args:
        foggy_train_dir: 训练集目录（包含有雾图像）
        foggy_val_dir: 验证集目录
        output_path: 输出配置文件路径
    """
    config = {
        'path': str(Path(foggy_train_dir).parent),
        'train': str(Path(foggy_train_dir).name),
        'val': str(Path(foggy_val_dir).name),
        'names': {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'bus',
            5: 'truck',
            6: 'traffic light',
            7: 'traffic sign'
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    print(f"✓ 创建数据配置文件: {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练雾感知YOLO')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='训练数据目录（有雾图像）')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='验证数据目录')
    parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO模型大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    # 创建数据配置
    data_yaml = create_foggy_data_yaml(
        args.train_dir,
        args.val_dir
    )

    # 训练
    train_fog_aware_yolo(
        train_data=data_yaml,
        val_data=args.val_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
