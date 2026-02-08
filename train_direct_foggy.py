"""
简化版端到端训练
使用已有的校园数据集（有雾图像+标注）
"""

from ultralytics import YOLO
from pathlib import Path
import yaml


def train_on_foggy_school_data(
    model_size: str = 's',
    epochs: int = 50,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """
    在校园有雾数据上训练YOLO

    策略：直接用有雾图像训练，让模型学习适应雾天场景

    Args:
        model_size: YOLO模型大小
        epochs: 训练轮数
        batch_size: 批次大小
        device: 设备
    """
    print("=" * 60)
    print("端到端雾天YOLO训练（校园数据集）")
    print("=" * 60)

    # 创建数据配置
    data_config = {
        'path': '/root/autodl-tmp/datasets/school_foggy',
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'bus',
            5: 'truck'
        }
    }

    # 保存配置
    config_path = 'school_foggy_data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f)

    print(f"\n✓ 创建数据配置: {config_path}")

    # 加载YOLO
    model = YOLO(f'yolo11{model_size}.pt')

    print(f"\n开始训练...")
    print(f"  模型: YOLOv11-{model_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  设备: {device}")

    # 训练
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project='runs/fog_aware',
        name='school_direct',
        exist_ok=True,

        # 数据增强（帮助适应雾天）
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

        # 优化器
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 其他
        patience=10,
        save=True,
        save_period=10,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    print(f"最佳权重: runs/fog_aware/school_direct/weights/best.pt")
    print(f"最后权重: runs/fog_aware/school_direct/weights/last.pt")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='端到端雾天YOLO训练')
    parser.add_argument('--model-size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO模型大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    train_on_foggy_school_data(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
