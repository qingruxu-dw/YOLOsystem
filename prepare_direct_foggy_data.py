"""
准备端到端训练数据
使用有雾图像（hazy_school）+ 标注
"""

import shutil
from pathlib import Path
import yaml


def prepare_foggy_training_data(
    hazy_dir: str = 'datasets/school/hazy_school',
    labels_dir: str = 'datasets/school/labels',
    output_dir: str = 'datasets/school_foggy'
):
    """
    准备端到端训练数据

    使用有雾图像作为输入，对应的标注作为目标
    让模型直接学习从有雾图像进行检测

    Args:
        hazy_dir: 有雾图像目录
        labels_dir: 标注目录
        output_dir: 输出目录
    """
    print("=" * 60)
    print("准备端到端雾天训练数据")
    print("=" * 60)

    hazy_path = Path(hazy_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    train_img_dir = output_path / 'images' / 'train'
    train_label_dir = output_path / 'labels' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    val_label_dir = output_path / 'labels' / 'val'

    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取所有有雾图像
    hazy_images = list(hazy_path.glob('*.jpg')) + list(hazy_path.glob('*.png'))
    print(f"\n找到 {len(hazy_images)} 张有雾图像")

    # 划分训练集和验证集（80/20）
    split_idx = int(len(hazy_images) * 0.8)
    train_images = hazy_images[:split_idx]
    val_images = hazy_images[split_idx:]

    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")

    # 复制训练集
    print("\n复制训练集...")
    for img_file in train_images:
        # 复制图像
        shutil.copy(img_file, train_img_dir / img_file.name)

        # 复制标注
        label_file = labels_path / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(label_file, train_label_dir / label_file.name)

    # 复制验证集
    print("复制验证集...")
    for img_file in val_images:
        # 复制图像
        shutil.copy(img_file, val_img_dir / img_file.name)

        # 复制标注
        label_file = labels_path / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(label_file, val_label_dir / label_file.name)

    print("\n✓ 数据准备完成！")
    print(f"  输出目录: {output_path}")
    print(f"  训练图像: {train_img_dir}")
    print(f"  训练标注: {train_label_dir}")
    print(f"  验证图像: {val_img_dir}")
    print(f"  验证标注: {val_label_dir}")

    # 创建数据配置文件
    data_config = {
        'path': str(output_path.absolute()),
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

    config_path = output_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f)

    print(f"\n✓ 创建数据配置: {config_path}")

    return str(output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='准备端到端训练数据')
    parser.add_argument('--hazy-dir', type=str,
                        default='datasets/school/hazy_school',
                        help='有雾图像目录')
    parser.add_argument('--labels-dir', type=str,
                        default='datasets/school/labels',
                        help='标注目录')
    parser.add_argument('--output-dir', type=str,
                        default='datasets/school_foggy',
                        help='输出目录')

    args = parser.parse_args()

    prepare_foggy_training_data(
        hazy_dir=args.hazy_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir
    )
