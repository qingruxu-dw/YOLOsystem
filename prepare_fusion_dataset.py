"""
准备Feature-level Fusion训练数据集

功能：
1. 整合你的校园数据 + Foggy Cityscapes数据
2. 创建配对的原图-去雾图数据集
3. 划分训练集/验证集/测试集
4. 生成YOLO格式的数据集配置
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import cv2
from yolosystem.coa_adapter import CoADehazer


class FusionDatasetPreparer:
    """Feature-level Fusion数据集准备器"""

    def __init__(self, output_dir: str = 'datasets/fusion_training', use_model_dehazing: bool = True):
        """
        Args:
            output_dir: 输出目录
            use_model_dehazing: 是否使用模型生成的去雾图作为训练数据（True: 使用模型，False: 使用GT清晰图）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_model_dehazing = use_model_dehazing
        
        if self.use_model_dehazing:
            print("初始化 CoA 去雾模型用于生成训练数据...")
            self.dehazer = CoADehazer()
            print("模型初始化完成")

        # 创建子目录
        self.dirs = {
            'train_original': self.output_dir / 'train' / 'images_original',
            'train_dehazed': self.output_dir / 'train' / 'images_dehazed',
            'train_labels': self.output_dir / 'train' / 'labels',

            'val_original': self.output_dir / 'val' / 'images_original',
            'val_dehazed': self.output_dir / 'val' / 'images_dehazed',
            'val_labels': self.output_dir / 'val' / 'labels',

            'test_original': self.output_dir / 'test' / 'images_original',
            'test_dehazed': self.output_dir / 'test' / 'images_dehazed',
            'test_labels': self.output_dir / 'test' / 'labels',
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_school_dataset(self, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        处理校园数据集

        Args:
            split_ratio: (训练集, 验证集, 测试集) 比例
        """
        print("\n" + "=" * 60)
        print("处理校园数据集...")
        print("=" * 60)

        # 数据路径
        clear_dir = Path('/data/home/sczd119/run/DehazyDet/dataset/VisDrone/VisDrone2019-DET-train/clear')
        hazy_dir = Path('/data/home/sczd119/run/DehazyDet/dataset/VisDrone/VisDrone2019-DET-train/hazy')
        label_dir = Path('/data/home/sczd119/run/DehazyDet/dataset/VisDrone/VisDrone2019-DET-train/labels')

        # 获取所有图像文件名
        image_files = sorted([f.stem for f in clear_dir.glob('*.jpg')])
        print(f"找到 {len(image_files)} 对图像")

        # 过滤出有标注的图像
        labeled_images = []
        for img_name in image_files:
            label_file = label_dir / f"{img_name}.txt"
            if label_file.exists():
                labeled_images.append(img_name)

        print(f"其中有标注的: {len(labeled_images)} 对")

        # 随机打乱
        random.shuffle(labeled_images)

        # 划分数据集
        n_total = len(labeled_images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        train_images = labeled_images[:n_train]
        val_images = labeled_images[n_train:n_train + n_val]
        test_images = labeled_images[n_train + n_val:]

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_images)} 对")
        print(f"  验证集: {len(val_images)} 对")
        print(f"  测试集: {len(test_images)} 对")

        # 复制文件
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for split_name, image_list in splits.items():
            print(f"\n复制 {split_name} 数据...")
            for img_name in tqdm(image_list):
                # 原图（有雾的图像作为输入）
                src_hazy = hazy_dir / f"{img_name}.jpg"
                dst_original = self.dirs[f'{split_name}_original'] / f"{img_name}.jpg"
                shutil.copy2(src_hazy, dst_original)

                # 去雾图处理
                dst_dehazed = self.dirs[f'{split_name}_dehazed'] / f"{img_name}.jpg"
                
                if self.use_model_dehazing:
                    # 使用模型生成去雾图（推荐：使训练和推理的数据分布一致）
                    img_hazy = cv2.imread(str(src_hazy))
                    if img_hazy is None:
                        print(f"Warning: 无法读取图像 {src_hazy}")
                        continue
                    # 使用 CoA 模型去雾
                    img_dehazed = self.dehazer.process_opencv(img_hazy)
                    cv2.imwrite(str(dst_dehazed), img_dehazed)
                else:
                    # 使用GT清晰图（仅作参考对比）
                    src_clear = clear_dir / f"{img_name}.jpg"
                    if src_clear.exists():
                        shutil.copy2(src_clear, dst_dehazed)
                    else:
                        print(f"Warning: 找不到对应的清晰图像 {src_clear}")

                # 标注
                src_label = label_dir / f"{img_name}.txt"
                dst_label = self.dirs[f'{split_name}_labels'] / f"{img_name}.txt"
                shutil.copy2(src_label, dst_label)

        return len(train_images), len(val_images), len(test_images)

    def process_foggy_cityscapes(self, dataset_path: str = 'datasets/foggy_cityscapes',
                                 max_samples: int = 500):
        """
        处理Foggy Cityscapes数据集（如果有的话）

        Args:
            dataset_path: Foggy Cityscapes数据集路径
            max_samples: 最多使用多少样本
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            print("\n⚠️  未找到Foggy Cityscapes数据集，跳过")
            print(f"   如需使用，请将数据集放在: {dataset_path}")
            return 0, 0, 0

        print("\n" + "=" * 60)
        print("处理Foggy Cityscapes数据集...")
        print("=" * 60)

        # TODO: 实现Foggy Cityscapes数据处理
        # 这里需要根据实际的Foggy Cityscapes数据集结构来实现

        print("⚠️  Foggy Cityscapes处理功能待实现")
        return 0, 0, 0

    # def create_dataset_yaml(self, num_classes: int = 6):
    #使用VisDrone数据集进行训练尝试
    def create_dataset_yaml(self, num_classes: int = 10):
        """
        创建YOLO格式的数据集配置文件

        Args:
            num_classes: 类别数量
        """
        print("\n" + "=" * 60)
        print("创建数据集配置文件...")
        print("=" * 60)

        # 类别名称
        # class_names = ['car', 'motorcycle', 'person', 'truck', 'bus', 'bicycle']
        class_names = [
            'pedestrian',       # 0
            'people',           # 1
            'bicycle',          # 2
            'car',              # 3
            'van',              # 4
            'truck',            # 5
            'tricycle',         # 6
            'awning-tricycle',  # 7
            'bus',              # 8
            'motor'             # 9
        ]
        # 数据集配置
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',

            # 类别信息
            'nc': num_classes,
            'names': class_names,

            # Feature-level Fusion特定配置
            'dual_input': True,
            'original_images': 'images_original',
            'dehazed_images': 'images_dehazed',
        }

        # 保存配置
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, sort_keys=False)

        print(f"✅ 配置文件已保存: {config_path}")
        return config_path

    def generate_statistics(self):
        """生成数据集统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计")
        print("=" * 60)

        stats = {}
        for split in ['train', 'val', 'test']:
            n_images = len(list(self.dirs[f'{split}_original'].glob('*.jpg')))
            n_labels = len(list(self.dirs[f'{split}_labels'].glob('*.txt')))
            stats[split] = {'images': n_images, 'labels': n_labels}

            print(f"\n{split.upper()}:")
            print(f"  图像对数: {n_images}")
            print(f"  标注文件: {n_labels}")

        # 统计类别分布
        print("\n类别分布统计:")
        # class_counts = {i: 0 for i in range(6)}
        class_counts = {i: 0 for i in range(10)}  # 1. 改为 10

        for split in ['train', 'val', 'test']:
            label_dir = self.dirs[f'{split}_labels']
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1

        # class_names = ['car', 'motorcycle', 'person', 'truck', 'bus', 'bicycle']
        class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        for class_id, count in class_counts.items():
            print(f"  {class_names[class_id]}: {count}")

        return stats


def main():
    """主函数"""
    print("=" * 60)
    print("Feature-level Fusion 数据集准备工具")
    print("=" * 60)

    # 设置随机种子
    random.seed(42)

    # 创建数据集准备器
    preparer = FusionDatasetPreparer(output_dir='datasets/fusion_training')

    # 处理校园数据集
    n_train, n_val, n_test = preparer.process_school_dataset(
        split_ratio=(0.7, 0.15, 0.15)
    )

    # 处理Foggy Cityscapes（如果有）
    n_train_fc, n_val_fc, n_test_fc = preparer.process_foggy_cityscapes(
        max_samples=500
    )

    # 创建数据集配置
    config_path = preparer.create_dataset_yaml(num_classes=6)

    # 生成统计信息
    stats = preparer.generate_statistics()

    # 总结
    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print("=" * 60)
    print(f"\n总计:")
    print(f"  训练集: {n_train + n_train_fc} 对")
    print(f"  验证集: {n_val + n_val_fc} 对")
    print(f"  测试集: {n_test + n_test_fc} 对")
    print(f"\n配置文件: {config_path}")
    print(f"\n下一步: 使用 train_feature_fusion_v2.py 开始训练")


if __name__ == '__main__':
    main()
