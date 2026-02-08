"""
验证Feature-level Fusion训练效果

检查项：
1. 训练曲线（损失下降）
2. 融合权重（学习到的参数）
3. 融合图像可视化
4. 检测性能对比
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo
from ultralytics import YOLO
import argparse


def load_checkpoint(checkpoint_path: str, model):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def check_training_curve(checkpoint_path: str, output_dir: Path):
    """检查训练曲线"""
    print("\n" + "="*60)
    print("1. 训练曲线分析")
    print("="*60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history = checkpoint['history']

    train_loss = history['train_loss']
    learning_rate = history['learning_rate']

    print(f"训练epochs: {len(train_loss)}")
    print(f"初始损失: {train_loss[0]:.4f}")
    print(f"最终损失: {train_loss[-1]:.4f}")
    print(f"损失下降: {(train_loss[0] - train_loss[-1]) / train_loss[0] * 100:.2f}%")

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_loss, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # 学习率曲线
    ax2.plot(learning_rate, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path}")

    # 判断训练是否有效
    if train_loss[-1] < train_loss[0] * 0.5:
        print("✓ 训练有效：损失显著下降")
    elif train_loss[-1] < train_loss[0]:
        print("⚠ 训练部分有效：损失有所下降，但可能需要更多epochs")
    else:
        print("✗ 训练可能无效：损失未下降")


def check_fusion_weights(model):
    """检查融合权重"""
    print("\n" + "="*60)
    print("2. 融合权重分析")
    print("="*60)

    fusion_module = model.fusion_module

    if hasattr(fusion_module, 'weight_original'):
        w_orig = torch.sigmoid(fusion_module.weight_original).item()
        w_dehz = torch.sigmoid(fusion_module.weight_dehazed).item()
        total = w_orig + w_dehz

        print(f"原图权重: {w_orig:.4f} ({w_orig/total*100:.1f}%)")
        print(f"去雾图权重: {w_dehz:.4f} ({w_dehz/total*100:.1f}%)")

        # 判断权重是否合理
        if 0.3 < w_orig/total < 0.7:
            print("✓ 权重平衡：模型学会了平衡两个输入")
        elif w_orig/total > 0.8:
            print("⚠ 偏向原图：模型更依赖原图")
        elif w_orig/total < 0.2:
            print("⚠ 偏向去雾图：模型更依赖去雾图")

    elif fusion_module.fusion_type == 'attention':
        print("使用注意力机制融合")
        print(f"注意力模块参数数量: {sum(p.numel() for p in fusion_module.attention.parameters())}")
        print("✓ 注意力权重会根据输入动态调整")

    else:
        print("使用简单平均融合")


def visualize_fusion(model, data_dir: str, output_dir: Path, device='cuda'):
    """可视化融合结果"""
    print("\n" + "="*60)
    print("3. 融合图像可视化")
    print("="*60)

    model.eval()
    model = model.to(device)

    # 读取一张测试图像
    data_path = Path(data_dir)
    val_orig_dir = data_path / 'val' / 'images_original'
    val_dehz_dir = data_path / 'val' / 'images_dehazed'

    img_files = list(val_orig_dir.glob('*.jpg'))[:3]  # 取3张图

    for idx, img_file in enumerate(img_files):
        # 读取原图和去雾图
        img_orig = cv2.imread(str(img_file))
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        img_dehz_path = val_dehz_dir / img_file.name
        img_dehz = cv2.imread(str(img_dehz_path))
        img_dehz = cv2.cvtColor(img_dehz, cv2.COLOR_BGR2RGB)

        # 预处理
        h, w = img_orig.shape[:2]
        img_orig_tensor = cv2.resize(img_orig, (640, 640))
        img_dehz_tensor = cv2.resize(img_dehz, (640, 640))

        img_orig_tensor = torch.from_numpy(img_orig_tensor).float().permute(2, 0, 1) / 255.0
        img_dehz_tensor = torch.from_numpy(img_dehz_tensor).float().permute(2, 0, 1) / 255.0

        img_orig_tensor = img_orig_tensor.unsqueeze(0).to(device)
        img_dehz_tensor = img_dehz_tensor.unsqueeze(0).to(device)

        # 获取融合结果
        with torch.no_grad():
            fused_tensor = model.fusion_module(img_orig_tensor, img_dehz_tensor)

        # 转换回图像
        fused_img = fused_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        fused_img = (fused_img * 255).clip(0, 255).astype(np.uint8)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.resize(img_orig, (640, 640)))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cv2.resize(img_dehz, (640, 640)))
        axes[1].set_title('Dehazed Image')
        axes[1].axis('off')

        axes[2].imshow(fused_img)
        axes[2].set_title('Fused Image')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f'fusion_visualization_{idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存可视化 {idx+1}: {save_path}")

    plt.close('all')


def test_detection_performance(model, data_dir: str, device='cuda'):
    """测试检测性能"""
    print("\n" + "="*60)
    print("4. 检测性能测试")
    print("="*60)

    model.eval()
    model = model.to(device)

    # 读取验证集
    data_path = Path(data_dir)
    val_orig_dir = data_path / 'val' / 'images_original'
    val_dehz_dir = data_path / 'val' / 'images_dehazed'

    img_files = list(val_orig_dir.glob('*.jpg'))

    print(f"测试图像数量: {len(img_files)}")

    total_detections = 0

    for img_file in img_files[:5]:  # 测试前5张
        # 读取图像
        img_orig = cv2.imread(str(img_file))
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        img_dehz_path = val_dehz_dir / img_file.name
        img_dehz = cv2.imread(str(img_dehz_path))
        img_dehz = cv2.cvtColor(img_dehz, cv2.COLOR_BGR2RGB)

        # 预处理
        img_orig_tensor = cv2.resize(img_orig, (640, 640))
        img_dehz_tensor = cv2.resize(img_dehz, (640, 640))

        img_orig_tensor = torch.from_numpy(img_orig_tensor).float().permute(2, 0, 1) / 255.0
        img_dehz_tensor = torch.from_numpy(img_dehz_tensor).float().permute(2, 0, 1) / 255.0

        img_orig_tensor = img_orig_tensor.unsqueeze(0).to(device)
        img_dehz_tensor = img_dehz_tensor.unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(img_orig_tensor, img_dehz_tensor)

        # 简单统计（实际需要NMS后处理）
        if isinstance(outputs, dict):
            # YOLO v11返回dict
            total_detections += 1  # 只统计成功推理的次数
        elif isinstance(outputs, (list, tuple)):
            total_detections += sum(out.numel() for out in outputs if hasattr(out, 'numel'))
        elif hasattr(outputs, 'numel'):
            total_detections += outputs.numel()

    print(f"✓ 模型可以正常前向传播")
    print(f"  输出元素总数: {total_detections}")
    print("\n注意：完整的检测性能评估需要：")
    print("  1. NMS后处理")
    print("  2. 与ground truth对比")
    print("  3. 计算mAP指标")


def main():
    parser = argparse.ArgumentParser(description='验证Feature-level Fusion训练效果')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--model-size', type=str, default='s', help='模型大小')
    parser.add_argument('--fusion-type', type=str, default='learned', help='融合类型')
    parser.add_argument('--output-dir', type=str, default='validation_results', help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Feature-level Fusion 训练效果验证")
    print("="*60)
    print(f"检查点: {args.checkpoint}")
    print(f"数据集: {args.data_dir}")

    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    model = create_feature_fusion_yolo(
        model_size=args.model_size,
        num_classes=6,
        fusion_type=args.fusion_type,
        pretrained=True
    )

    # 加载检查点
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"✓ 加载检查点成功 (epoch {checkpoint['epoch']})")

    # 1. 检查训练曲线
    check_training_curve(args.checkpoint, output_dir)

    # 2. 检查融合权重
    check_fusion_weights(model)

    # 3. 可视化融合结果
    visualize_fusion(model, args.data_dir, output_dir, device)

    # 4. 测试检测性能
    test_detection_performance(model, args.data_dir, device)

    print("\n" + "="*60)
    print("验证完成！")
    print("="*60)
    print(f"结果保存在: {output_dir}")
    print("\n建议：")
    print("1. 查看训练曲线，确认损失下降")
    print("2. 查看融合图像，确认融合效果合理")
    print("3. 如果效果好，可以进行阶段2训练（端到端微调）")


if __name__ == '__main__':
    main()
