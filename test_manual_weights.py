"""
测试手动调整融合权重的效果

不需要重新训练，直接修改权重比例
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo


def test_different_weights(checkpoint_path: str, data_dir: str, model_size: str = 's'):
    """测试不同的融合权重"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载融合模型
    print("加载融合模型...")
    model = create_feature_fusion_yolo(
        model_size=model_size,
        num_classes=6,
        fusion_type='learned',
        pretrained=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载YOLO用于检测
    yolo_model = YOLO(f'yolo11{model_size}.pt')

    # 准备测试数据
    data_path = Path(data_dir)
    val_orig_dir = data_path / 'val' / 'images_original'
    val_dehz_dir = data_path / 'val' / 'images_dehazed'
    img_files = list(val_orig_dir.glob('*.jpg'))[:10]

    # 测试不同的权重比例
    weight_configs = [
        (0.3, 0.7, "30%有雾 + 70%清晰"),
        (0.4, 0.6, "40%有雾 + 60%清晰"),
        (0.5, 0.5, "50%有雾 + 50%清晰（当前）"),
        (0.6, 0.4, "60%有雾 + 40%清晰"),
        (0.7, 0.3, "70%有雾 + 30%清晰"),
    ]

    print("\n" + "="*60)
    print("测试不同的融合权重")
    print("="*60)

    results = {}

    for w_orig, w_dehz, desc in weight_configs:
        print(f"\n测试: {desc}")

        # 手动设置权重
        with torch.no_grad():
            model.fusion_module.weight_original.data = torch.tensor(w_orig)
            model.fusion_module.weight_dehazed.data = torch.tensor(w_dehz)

        total_detections = 0
        confidences = []

        for img_file in tqdm(img_files, desc="检测中"):
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

            # 融合
            with torch.no_grad():
                fused_img = model.fusion_module(img_orig_tensor, img_dehz_tensor)
                fused_img_np = fused_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
                fused_img_np = (fused_img_np * 255).clip(0, 255).astype(np.uint8)

            # 检测
            yolo_results = yolo_model(fused_img_np, conf=0.25, verbose=False)

            if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                total_detections += len(yolo_results[0].boxes)
                confidences.extend(yolo_results[0].boxes.conf.cpu().numpy().tolist())

        avg_conf = np.mean(confidences) if confidences else 0

        results[desc] = {
            'total': total_detections,
            'avg_conf': avg_conf,
            'per_image': total_detections / len(img_files)
        }

        print(f"  检测数: {total_detections}")
        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  平均每张图: {total_detections / len(img_files):.1f}")

    # 打印总结
    print("\n" + "="*60)
    print("结果总结")
    print("="*60)

    for desc, data in results.items():
        print(f"\n{desc}:")
        print(f"  总检测数: {data['total']}")
        print(f"  平均置信度: {data['avg_conf']:.3f}")
        print(f"  平均每张图: {data['per_image']:.1f}")

    # 找出最佳权重
    best_config = max(results.items(), key=lambda x: x[1]['total'])
    print("\n" + "="*60)
    print(f"✓ 最佳配置: {best_config[0]}")
    print(f"  检测数: {best_config[1]['total']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='测试不同的融合权重')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型检查点')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--model-size', type=str, default='s', help='模型大小')

    args = parser.parse_args()

    test_different_weights(args.checkpoint, args.data_dir, args.model_size)


if __name__ == '__main__':
    main()
