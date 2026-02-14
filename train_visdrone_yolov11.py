"""
Train YOLOv11 on VisDrone-style dataset using Ultralytics API.

Usage examples:
  python3 train_visdrone_yolov11.py --data datasets/fusion_training/dataset.yaml --model-size n --epochs 50 --batch 16 --imgsz 640

Notes:
- Ensure your dataset is in YOLO format (images + labels .txt in same relative structure) and paths in the YAML are correct.
- To compare fairly with your fusion model, set --model-size to the same size used by the fusion model (n/s/m/...)
"""
import argparse
from pathlib import Path
import os
import torch

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError('Please install ultralytics package in your environment: pip install ultralytics') from e


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv11 on VisDrone')
    p.add_argument('--data', type=str, required=True, help='Path to dataset yaml (train/val paths + nc + names)')
    p.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLO11 model size')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', type=str, default='0', help="cuda device or 'cpu'")
    p.add_argument('--project', type=str, default='runs/train_visdrone', help='save directory')
    p.add_argument('--name', type=str, default=None, help='experiment name')
    p.add_argument('--pretrained', type=str, default=None, help='path to pretrained weights to fine-tune')
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Select base weights
    if args.pretrained:
        weights = args.pretrained
    else:
        # official checkpoint name convention in this repo: yolo11{size}.pt
        weights = f'yolo11{args.model_size}.pt'

    # Create model object (will download if not available locally)
    print(f"Loading base weights: {weights}")
    model = YOLO(weights)

    # Train: ultralytics API accepts data (yaml), epochs, batch, imgsz, device, project/name
    project = args.project
    name = args.name if args.name else f'yolov11_{args.model_size}_visdrone'
    print(f"Start training: project={project} name={name}")

    # Use COCO->custom class mapping via data yaml which must contain 'nc' and 'names'
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=project,
        name=name
    )
    print("\n训练已完成,正在运行最终验证以获取详尽指标...")
    results = model.val()
    
    # 打印关键指标
    print(f"\n" + "="*30)
    print(f"最终对比指标 (Best Weights):")
    print(f"mAP50:    {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"可在 {os.path.join(project, name)} 查看详细曲线图 (results.png)")
    print("="*30)
    print('Training finished. Best weights will be in', os.path.join(project, name))


if __name__ == '__main__':
    main()
