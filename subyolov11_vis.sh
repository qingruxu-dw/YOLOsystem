#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=log_%j.txt
#SBATCH --partition=gpu       # 使用 gpu 分区
#SBATCH --gres=gpu:1          # 申请 1 块 GPU

# 加载必要的模块（根据集群配置调整）
module load cuda/11.8           # 加载 CUDA 11.8
module load miniforge3/24.11          # 加载 Conda

# 激活你的 Conda 环境（假设环境名为 `my_env`）
# source activate CoA
export PYTHONUNBUFFERED=1

# python3 train_visdrone_yolov11.py \
#   --data datasets/fusion_training/dataset.yaml \
#   --epochs 100 \
#   --batch 16 \
#   --project runs/yolov8_train_visdrone \
#   --model yolov8n.pt \
#   --name yolov8_n_visdrone

# python3 train_visdrone_yolov11.py \
#   --data datasets/fusion_training/dataset.yaml \
#   --epochs 100 \
#   --batch 16 \
#   --project runs/yolov10_train_visdrone \
#   --model yolov10n.pt \
#   --name yolov10_n_visdrone


# python3 train_visdrone_yolov11.py \
#   --data datasets/fusion_training/dataset.yaml \
#   --epochs 100 \
#   --batch 8 \
#   --project runs/yolov11_train_visdrone \
#   --model yolo11n.pt \
#   --name yolov11n_visdrone


python train_visdrone_yolov11.py \
  --data datasets/fusion_training/dataset.yaml \
  --model yolo11n.pt \
  --epochs 300 \
  --batch 16 \
  --project runs/train_visdrone \
  --name yolo11n_baseline