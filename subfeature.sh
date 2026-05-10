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

# 运行你的 PyTorch 脚本
# python3 train_feature_fusion_v2.py \
#     --data-dir datasets/fusion_training \
#     --model-size n \
#     --num-classes 10 \
#     --stage1-epochs 0 \
#     --do-stage2 \
#     --resume-checkpoint runs/two_stage_training/stage1_best.pth \
#     --pretrained \
#     --batch-size 8
# python train_feature_fusion_v2.py \
#   --data-dir datasets/fusion_training \
#   --model-size n \
#   --fusion-type attention \
#   --pretrained \
#   --stage1-epochs 10 \
#   --stage1-lr 0.001 \
#   --do-stage2 \
#   --stage2-epochs 220 \
#   --stage2-lr 0.0001 \
#   --batch-size 16 \
#   --workers 8 \
#   --output-dir runs/visdrone_fusion_v2 \
#   --resume-checkpoint runs/visdrone_fusion_v2/best.pth


python train_feature_fusion_v2.py \
  --data-dir datasets/FusionData_Train \
  --model-size n \
  --fusion-type attention \
  --pretrained \
  --stage1-epochs 10 \
  --stage1-lr 0.001 \
  --do-stage2 \
  --stage2-epochs 300 \
  --stage2-lr 0.0001 \
  --stage2-warmup-epochs 5 \
  --optimizer adamw \
  --batch-size 16 \
  --workers 4 \
  --output-dir runs/visdrone_fusion
