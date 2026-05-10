#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=gpu       # 使用 gpu 分区
#SBATCH --gres=gpu:1          # 申请 1 块 GPU
#SBATCH --mem=55G             # 单卡关联内存（需 ≤55G，具体值需确认）
#SBATCH --cpus-per-task=4     # 每 GPU 卡分配的 CPU 核心
#SBATCH --output=log_%j.txt

# 加载必要的模块（根据集群配置调整）
module load cuda/11.8           # 加载 CUDA 11.8
module load miniconda3          # 加载 Conda

# 激活你的 Conda 环境（假设环境名为 `my_env`）
source activate CoA

# 检查 GPU 是否可用（调试用）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 运行你的 PyTorch 脚本
python KD.py --batch_size 32 --use_gpu
