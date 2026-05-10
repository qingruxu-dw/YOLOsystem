#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=log_%j.txt
#SBATCH --partition=gpu       # 使用 gpu 分区
#SBATCH --gres=gpu:1          # 申请 1 块 GPU

# 加载必要的模块（根据集群配置调整）
module load cuda/11.8           # 加载 CUDA 11.8
module load miniforge3/24.11          # 加载 Conda

# 激活你的 Conda 环境（假设环境名为 `my_env`）
source activate CoA
export PYTHONUNBUFFERED=1

# 运行你的 PyTorch 脚本
python KD.py 
