#!/bin/bash
# AutoDL服务器快速部署脚本

echo "=========================================="
echo "Feature-level Fusion YOLO 部署脚本"
echo "=========================================="

# 1. 解压代码
echo "1. 解压代码..."
cd /root/autodl-tmp
tar -xzf yolosystem_code_only.tar.gz
echo "✓ 代码解压完成"

# 2. 解压数据
echo "2. 解压数据..."
tar -xzf project_data.tar.gz
echo "✓ 数据解压完成"

# 3. 安装依赖
echo "3. 安装依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "✓ 依赖安装完成"

# 4. 验证环境
echo "4. 验证环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
nvidia-smi
echo "✓ 环境验证完成"

# 5. 准备数据集
echo "5. 准备数据集..."
python prepare_fusion_dataset.py
echo "✓ 数据集准备完成"

# 6. 显示训练命令
echo ""
echo "=========================================="
echo "部署完成！现在可以开始训练："
echo "=========================================="
echo ""
echo "# 使用tmux后台运行（推荐）"
echo "tmux new -s training"
echo ""
echo "# 启动训练（A100优化配置）"
echo "python train_feature_fusion.py \\"
echo "    --data-dir datasets/fusion_training \\"
echo "    --model-size s \\"
echo "    --epochs 50 \\"
echo "    --batch-size 32 \\"
echo "    --lr 0.001 \\"
echo "    --output-dir runs/feature_fusion_a100"
echo ""
echo "# 分离tmux会话: Ctrl+B, 然后按 D"
echo "# 重新连接: tmux attach -t training"
echo ""
echo "预计训练时间: 10-12小时"
echo "预计成本: 33-40元"
echo ""
echo "祝训练顺利！🚀"
