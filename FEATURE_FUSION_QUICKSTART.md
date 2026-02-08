# Feature-level Fusion YOLO 快速开始

## 📋 概述

本项目实现了基于YOLOv11的Feature-level Fusion网络，用于雾天目标检测。

**核心创新**：
- 双路输入：原图 + 去雾图
- 多层级特征融合：在backbone的P2、P3、P4层进行特征融合
- 自适应注意力机制：自动学习最优融合权重

---

## 🚀 快速开始（本地测试）

### 1. 准备数据

```bash
# 运行数据准备脚本
python prepare_fusion_dataset.py
```

输出：
```
数据集准备完成！
训练集: 49 对
验证集: 10 对
测试集: 11 对
```

### 2. 测试网络架构

```bash
# 测试Feature-level Fusion网络
cd yolosystem
python feature_fusion_yolo.py
```

预期输出：
```
创建Feature-level Fusion YOLO模型...
输入形状: torch.Size([2, 3, 640, 640])
模型创建成功！
总参数量: X.XX M
```

---

## ☁️ AutoDL云端训练（推荐）

### 1. 配置AutoDL

详细步骤见 [AUTODL_GUIDE.md](AUTODL_GUIDE.md)

**快速配置：**
- GPU: RTX 3090 (24GB)
- 镜像: PyTorch 2.0.0 + Python 3.10 + CUDA 11.8
- 预估成本: 40-60元（24小时）

### 2. 上传代码和数据

```bash
# 在本地打包
tar -czf project_data.tar.gz "project(labelimg)"

# 上传到AutoDL
scp -P <端口> project_data.tar.gz root@<服务器地址>:/root/autodl-tmp/

# SSH连接到AutoDL
ssh -p <端口> root@<服务器地址>

# 解压
cd /root/autodl-tmp
tar -xzf project_data.tar.gz
```

### 3. 安装依赖

```bash
cd /root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 准备数据

```bash
python prepare_fusion_dataset.py
```

### 5. 开始训练

```bash
# 使用tmux后台运行
tmux new -s training

# 启动训练
python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --model-size n \
    --epochs 50 \
    --batch-size 16 \
    --output-dir runs/feature_fusion_v1

# 分离tmux会话（Ctrl+B, D）
# 训练将在后台继续运行
```

### 6. 监控训练

```bash
# 重新连接tmux
tmux attach -t training

# 或查看GPU使用
watch -n 1 nvidia-smi

# 查看输出目录
ls -lh runs/feature_fusion_v1/
```

### 7. 下载模型

```bash
# 训练完成后，在本地运行
scp -P <端口> root@<服务器地址>:/root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system/runs/feature_fusion_v1/best.pt ./
```

---

## 📊 评估模型

### 在测试集上评估

```bash
python evaluate_feature_fusion.py \
    --model runs/feature_fusion_v1/best.pt \
    --data-dir datasets/fusion_training \
    --split test \
    --output evaluation_results.json
```

### 与基线方法对比

```bash
python evaluate_feature_fusion.py \
    --model runs/feature_fusion_v1/best.pt \
    --data-dir datasets/fusion_training \
    --split test \
    --compare \
    --output comparison_results.json
```

---

## 📁 项目结构

```
YOLOsystem-copilot-add-dehazing-and-detection-system/
├── yolosystem/
│   ├── feature_fusion_yolo.py      # Feature-level Fusion网络架构
│   ├── dehazing.py                 # 去雾模块
│   ├── detection.py                # 检测模块
│   └── fusion.py                   # 原有融合模块
│
├── project(labelimg)/              # 你的标注数据
│   ├── dataset/
│   │   └── joint_train/
│   │       ├── clear_school/       # 清晰图像
│   │       └── hazy_school/        # 雾天图像
│   └── dataset(label)/
│       ├── imgs/                   # 图像
│       └── labels/                 # YOLO格式标注
│
├── datasets/
│   └── fusion_training/            # 准备好的训练数据
│       ├── train/
│       │   ├── images_original/
│       │   ├── images_dehazed/
│       │   └── labels/
│       ├── val/
│       └── test/
│
├── runs/
│   └── feature_fusion_v1/          # 训练输出
│       ├── best.pt                 # 最佳模型
│       └── last.pt                 # 最后一个epoch
│
├── prepare_fusion_dataset.py       # 数据准备脚本
├── train_feature_fusion.py         # 训练脚本
├── evaluate_feature_fusion.py      # 评估脚本
├── AUTODL_GUIDE.md                 # AutoDL详细指南
└── requirements.txt                # 依赖列表
```

---

## 🔧 训练参数说明

### 基本参数

```bash
--data-dir          # 数据集目录
--model-size        # 模型大小: n, s, m, l, x
--epochs            # 训练轮数（推荐50-100）
--batch-size        # 批次大小（根据显存调整）
--lr                # 学习率（默认0.001）
--output-dir        # 输出目录
```

### 推荐配置

**RTX 3090 (24GB):**
```bash
--model-size n --batch-size 16 --epochs 50
```

**A100 (40GB):**
```bash
--model-size s --batch-size 32 --epochs 50
```

**显存不足时:**
```bash
--model-size n --batch-size 8 --epochs 50
```

---

## 📈 预期结果

### 训练时间

| GPU | Batch Size | 预估时间 |
|-----|-----------|---------|
| RTX 3090 | 16 | 20-24小时 |
| A100 | 32 | 10-12小时 |

### 性能提升

相比简单串联方法，Feature-level Fusion预期提升：
- **检测数量**: +3-5%
- **mAP**: +2-4%
- **小目标检测**: +5-8%

---

## ❓ 常见问题

### Q1: 显存不足怎么办？

**A:** 减小batch size或使用更小的模型：
```bash
python train_feature_fusion.py --batch-size 8 --model-size n
```

### Q2: 训练速度太慢？

**A:**
1. 使用更强的GPU（A100）
2. 增大batch size
3. 减少数据加载workers

### Q3: 如何恢复中断的训练？

**A:** 使用 `--resume` 参数（需要在代码中实现）：
```bash
python train_feature_fusion.py --resume runs/feature_fusion_v1/last.pt
```

### Q4: 数据量太少怎么办？

**A:**
1. 使用数据增强
2. 结合Foggy Cityscapes数据集
3. 使用预训练权重（迁移学习）

---

## 📚 相关文档

- [AutoDL详细指南](AUTODL_GUIDE.md) - 云端训练完整教程
- [README.md](README.md) - 项目总体说明
- [COMPREHENSIVE_COMPARISON_REPORT.md](outputs/COMPREHENSIVE_COMPARISON_REPORT.md) - 原有方法对比报告

---

## 🎯 下一步

1. ✅ 准备数据集
2. ✅ 在AutoDL上训练模型
3. ⏳ 评估模型性能
4. ⏳ 与原有方法对比
5. ⏳ 撰写论文

---

## 💡 技术支持

如有问题，请查看：
1. [AUTODL_GUIDE.md](AUTODL_GUIDE.md) - 详细的AutoDL使用指南
2. 训练日志 `training.log`
3. AutoDL客服（24小时在线）

祝训练顺利！🚀
