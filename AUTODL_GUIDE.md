# AutoDL 部署和训练指南

## 📋 目录
1. [AutoDL配置](#1-autodl配置)
2. [环境搭建](#2-环境搭建)
3. [数据准备](#3-数据准备)
4. [开始训练](#4-开始训练)
5. [监控训练](#5-监控训练)
6. [下载模型](#6-下载模型)
7. [常见问题](#7-常见问题)

---

## 1. AutoDL配置

### 1.1 注册和充值

1. 访问 https://www.autodl.com/
2. 注册账号（建议使用学生认证获得优惠）
3. 充值 **200-300元**（足够训练使用）

### 1.2 创建实例

**推荐配置（1天训练完成）：**

```
GPU: RTX 3090 (24GB) × 1
价格: ~2.0元/小时
镜像: PyTorch 2.0.0 + Python 3.10 + CUDA 11.8
系统盘: 50GB
数据盘: 100GB
预估成本: 48-60元（24小时）
```

**高性能配置（12小时完成）：**

```
GPU: A100 (40GB) × 1
价格: ~8.0元/小时
镜像: PyTorch 2.0.0 + Python 3.10 + CUDA 11.8
系统盘: 50GB
数据盘: 100GB
预估成本: 96-120元（12-15小时）
```

**创建步骤：**

1. 点击"租用实例"
2. 选择GPU类型
3. 选择镜像：`PyTorch 2.0.0 Python 3.10 CUDA 11.8`
4. 配置存储
5. 点击"立即租用"

---

## 2. 环境搭建

### 2.1 SSH连接

AutoDL会提供SSH连接信息：

```bash
# 示例（替换为你的实际信息）
ssh -p 12345 root@region-x.autodl.com
密码: xxxxxx
```

### 2.2 安装依赖

```bash
# 进入工作目录
cd /root/autodl-tmp

# 克隆代码（如果已上传到GitHub）
git clone <你的仓库URL>
cd YOLOsystem-copilot-add-dehazing-and-detection-system

# 或者使用AutoDL文件传输上传代码

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装额外依赖
pip install ultralytics==8.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 验证环境

```bash
# 检查GPU
nvidia-smi

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 检查YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## 3. 数据准备

### 3.1 上传数据

**方法1：使用SCP（推荐）**

在本地电脑运行：

```bash
# 上传整个project(labelimg)文件夹
scp -P 12345 -r "project(labelimg)" root@region-x.autodl.com:/root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system/

# 或者打包后上传（更快）
tar -czf project_labelimg.tar.gz "project(labelimg)"
scp -P 12345 project_labelimg.tar.gz root@region-x.autodl.com:/root/autodl-tmp/

# 在服务器上解压
ssh -p 12345 root@region-x.autodl.com
cd /root/autodl-tmp
tar -xzf project_labelimg.tar.gz
```

**方法2：使用AutoDL文件传输**

1. 在AutoDL控制台点击"JupyterLab"
2. 使用Web界面上传文件
3. 上传 `project(labelimg)` 文件夹

### 3.2 准备训练数据

```bash
cd /root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system

# 运行数据准备脚本
python prepare_fusion_dataset.py
```

输出：
```
数据集准备完成！
训练集: 49 对
验证集: 10 对
测试集: 11 对
配置文件: datasets/fusion_training/dataset.yaml
```

---

## 4. 开始训练

### 4.1 快速开始

```bash
# 使用默认配置训练
python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --model-size n \
    --epochs 50 \
    --batch-size 16 \
    --output-dir runs/feature_fusion_v1
```

### 4.2 完整参数

```bash
python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --num-classes 6 \
    --model-size n \
    --pretrained \
    --img-size 640 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --weight-decay 0.0005 \
    --workers 8 \
    --val-interval 5 \
    --output-dir runs/feature_fusion_v1
```

### 4.3 后台运行（推荐）

使用 `nohup` 后台运行，防止SSH断开导致训练中断：

```bash
nohup python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --model-size n \
    --epochs 50 \
    --batch-size 16 \
    --output-dir runs/feature_fusion_v1 \
    > training.log 2>&1 &

# 查看进程
ps aux | grep train_feature_fusion

# 查看日志
tail -f training.log
```

### 4.4 使用tmux（更推荐）

```bash
# 创建tmux会话
tmux new -s training

# 在tmux中运行训练
python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --model-size n \
    --epochs 50 \
    --batch-size 16 \
    --output-dir runs/feature_fusion_v1

# 分离会话（训练继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t training

# 查看所有会话
tmux ls
```

---

## 5. 监控训练

### 5.1 查看日志

```bash
# 实时查看日志
tail -f training.log

# 查看最后100行
tail -n 100 training.log

# 搜索关键信息
grep "Epoch" training.log
grep "loss" training.log
```

### 5.2 监控GPU使用

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 5.3 查看训练进度

```bash
# 查看输出目录
ls -lh runs/feature_fusion_v1/

# 查看最新的检查点
ls -lht runs/feature_fusion_v1/*.pt
```

---

## 6. 下载模型

### 6.1 训练完成后

```bash
# 查看模型文件
ls -lh runs/feature_fusion_v1/
# best.pt  - 最佳模型
# last.pt  - 最后一个epoch的模型
```

### 6.2 下载到本地

**方法1：使用SCP**

在本地电脑运行：

```bash
# 下载最佳模型
scp -P 12345 root@region-x.autodl.com:/root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system/runs/feature_fusion_v1/best.pt ./

# 下载整个输出目录
scp -P 12345 -r root@region-x.autodl.com:/root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system/runs/feature_fusion_v1 ./
```

**方法2：使用AutoDL文件传输**

1. 在AutoDL控制台点击"JupyterLab"
2. 导航到 `runs/feature_fusion_v1/`
3. 右键点击文件 → Download

---

## 7. 常见问题

### 7.1 显存不足

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**

```bash
# 减小batch size
python train_feature_fusion.py --batch-size 8  # 从16改为8

# 或使用更小的模型
python train_feature_fusion.py --model-size n  # 使用nano版本

# 或使用梯度累积
python train_feature_fusion.py --batch-size 4 --accumulate 4
```

### 7.2 SSH断开

**问题：** SSH连接断开导致训练中断

**解决方案：**

1. 使用 `tmux` 或 `screen`
2. 使用 `nohup` 后台运行
3. 在AutoDL控制台设置"自动关机时间"为较长时间

### 7.3 数据加载慢

**解决方案：**

```bash
# 减少workers数量
python train_feature_fusion.py --workers 4  # 从8改为4

# 或将数据复制到系统盘（SSD更快）
cp -r datasets/fusion_training /root/
python train_feature_fusion.py --data-dir /root/fusion_training
```

### 7.4 训练速度慢

**优化建议：**

1. **使用更强的GPU**：A100 > RTX 3090 > RTX 3080
2. **增大batch size**：在显存允许的情况下
3. **使用混合精度训练**：添加 `--amp` 参数
4. **减少验证频率**：`--val-interval 10`

### 7.5 如何暂停和恢复训练

**暂停训练：**

```bash
# 找到进程ID
ps aux | grep train_feature_fusion

# 杀死进程
kill <PID>
```

**恢复训练：**

```bash
# 从检查点恢复（需要在代码中实现）
python train_feature_fusion.py \
    --resume runs/feature_fusion_v1/last.pt \
    --epochs 100
```

---

## 8. 训练时间估算

### 8.1 基于GPU类型

| GPU型号 | 显存 | 价格/小时 | 预估训练时间 | 总成本 |
|---------|------|-----------|-------------|--------|
| RTX 3090 | 24GB | 2元 | 20-24小时 | 40-48元 |
| A100 | 40GB | 8元 | 10-12小时 | 80-96元 |
| 2×RTX 3090 | 48GB | 4元 | 12-15小时 | 48-60元 |

### 8.2 基于数据量

- **70对图像**：约15-20小时（RTX 3090）
- **500对图像**：约30-40小时（RTX 3090）

---

## 9. 训练完成后

### 9.1 评估模型

```bash
# 在测试集上评估
python evaluate_feature_fusion.py \
    --model runs/feature_fusion_v1/best.pt \
    --data-dir datasets/fusion_training \
    --split test
```

### 9.2 可视化结果

```bash
# 生成对比图
python visualize_fusion_results.py \
    --model runs/feature_fusion_v1/best.pt \
    --image-dir datasets/fusion_training/test/images_original \
    --output-dir outputs/fusion_results
```

### 9.3 关闭实例

**重要：训练完成后立即关闭实例，避免浪费费用！**

1. 下载所有需要的文件
2. 在AutoDL控制台点击"关机"
3. 确认数据已备份

---

## 10. 快速命令参考

```bash
# 一键启动训练（复制粘贴即可）
cd /root/autodl-tmp/YOLOsystem-copilot-add-dehazing-and-detection-system && \
tmux new -s training -d && \
tmux send-keys -t training "python train_feature_fusion.py --data-dir datasets/fusion_training --model-size n --epochs 50 --batch-size 16 --output-dir runs/feature_fusion_v1" C-m

# 查看训练状态
tmux attach -t training

# 监控GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f training.log
```

---

## 📞 需要帮助？

如果遇到问题：

1. 查看日志文件 `training.log`
2. 检查GPU使用情况 `nvidia-smi`
3. 查看AutoDL控制台的实例状态
4. 联系AutoDL客服（24小时在线）

祝训练顺利！🚀
