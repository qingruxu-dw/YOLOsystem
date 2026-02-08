# 🚀 服务器部署快速指南

## 📦 准备好的文件

在本地目录下有两个打包文件：

1. **yolosystem_code_only.tar.gz** (68KB) - 代码文件
2. **project_data.tar.gz** (202MB) - 数据集

---

## 📤 步骤1：上传文件到服务器

### 获取服务器SSH信息

在你的云服务器控制台找到：
- **服务器地址**: 例如 `region-1.autodl.com`
- **SSH端口**: 例如 `12345`
- **密码**: 你的服务器密码

### 上传命令

在本地PowerShell中运行（替换 `<端口>` 和 `<服务器地址>`）：

```powershell
# 进入项目目录
cd "e:\2025\大创\YOLOsystem-copilot-add-dehazing-and-detection-system"

# 上传代码（很快）
scp -P <端口> yolosystem_code_only.tar.gz root@<服务器地址>:/root/autodl-tmp/

# 上传数据（需要几分钟）
scp -P <端口> project_data.tar.gz root@<服务器地址>:/root/autodl-tmp/
```

**示例**（假设端口是12345，地址是region-1.autodl.com）：
```powershell
scp -P 12345 yolosystem_code_only.tar.gz root@region-1.autodl.com:/root/autodl-tmp/
scp -P 12345 project_data.tar.gz root@region-1.autodl.com:/root/autodl-tmp/
```

---

## 🔧 步骤2：连接服务器并部署

### 连接SSH

```bash
ssh -p <端口> root@<服务器地址>
# 输入密码
```

### 运行部署脚本

```bash
cd /root/autodl-tmp

# 解压代码
tar -xzf yolosystem_code_only.tar.gz

# 运行一键部署脚本
bash deploy_autodl.sh
```

部署脚本会自动完成：
- ✅ 解压数据
- ✅ 安装依赖
- ✅ 验证环境
- ✅ 准备数据集

---

## 🎯 步骤3：开始训练

### 使用tmux后台运行（推荐）

```bash
# 创建tmux会话
tmux new -s training

# 启动训练（A100优化配置）
python train_feature_fusion.py \
    --data-dir datasets/fusion_training \
    --model-size s \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --output-dir runs/feature_fusion_a100

# 分离tmux会话（训练继续在后台运行）
# 按键：Ctrl+B，然后按 D
```

### 重新连接查看进度

```bash
# 重新连接tmux
tmux attach -t training

# 或者查看GPU使用
watch -n 1 nvidia-smi
```

---

## 📊 步骤4：监控训练

### 查看训练日志

```bash
# 如果使用了nohup
tail -f training.log

# 查看输出目录
ls -lh runs/feature_fusion_a100/
```

### 查看GPU使用

```bash
# 实时监控
nvidia-smi -l 1

# 或使用watch
watch -n 1 nvidia-smi
```

---

## 💾 步骤5：下载训练好的模型

### 训练完成后

在本地PowerShell运行：

```powershell
# 下载最佳模型
scp -P <端口> root@<服务器地址>:/root/autodl-tmp/runs/feature_fusion_a100/best.pt ./

# 下载整个输出目录
scp -P <端口> -r root@<服务器地址>:/root/autodl-tmp/runs/feature_fusion_a100 ./
```

---

## ⏱️ 预期时间和成本

| 项目 | 时间/成本 |
|------|----------|
| 上传文件 | 5-10分钟 |
| 部署环境 | 5分钟 |
| 训练时间 | **10-12小时** |
| 总成本 | **33-40元** (¥3.28/时 × 10-12小时) |

---

## ⚠️ 重要提示

1. **及时关机**：训练完成后立即关闭实例，避免浪费费用
2. **定期备份**：每隔2-3小时下载一次检查点
3. **使用tmux**：避免SSH断开导致训练中断
4. **监控显存**：确保batch size不会导致OOM

---

## 🆘 常见问题

### Q1: 上传速度慢？
**A**: 数据文件202MB，根据网速可能需要5-10分钟，耐心等待

### Q2: SSH连接断开怎么办？
**A**: 使用tmux后台运行，训练不会中断。重新连接后运行 `tmux attach -t training`

### Q3: 显存不足？
**A**: 减小batch size：`--batch-size 16` 或 `--batch-size 8`

### Q4: 如何暂停训练？
**A**: 在tmux中按 `Ctrl+C`，模型会自动保存到 `last.pt`

---

## 📞 需要帮助？

查看详细文档：
- [AUTODL_GUIDE.md](AUTODL_GUIDE.md) - 完整的AutoDL使用教程
- [FEATURE_FUSION_QUICKSTART.md](FEATURE_FUSION_QUICKSTART.md) - 快速开始指南

祝训练顺利！🚀
