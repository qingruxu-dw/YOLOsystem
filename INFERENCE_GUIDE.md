# 融合检测推理使用指南

## 📋 简介

本指南介绍如何使用融合检测模型进行推理测试，适合快速部署和测试。

## 🚀 快速开始

### 方法1：简单融合检测（推荐）⭐

使用预训练YOLO + 图像级融合，无需训练，开箱即用。

#### 单张图像检测

```bash
python simple_fusion_inference.py \
    --input test_image.jpg \
    --output outputs/test \
    --mode image \
    --conf 0.25 \
    --fusion-weight 0.7
```

**参数说明**：
- `--input`: 输入图像路径
- `--output`: 输出目录
- `--mode`: 处理模式（image/video/folder）
- `--conf`: 置信度阈值（0-1）
- `--fusion-weight`: 去雾图权重（0-1），推荐0.7

**输出文件**：
- `{name}_foggy.jpg` - 有雾图检测结果
- `{name}_dehazed.jpg` - 去雾图检测结果
- `{name}_fusion.jpg` - 融合图检测结果
- `{name}_comparison.jpg` - 三者对比图

#### 视频检测

```bash
python simple_fusion_inference.py \
    --input test_video.mp4 \
    --output outputs/test_video.mp4 \
    --mode video \
    --conf 0.25 \
    --fusion-weight 0.7
```

#### 批量处理

```bash
python simple_fusion_inference.py \
    --input test_images/ \
    --output outputs/batch \
    --mode folder \
    --conf 0.25 \
    --fusion-weight 0.7
```

---

### 方法2：训练好的融合模型

使用训练好的Feature Fusion模型（需要先训练）。

```bash
python inference_fusion.py \
    --checkpoint runs/fusion_fixed/final_stage1.pth \
    --input test_image.jpg \
    --output output.jpg \
    --model-size s \
    --conf 0.25
```

**参数说明**：
- `--checkpoint`: 训练好的模型权重路径
- `--input`: 输入图像路径
- `--output`: 输出路径
- `--model-size`: YOLO模型大小（n/s/m/l/x）
- `--conf`: 置信度阈值

---

## 📊 使用示例

### 示例1：测试单张雾天图像

```bash
# 使用简单融合（推荐）
python simple_fusion_inference.py \
    --input examples/foggy_street.jpg \
    --output outputs/demo \
    --mode image
```

**预期输出**：
```
加载YOLO模型: yolo11n.pt
✓ 模型加载完成

处理图像: examples/foggy_street.jpg
正在去雾...
融合图像（去雾权重: 70.0%）...
正在检测...

检测结果:
  有雾图: 15 个目标
  去雾图: 18 个目标
  融合图: 20 个目标

✓ 结果已保存到: outputs/demo
```

### 示例2：批量处理测试集

```bash
python simple_fusion_inference.py \
    --input datasets/test_images/ \
    --output outputs/test_results \
    --mode folder \
    --fusion-weight 0.7
```

**预期输出**：
```
找到 50 张图像

处理图像: image_001.jpg
...

=== 批量处理总结 ===
处理图像数: 50
平均检测数:
  有雾图: 12.3
  去雾图: 14.8
  融合图: 15.6
融合提升: +26.8%
```

### 示例3：处理视频

```bash
python simple_fusion_inference.py \
    --input test_video.mp4 \
    --output outputs/result.mp4 \
    --mode video \
    --conf 0.3
```

---

## ⚙️ 参数调优

### 置信度阈值（--conf）

- **0.15-0.20**: 检测更多目标，但可能有误检
- **0.25**: 默认值，平衡准确率和召回率
- **0.30-0.40**: 更严格，减少误检

### 融合权重（--fusion-weight）

- **0.5**: 有雾图和去雾图各占50%
- **0.7**: 去雾图占70%（推荐，效果最好）
- **1.0**: 只使用去雾图
- **0.0**: 只使用有雾图

**建议**：
- 浓雾场景：使用0.7-0.8
- 中雾场景：使用0.6-0.7
- 轻雾场景：使用0.3-0.5

---

## 🔧 环境要求

### 必需依赖

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
```

### 可选依赖

```bash
pip install tqdm  # 进度条
```

### 硬件要求

- **CPU**: 可运行，但速度较慢
- **GPU**: 推荐使用CUDA GPU（速度提升10-20倍）
- **内存**: 至少4GB RAM
- **显存**: 至少2GB VRAM（GPU模式）

---

## 📁 文件结构

```
YOLOsystem/
├── simple_fusion_inference.py      # 简单融合推理（推荐）
├── inference_fusion.py             # 训练模型推理
├── yolo11n.pt                      # YOLO预训练模型
├── yolosystem/
│   ├── dehazing.py                # 去雾模块
│   └── feature_fusion_yolo_simple.py  # 融合模型
└── outputs/                        # 输出目录
    └── inference/                  # 推理结果
```

---

## 🎯 性能对比

### 检测性能（Foggy Cityscapes）

| 方法 | 浓雾 | 中雾 | 轻雾 |
|------|------|------|------|
| 有雾图 | 301 | 352 | 397 |
| 去雾图 | 343 | 367 | 381 |
| **融合** | **340** | **376** | **393** |
| **提升** | **+13%** | **+6.8%** | **-1%** |

### 推理速度

| 设备 | 单张图像 | 视频(30fps) |
|------|---------|------------|
| CPU (i7) | ~2秒 | ~0.5x实时 |
| GPU (RTX 3060) | ~0.1秒 | ~5x实时 |
| GPU (A100) | ~0.05秒 | ~10x实时 |

---

## 🐛 常见问题

### Q1: 提示"无法读取图像"

**原因**：图像路径错误或格式不支持

**解决**：
- 检查路径是否正确
- 确保图像格式为jpg/png
- 使用绝对路径

### Q2: GPU内存不足

**原因**：图像分辨率过高或批次过大

**解决**：
```bash
# 降低图像分辨率
python simple_fusion_inference.py \
    --input large_image.jpg \
    --output output.jpg \
    --mode image
```

### Q3: 检测结果不理想

**原因**：参数设置不当

**解决**：
- 调整置信度阈值（--conf）
- 调整融合权重（--fusion-weight）
- 尝试不同的YOLO模型（yolo11s.pt, yolo11m.pt）

### Q4: 去雾效果不好

**原因**：图像不适合暗通道先验算法

**解决**：
- 降低融合权重（使用更多有雾图信息）
- 或直接使用有雾图检测（--fusion-weight 0.0）

---

## 📝 Python API使用

### 基础用法

```python
from simple_fusion_inference import SimpleFusionDetector

# 创建检测器
detector = SimpleFusionDetector(model_path='yolo11n.pt')

# 检测单张图像
stats = detector.detect_image(
    img_path='test.jpg',
    output_dir='outputs',
    conf_threshold=0.25,
    fusion_weight=0.7
)

print(f"检测到 {stats['fusion']} 个目标")
```

### 批量处理

```python
# 批量处理文件夹
detector.detect_folder(
    folder_path='test_images/',
    output_dir='outputs/batch',
    conf_threshold=0.25,
    fusion_weight=0.7
)
```

### 视频处理

```python
# 处理视频
detector.detect_video(
    video_path='test.mp4',
    output_path='output.mp4',
    conf_threshold=0.25,
    fusion_weight=0.7
)
```

---

## 🚀 部署建议

### 本地部署

1. 克隆仓库
2. 安装依赖
3. 下载YOLO模型（已包含）
4. 运行推理脚本

### 服务器部署

```bash
# 使用GPU
CUDA_VISIBLE_DEVICES=0 python simple_fusion_inference.py \
    --input test.jpg \
    --output output.jpg \
    --mode image
```

### Docker部署

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install ultralytics opencv-python

CMD ["python", "simple_fusion_inference.py"]
```

---

## 📞 技术支持

如有问题，请：
1. 查看本文档的常见问题部分
2. 提交GitHub Issue
3. 联系项目维护者

---

**最后更新**: 2025-02-08
