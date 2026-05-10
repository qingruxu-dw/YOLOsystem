# 🛸 VisDrone 数据集训练与对比指南

## 📋 目标

在 VisDrone 无人机视角数据集上验证模型的检测性能。通过以下三组实验进行对比：
1. **对照组 (Baseline)**: 标准 YOLOv11n (针对 VisDrone 重新训练)
2. **去雾处理组**: 在去雾后的图像上直接运行 Baseline
3. **融合实验组 (Proposed)**: 使用 Feature-level Fusion YOLO 进行检测

---

## 📥 数据集准备

### 1. 数据格式要求
确保你的 VisDrone 数据集已转换为 **YOLO 格式**（`.txt` 标签）。

### 2. 标准文件结构
为了避免 Ultralytics 出现路径匹配错误，**强烈建议**使用以下结构：
```text
datasets/visdrone_data/
├── train/
│   ├── images/  # 存放图片
│   └── labels/  # 存放对应 .txt 标签
└── val/
    ├── images/
    └── labels/
```

### 3. 创建数据描述文件
根据 `datasets/visdrone.yaml.example` 修改你的路径：
```yaml
path: /data/run01/sczd119/YOLOsystem/datasets/visdrone_data  # 数据集根目录
train: train/images
val: val/images

nc: 10
names: ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
```

---

## 🚀 核心过程 1：训练对照组 (Baseline)

在进行融合对比前，必须先训练一个在 VisDrone 数据分布下的标准 YOLO 模型，否则官方的 COCO 权重无法在 VisDrone 类别上正确检测。

### 使用文件: `train_visdrone_yolov11.py`

**运行训练命令：**
```bash
python3 train_visdrone_yolov11.py \
    --data datasets/visdrone.yaml \
    --model-size n \
    --epochs 50 \
    --batch 16 \
    --device 0 \
    --project runs/train_visdrone \
    --name yolov11_baseline
```

**关键参数说明：**
- `--model-size`: 务必与你的融合模型大小保持一致（如均为 `n`）。
- `--pretrained`: 自动加载官方预训练权重进行微调。

---

## 🚀 核心过程 2：训练融合模型

使用你的特征融合框架进行两阶段训练。

### 使用文件: `train_feature_fusion_v2.py` (或其他对应的训练脚本)

**训练命令：**
```bash
python3 train_feature_fusion_v2.py \
    --data datasets/fusion_data.yaml \
    --epochs 50 \
    --batch 8 \
    --model-size n
```

---

## 📊 核心过程 3：测试训练效果与对比

训练完成后，使用对比脚本衡量融合模型相对于 Baseline 的提升。

### 使用文件: `test_fusion_detection.py`

**对比测试命令：**
```bash
python3 test_fusion_detection.py \
    --checkpoint YOLOsystem/stage2_best.pth \
    --yolo-weights yolov11_visdrone.pt \
    --data-dir datasets/fusion_training \
    --model-size n \
    --conf-threshold 0.25
```

---

## 💡 关键说明与技巧

### 1. 为什么 Loss 会是 0？
*   **现象**: 训练进度条显示 `box_loss = 0`。
*   **原因**: 标签定位失败。
*   **解决**: 
    1. 确保标签文件夹名为 `labels`（且位于 `images` 同级）。
    2. 删除所有旧缓存：`find . -name "*.cache" -delete`。

### 2. 如何判定训练好坏？
*   **指标 (Metrics)**: 在训练文件夹的 `results.png` 中查看 `mAP50` 曲线。
*   **收敛**: `box_loss` 应随 Epoch 增加而稳步下降，`mAP` 应上升至 0.2~0.4 之间（VisDrone 较难）。
*   **可视化**: 检查 `runs/.../val_batch0_labels.jpg` 确认标注框是否正确叠加在图片上。

### 3. 如何进行数据集替换？
只需修改 `.yaml` 文件中的路径。如果你的数据集图片叫 `images_original` 和 `images_dehazed`：
1. **必须**建立软链接到 `images` 和 `labels`。
2. 运行我们提供的软链接创建命令：
   ```bash
   ln -sf [你的图片目录]/* images/
   ln -sf [你的标签目录]/* labels/
   ```

---

## 🛠 相关文件清单

| 文件名 | 用途 |
| :--- | :--- |
| `train_visdrone_yolov11.py` | 训练标准 YOLOv11 对照组 |
| `datasets/visdrone.yaml.example` | 数据集配置文件模板 |
| `test_fusion_detection.py` | 最终对比测试脚本(原图 vs 去雾图 vs 融合图) |
| `inference_fusion.py` | 单张图/视频的融合检测推理(带可视化，**支持 Text-Guided Dehazing**) |
| `simple_fusion_inference.py` | 简单推理脚本 (**支持 Text-Guided Dehazing**) |

---
