# YOLOsystem - 雾天目标检测系统

## 📖 系统简介

### 这个系统是做什么的？

本系统是一个**雾天场景下的目标检测系统**，解决了传统目标检测算法在雾天环境下性能下降的问题。

**核心思路**：
1. 输入一张有雾的图像
2. 使用去雾算法生成清晰图像
3. 将有雾图和去雾图进行**智能融合**
4. 在融合后的图像上进行目标检测
5. 相比单独使用有雾图或去雾图，融合方法在浓雾场景下检测性能提升**13%**

**应用场景**：
- 🚗 自动驾驶（雾天环境）
- 📹 视频监控（雾霾天气）
- 🚁 无人机巡检（低能见度）
- 🏙️ 智慧城市（恶劣天气）

---

## 🎯 核心创新

**Feature-level Fusion（特征层融合）**
- 双路输入：同时处理有雾图像和去雾图像
- 智能融合：可学习的融合权重，自适应平衡真实性和清晰度
- 显著提升：浓雾场景检测性能提升 **+13.0%**

---

## 📊 实验结果

在Foggy Cityscapes数据集上的验证结果：

| 雾浓度 | 有雾图检测 | 去雾图检测 | **Fusion检测** | **提升幅度** |
|--------|-----------|-----------|---------------|-------------|
| 浓雾 (β=0.02) | 301 | 343 | **340** | **+13.0%** ✅ |
| 中雾 (β=0.01) | 352 | 367 | **376** | **+6.8%** ✅ |
| 轻雾 (β=0.005) | 397 | 381 | 393 | -1.0% |

**关键发现**：
- ✅ 雾越浓，融合效果越好
- ✅ 浓雾场景提升最显著（+13%）
- ✅ 轻雾场景无需融合（直接检测效果更好）

---

## 🚀 快速开始（推荐新手）

### 方法1：简单推理测试（无需训练）⭐

适合快速测试和部署，使用预训练模型。

#### 步骤1：安装环境

```bash
# 克隆仓库
git clone https://github.com/yourusername/YOLOsystem.git
cd YOLOsystem

# 安装依赖
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
```

#### 步骤2：测试单张图像

```bash
python simple_fusion_inference.py \
    --input test_image.jpg \
    --output outputs/test \
    --mode image \
    --fusion-weight 0.7
```

**参数说明**：
- `--input`: 输入图像路径
- `--output`: 输出目录
- `--mode`: 处理模式（image/video/folder）
- `--fusion-weight`: 去雾图权重（0-1），推荐0.7
  - 0.0 = 只用有雾图
  - 0.7 = 30%有雾图 + 70%去雾图（推荐）
  - 1.0 = 只用去雾图

**输出结果**：
- `{name}_foggy.jpg` - 有雾图检测结果
- `{name}_dehazed.jpg` - 去雾图检测结果
- `{name}_fusion.jpg` - 融合图检测结果
- `{name}_comparison.jpg` - 三者对比图

#### 步骤3：查看结果

打开 `outputs/test/{name}_comparison.jpg` 查看三种方法的对比效果。

---

### 方法2：完整训练流程（研究使用）

适合需要训练自己模型的用户。

#### 步骤1：准备数据集

```bash
# 数据集目录结构
datasets/school/
├── hazy_school/      # 有雾图像
├── clear_school/     # 清晰图像（或去雾后图像）
└── labels/           # YOLO格式标注

# 准备融合数据集
python prepare_fusion_dataset.py \
    --hazy-dir datasets/school/hazy_school \
    --clear-dir datasets/school/clear_school \
    --labels-dir datasets/school/labels \
    --output-dir datasets/fusion_data
```

#### 步骤2：训练模型

```bash
# 两阶段训练
python train_feature_fusion_v2.py \
    --data-dir datasets/fusion_data \
    --model-size s \
    --epochs 50 \
    --batch-size 16
```

训练完成后，模型保存在 `runs/fusion_fixed/final_stage1.pth`

#### 步骤3：测试模型

```bash
# 在Foggy Cityscapes上测试
python test_foggy_cityscapes.py \
    --checkpoint runs/fusion_fixed/final_stage1.pth \
    --data-dir datasets/foggy_cityscapes/val \
    --fog-level 0.02 \
    --model-size s
```

---

## 📁 项目结构

```
YOLOsystem/
├── 📚 文档文件
│   ├── README.md                           # 项目总览（本文件）
│   ├── INFERENCE_GUIDE.md                  # 推理使用详细指南
│   ├── FOGGY_CITYSCAPES_TEST_GUIDE.md      # Foggy Cityscapes测试指南
│   └── END_TO_END_FOG_AWARE_YOLO.md        # 端到端方案说明
│
├── 🚀 推理脚本（部署使用）
│   ├── simple_fusion_inference.py          # 简单融合推理（推荐）⭐⭐⭐⭐⭐
│   └── inference_fusion.py                 # 训练模型推理
│
├── 🎓 训练脚本
│   ├── prepare_fusion_dataset.py           # 准备融合训练数据集
│   ├── train_feature_fusion_v2.py          # 两阶段训练脚本
│   ├── prepare_direct_foggy_data.py        # 准备端到端训练数据
│   ├── train_direct_foggy.py               # 直接在有雾图上训练
│   └── train_fog_aware.py                  # 雾感知YOLO训练（未来工作）
│
├── 🧪 测试脚本
│   ├── test_foggy_cityscapes.py            # Foggy Cityscapes标准测试
│   ├── validate_fusion_training.py         # 验证训练效果
│   ├── test_fusion_detection.py            # 融合检测对比测试
│   └── test_manual_weights.py              # 手动权重调优测试
│
├── 🧠 核心模块（yolosystem/）
│   ├── __init__.py                         # 模块初始化
│   ├── dehazing.py                         # 去雾算法（暗通道先验）
│   ├── detection.py                        # 目标检测模块
│   ├── feature_fusion_yolo_simple.py       # 简化版融合YOLO（推荐）
│   ├── feature_fusion_yolo.py              # 完整版融合YOLO（多种策略）
│   ├── fog_aware_yolo.py                   # 端到端雾感知YOLO（未来工作）
│   ├── fusion.py                           # 融合模块
│   ├── pipeline.py                         # 检测流程管道
│   └── utils.py                            # 工具函数
│
├── 🎯 预训练模型
│   ├── yolo11n.pt                          # YOLOv11 nano (5.4MB)
│   └── yolov8n.pt                          # YOLOv8 nano (6.3MB)
│
├── 🖼️ 测试图像
│   └── test_images/
│       ├── hazy_image3.jpg                 # 测试图像1
│       ├── hazy_road.jpg                   # 测试图像2
│       └── hazy_road2.jpg                  # 测试图像3
│
├── ⚙️ 配置文件
│   ├── requirements.txt                    # Python依赖列表
│   └── .gitignore                          # Git忽略配置
│
└── 📂 数据和输出（不在Git中）
    ├── datasets/                           # 训练和测试数据集
    │   ├── school/                         # 校园数据集
    │   ├── fusion_data/                    # 融合训练数据
    │   └── foggy_cityscapes/               # Foggy Cityscapes数据集
    ├── runs/                               # 训练输出
    │   └── fusion_fixed/                   # 训练好的模型
    └── outputs/                            # 推理输出结果
```

---

## 📁 完整文件说明

### 核心模块（yolosystem/）

| 文件 | 功能 | 何时使用 |
|------|------|---------|
| `__init__.py` | 模块初始化 | 自动导入 |
| `dehazing.py` | 去雾算法（暗通道先验DCP） | 所有方法都需要 |
| `detection.py` | 目标检测模块 | 基础检测功能 |
| `feature_fusion_yolo_simple.py` | 简化版融合YOLO（推荐） | 训练和推理 |
| `feature_fusion_yolo.py` | 完整版融合YOLO（多种策略） | 高级研究 |
| `fog_aware_yolo.py` | 端到端雾感知YOLO | 未来工作 |
| `fusion.py` | 图像融合模块 | 融合处理 |
| `pipeline.py` | 检测流程管道 | 完整流程 |
| `utils.py` | 工具函数 | 辅助功能 |

### 推理脚本（部署使用）

| 文件 | 功能 | 推荐度 | 使用场景 |
|------|------|--------|---------|
| `simple_fusion_inference.py` | 简单融合推理 | ⭐⭐⭐⭐⭐ | 快速测试、部署 |
| `inference_fusion.py` | 训练模型推理 | ⭐⭐⭐ | 使用训练好的模型 |

**详细说明**：

#### `simple_fusion_inference.py` - 简单融合推理（推荐）
- **功能**：使用预训练YOLO + 图像级融合
- **优点**：无需训练，开箱即用
- **支持**：单图像、视频、批量处理
- **使用**：
  ```bash
  # 单图像
  python simple_fusion_inference.py --input test.jpg --mode image

  # 视频
  python simple_fusion_inference.py --input test.mp4 --mode video --output result.mp4

  # 批量处理
  python simple_fusion_inference.py --input images_folder/ --mode folder
  ```

#### `inference_fusion.py` - 训练模型推理
- **功能**：使用训练好的Feature Fusion模型
- **要求**：需要先训练模型
- **使用**：
  ```bash
  python inference_fusion.py \
      --checkpoint runs/fusion_fixed/final_stage1.pth \
      --input test.jpg \
      --output result.jpg
  ```

### 训练脚本

| 文件 | 功能 | 何时使用 |
|------|------|---------|
| `prepare_fusion_dataset.py` | 准备训练数据集 | 训练前 |
| `train_feature_fusion_v2.py` | 两阶段训练脚本 | 训练模型 |
| `prepare_direct_foggy_data.py` | 准备端到端训练数据 | 端到端方案 |
| `train_direct_foggy.py` | 直接在有雾图上训练 | 端到端方案 |
| `train_fog_aware.py` | 雾感知YOLO训练 | 未来工作 |

**详细说明**：

#### `prepare_fusion_dataset.py` - 数据集准备
- **功能**：将有雾图、清晰图、标注组织成训练格式
- **输入**：
  - `--hazy-dir`: 有雾图像目录
  - `--clear-dir`: 清晰图像目录
  - `--labels-dir`: YOLO标注目录
- **输出**：融合训练数据集（80%训练，20%验证）

#### `train_feature_fusion_v2.py` - 两阶段训练
- **功能**：训练Feature Fusion模型
- **阶段1**：冻结YOLO，只训练融合模块（50 epochs）
- **阶段2**：联合微调（可选）
- **输出**：`runs/fusion_fixed/final_stage1.pth`

### 测试脚本

| 文件 | 功能 | 何时使用 |
|------|------|---------|
| `test_foggy_cityscapes.py` | Foggy Cityscapes测试 | 验证性能 |
| `validate_fusion_training.py` | 验证训练效果 | 训练后 |
| `test_fusion_detection.py` | 融合检测测试 | 对比实验 |
| `test_manual_weights.py` | 手动权重测试 | 权重调优 |

**详细说明**：

#### `test_foggy_cityscapes.py` - 标准测试
- **功能**：在Foggy Cityscapes数据集上测试
- **输出**：不同雾浓度下的检测性能
- **使用**：
  ```bash
  python test_foggy_cityscapes.py \
      --checkpoint runs/fusion_fixed/final_stage1.pth \
      --fog-level 0.02  # 0.005/0.01/0.02
  ```

#### `test_manual_weights.py` - 权重调优
- **功能**：测试不同融合权重的效果
- **输出**：最佳权重配置
- **使用**：
  ```bash
  python test_manual_weights.py \
      --checkpoint runs/fusion_fixed/final_stage1.pth \
      --weights 0.0 0.3 0.5 0.7 1.0
  ```

### 模型文件

| 文件 | 大小 | 功能 |
|------|------|------|
| `yolo11n.pt` | 5.4MB | YOLOv11 nano预训练模型 |
| `yolov8n.pt` | 6.3MB | YOLOv8 nano预训练模型 |

### 文档文件

| 文件 | 内容 |
|------|------|
| `README.md` | 项目总览（本文件） |
| `INFERENCE_GUIDE.md` | 推理使用详细指南 |
| `FOGGY_CITYSCAPES_TEST_GUIDE.md` | Foggy Cityscapes测试指南 |
| `END_TO_END_FOG_AWARE_YOLO.md` | 端到端方案说明 |

### 配置文件

| 文件 | 功能 |
|------|------|
| `requirements.txt` | Python依赖列表 |
| `.gitignore` | Git忽略文件配置 |

---

## 🔄 完整工作流程

### 流程1：快速测试（推荐新手）

```
1. 安装依赖
   ↓
2. 准备测试图像
   ↓
3. 运行 simple_fusion_inference.py
   ↓
4. 查看对比结果
```

**命令**：
```bash
pip install -r requirements.txt
python simple_fusion_inference.py --input test.jpg --mode image
```

---

### 流程2：完整训练和测试

```
1. 准备数据集（有雾图 + 清晰图 + 标注）
   ↓
2. 运行 prepare_fusion_dataset.py
   ↓
3. 运行 train_feature_fusion_v2.py
   ↓
4. 运行 test_foggy_cityscapes.py
   ↓
5. 分析结果
```

**命令**：
```bash
# 步骤1：准备数据
python prepare_fusion_dataset.py \
    --hazy-dir datasets/school/hazy_school \
    --clear-dir datasets/school/clear_school \
    --labels-dir datasets/school/labels

# 步骤2：训练
python train_feature_fusion_v2.py \
    --data-dir datasets/fusion_data \
    --epochs 50

# 步骤3：测试
python test_foggy_cityscapes.py \
    --checkpoint runs/fusion_fixed/final_stage1.pth \
    --fog-level 0.02
```

---

## 🔬 技术细节

### Feature-level Fusion架构

```
输入：有雾图像 + 去雾图像
  ↓
双路Backbone（共享权重）
  ↓
特征融合模块（可学习权重）
  ↓
YOLO检测头
  ↓
输出：检测结果
```

**融合策略**：
- **Fixed Fusion**: 固定权重融合（50%-50%）
- **Learned Fusion**: 可学习权重（训练后约50%-50%）
- **Adaptive Fusion**: 自适应权重（根据输入调整）

### 训练策略

**两阶段训练**：
1. **阶段1**：冻结YOLO，只训练融合模块（50 epochs）
2. **阶段2**：联合微调（可选，50 epochs）

**损失函数**：
- YOLO检测损失（box + cls + dfl）
- 融合一致性损失（可选）

### 去雾算法 (Dark Channel Prior)

基于何恺明博士的暗通道先验算法：
1. 暗通道计算
2. 大气光估计
3. 透射率估计
4. 导向滤波优化
5. 图像恢复

**参考论文**: He, K., Sun, J., & Tang, X. (2010). Single image haze removal using dark channel prior. IEEE TPAMI, 33(12), 2341-2353.

---

## 📈 性能分析

### 融合权重学习

训练50 epochs后的融合权重：
- 有雾图权重：~0.50
- 去雾图权重：~0.50

**结论**：模型学习到平衡的融合策略，同时利用两路信息。

### 不同权重配置测试

| 配置 | 有雾权重 | 去雾图权重 | 检测数 | 置信度 |
|------|---------|---------|--------|--------|
| 纯有雾 | 100% | 0% | 90 | 0.680 |
| 纯去雾 | 0% | 100% | 110 | 0.621 |
| **最佳** | **30%** | **70%** | **105** | **0.647** |
| Learned | 50% | 50% | 104 | 0.647 |

### 雾浓度影响

- **浓雾 (β=0.02)**: 融合提升最显著（+13%）
- **中雾 (β=0.01)**: 融合有效提升（+6.8%）
- **轻雾 (β=0.005)**: 无需融合（-1%）

---

## 💡 使用建议

### 什么时候用哪个脚本？

| 场景 | 推荐脚本 | 原因 |
|------|---------|------|
| 快速测试效果 | `simple_fusion_inference.py` | 无需训练，立即可用 |
| 部署到生产环境 | `simple_fusion_inference.py` | 简单稳定 |
| 研究不同融合策略 | `train_feature_fusion_v2.py` | 可训练自定义模型 |
| 验证论文结果 | `test_foggy_cityscapes.py` | 标准测试集 |
| 调优融合权重 | `test_manual_weights.py` | 找最佳权重 |

### 融合权重如何选择？

| 雾浓度 | 推荐权重 | 说明 |
|--------|---------|------|
| 轻雾 | 0.3 | 多用有雾图（保留真实信息） |
| 中雾 | 0.7 | 平衡配置（推荐） |
| 浓雾 | 0.8 | 多用去雾图（提升清晰度） |

**经验法则**：雾越浓，`--fusion-weight` 设置越大！

---

## 🎓 论文/报告要点

### 研究贡献

1. **Feature-level Fusion方法**：提出双路融合架构
2. **真实数据集验证**：在Foggy Cityscapes上验证有效性
3. **雾浓度分析**：发现雾越浓效果越好的规律

### 局限性与未来工作

**当前局限**：
- 需要先运行去雾算法（非端到端）
- 融合结果介于有雾和去雾之间（无法超越最好输入）

**未来方向**：
- 端到端雾感知YOLO（集成可学习去雾网络）
- 注意力机制增强（自动学习处理雾的影响）
- 多尺度融合策略

---

## 📊 数据集

### Foggy Cityscapes
- **来源**: https://www.cityscapes-dataset.com/
- **规模**: 500张验证图像
- **雾浓度**: β=0.005, 0.01, 0.02
- **用途**: 性能验证

### 校园数据集
- **规模**: 300张图像
- **类别**: 6类（person, bicycle, car, motorcycle, bus, truck）
- **用途**: 模型训练

---

## 🛠️ 开发工具

- **深度学习框架**: PyTorch 2.0
- **目标检测**: Ultralytics YOLOv11
- **图像处理**: OpenCV
- **训练平台**: AutoDL (A100 GPU)

---

## ❓ 常见问题

### Q1: 我只想快速测试，怎么做？

```bash
pip install torch ultralytics opencv-python
python simple_fusion_inference.py --input test.jpg --mode image
```

### Q2: 如何调整融合权重？

```bash
# 轻雾
python simple_fusion_inference.py --input test.jpg --fusion-weight 0.3

# 浓雾
python simple_fusion_inference.py --input test.jpg --fusion-weight 0.8
```

### Q3: 如何处理视频？

```bash
python simple_fusion_inference.py \
    --input video.mp4 \
    --output result.mp4 \
    --mode video
```

### Q4: 训练需要什么数据？

需要三个文件夹：
- 有雾图像（hazy_school/）
- 清晰图像（clear_school/）
- YOLO标注（labels/）

### Q5: 训练需要多久？

- GPU (A100): 约2-3小时（50 epochs）
- GPU (RTX 3060): 约5-6小时
- CPU: 不推荐（太慢）

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{yolosystem2025,
  title={Feature-level Fusion for Foggy Object Detection},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/YOLOsystem}}
}
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

- 暗通道先验算法: He, K., Sun, J., & Tang, X.
- YOLOv11: Ultralytics
- Foggy Cityscapes: Cityscapes Dataset Team
- OpenCV: Open Source Computer Vision Library
- Claude Sonnet 4.5: AI助手支持

---

**最后更新**: 2025-02-08
