# 去雾模型替换方案说明 (CoA Model Integration)

本文档记录了将 YOLO 系统中的传统去雾模块（Dark Channel Prior）替换为基于深度学习的 CoA 去雾模型的过程和使用指南。

## 1. 变更概述

*   **原去雾方案**: 基于传统算法的 `DehazingModule` (yolosystem/dehazing.py)。
*   **新去雾方案**: 基于 CoA (Contrastive Learning) 的深度学习模型 `Student_x`。
*   **模型权重**: 使用 `YOLOsystem/yolosystem/CoA/model/EMA_model/EMA_r.pth`。
*   **目的**: 提升去雾效果，并让检测网络利用高质量的去雾特征。

## 2. 核心修改内容

我们通过以下三个主要步骤完成了替换：

### 2.1 新增适配器 (`yolosystem/coa_adapter.py`)
创建了一个 `CoADehazer` 类，作为 CoA 模型与 YOLO 系统之间的桥梁。
*   **功能**:
    *   自动加载 CoA 模型结构和权重。
    *   处理 OpenCV (BGR) 和 PIL 格式之间的转换。
    *   处理输入尺寸调整（Padding 到 16 的倍数）和 Tensor 归一化。
*   **优势**: 对上层代码屏蔽了深度学习模型的复杂性，接口与原 `DehazingModule` 保持类似。

### 2.2 修改数据准备 (`prepare_fusion_dataset.py`)
更新了数据集生成逻辑，加入了 `use_model_dehazing` 开关。
*   **变更**: 在生成 Fusion 数据集时，脚本现在会调用 `CoADehazer` 对每一张有雾训练图进行处理。
*   **结果**: `images_dehazed` 文件夹中现在保存的是**CoA 模型生成的去雾图**，而不是简单的 GT 清晰图。
*   **意义**: 确保“训练时的去雾输入”与“推理时的去雾输入”分布一致，最大化联合训练的效果。

### 2.3 修改推理脚本 (`inference_fusion.py`)
更新了推理流程，使其在测试时实时调用 CoA 模型。
*   **流程**: `输入图片` -> `CoADehazer` -> `去雾图片` -> `Fusion YOLO (输入 + 去雾)` -> `检测结果`。

## 3. 使用指南

要启用新的去雾模型进行训练和测试，请按照以下步骤操作：

### 步骤 1: 准备环境
确保已安装 CoA 所需的依赖（如 clip, torch 等）。适配器代码会自动处理路径导入，只要 `yolosystem/CoA` 目录存在即可。

### 步骤 2: 重新生成数据集 (关键)
因为更换了去雾模型，必须重新生成用于联合训练的数据集。

```bash
cd /data/home/sczd119/run/YOLOsystem

# 运行数据准备脚本
# 注意：请先检查脚本中的 raw_data 路径是否指向您的真实数据位置
python3 prepare_fusion_dataset.py
```

脚本运行后，`datasets/fusion_training` 目录下将包含由 CoA 模型处理过的去雾图像。

### 步骤 3: 启动联合训练
使用新的数据集训练 Feature Fusion YOLO 模型。

```bash
# 训练脚本会自动读取 datasets/fusion_training 中的数据
python3 train_feature_fusion_v2.py
```

### 步骤 4: 推理与测试
使用训练好的权重进行推理。系统会自动加载 CoA 模型进行实时去雾。

```bash
# 请将 weights_path 替换为您训练出的 best.pth 路径
python3 inference_fusion.py --checkpoint_path runs/feature_fusion/exp/weights/best.pth
```

## 4. 常见问题排查

*   **ImportError**: 如果提示找不到 `Student_x`，请检查 `yolosystem/CoA` 目录结构是否完整。
*   **CUDA Out of Memory**: CoA 模型比传统算法占用更多显存。如果在数据准备或训练时遇到显存不足，尝试减小 `batch_size`。
*   **路径问题**: 默认权重路径硬编码为 `yolosystem/CoA/model/EMA_model/EMA_r.pth`，如果移动了文件，请在 `coa_adapter.py` 中修改或在初始化时传入新路径。
