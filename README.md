# Depth-Aware Foggy YOLO Fusion (YOLOsystem + CoA)

本项目实现了一种针对真实雾天场景的端到端目标检测框架。针对现实场景中“雾浓度不均且与景深强相关”的物理特性，我们引入了基于深度感知的雾气模拟机制，并提出了一种双流网络：底层通过结合预训练的 CoA (Contrastive Learning for Dehazing) 模型进行去雾增强特征提取，上层结合 YOLO (YOLOv8/v10/v11/RT-DETR) 的强悍特征检测能力。该模型实现了在雾天环境下目标检测精度的极大提升。

## 🌟 主要特性 (Features)

- **基于景深的真实物理雾气模拟：** 并非简单的全局蒙层叠加。利用 MiDaS 深度图估计技术捕捉图像景深分布，配合大气散射模型 (Atmospheric Scattering Model) 合成雾浓度随景深变化的逼真雾天数据，并在 VisDrone 数据集上构建。
- **CoA 自监督去雾与分布对齐：** 利用构建好的真实物理雾天数据，可对强大的去雾网络 CoA (Contrastive Learning for Dehazing) 进行精调或训练。
- **端到端双流检测与在线去雾显示：** 无须事先手动将雾天图片全都经过去雾模型算一遍！模型训练及推理过程中，**双流结构会直接将原始“雾图”作为输入**。检测网络会自动与内置的 CoA 特征分支进行融合；并且在训练过程中，**融合网络还会自动生成和保存去雾后的图像结果**。

---

## 🛠️ 环境配置 (Prerequisites)

本项目基于 Python 3.8+ 及 PyTorch 框架。建议使用 Anaconda 创建独立的虚拟环境：

```bash
# 克隆仓库
git clone https://github.com/your-username/Depth-Aware-Foggy-YO.git
cd Depth-Aware-Foggy-YO

# 建议使用创建好的 conda 环境
conda env create -f environment.yml
# 或者 pip 安装依赖：
# pip install -r requirements.txt
```

---

## 📁 项目结构 (Project Structure)

为了让模型顺利运行，请确保您的项目文件及数据集按照以下结构进行组织（已省略与本项目运行无关的日志及缓存文件）：

```text
Depth-Aware-Foggy-YOLO/
│
├── yolosystem/                  # 🧠 核心：双流融合及模型结构的具体实现
│   ├── CoA/                     # CoA 去雾模型的实现库 (原作者底层实现)
│   ├── coa_adapter.py           # 适配器：衔接 CoA 输出至目标检测网络
│   ├── feature_fusion_yolo.py   # ★ 融合架构主网络构建代码
│   ├── fusion_dataset.py        # ★ 数据加载：双流输入与端到端协同构建
│   └── fusion.py                # 特征融合操作逻辑
│
├── datasets/                    # 🎯 数据集目录 (构建的端到端融合数据集)
│   └── fusion_training/         # 融合网络训练专属数据格式
│       ├── train/               # 训练集目录
│       │   ├── images_original/ # 训练集 - 存放带雾图 (Hazy)
│       │   ├── images_dehazed/  # 训练集 - 存放预去雾图 (若采用双流补偿)
│       │   ├── clear/           # 训练集 - 存放 GT 清晰原图
│       │   └── labels/          # 训练集 - YOLO 格式标签 (.txt)
│       ├── VisDrone2019-DET-val/# 验证集目录
│       │   ├── images/          # 验证集 - 默认测试用的带雾图
│       │   ├── images_dehazed/  # 验证集 - 去雾图
│       │   ├── clear/           # 验证集 - GT 清晰图
│       │   └── labels/          # 验证集 - 标签
│       └── dataset.yaml         # 模型读取该数据集的目标配置
│
├── weights/                     # 🎯 权重存放处 (请将下载好的权重放于此)
│   ├── coa_best.pth             # CoA 预训练去雾权重
│   └── yolo11n.pt               # YOLOv11 基础权重
│
├── train_feature_fusion_v2.py   # ★ 核心：双流融合网络端到端训练代码
├── inference_fusion.py          # ★ 核心：融合网络推理及输出去雾/检测可视化图
├── eval_fusion.py               # ★ 核心：针对带雾数据集的精度评测代码
│
├── prepare_visdrone.py          # 数据脚本：转标签为 YOLO 格式
├── deepaddhaze.py               # 数据脚本：Midas深度估计 & 大气散射物理加雾
│
├── environment.yml              # Conda 环境依赖
├── requirements.txt             # Pip 依赖
└── README.md                    # 项目说明
```

---

## 🚀 详细指南 (Pipeline)

整个工程的使用流程主要分为3个核心步骤：

### 步骤一：数据格式转换与生成真实分布的物理雾图
我们以 [VisDrone](http://aiskyeye.com/) 无人机数据集为例：
1. **下载并转化标签:** 运行 `prepare_visdrone.py`。该脚本负责将原始 VisDrone 标注文件解析并转换为 YOLO 指定格式 (`.txt` 文件，坐标归一化)。
   ```bash
   python prepare_visdrone.py 
   ```
2. **深度估计与加雾:** 运行 `deepaddhaze.py`。该脚本会调用 MiDaS 生成影像深度图，然后利用真实的大气散射机制对每张图像进行加雾处理（可通过代码调整生成轻雾、中雾、浓雾数据）。
   ```bash
   python deepaddhaze.py
   ```
   *执行完此步骤后，您将获得完全配对的 清晰图像(Clean)、带雾图像(Haze) 以及 深度图(Depth)。这些将作为下一步以及整个检测网络的基石训练集。*


### 步骤二：独立训练或精调 CoA 去雾模型 (可选)
如果项目中需要针对你的“特定数据集（如上步通过 VisDrone 生成的自制复杂雾天数据）”重新精调去雾主干权重，您需要训练咱们结构图中的去雾核心框架。
> 🔗 **CoA 模型参考资源：** 您可以从原项目仓库下载 CoA 模型结构及源码：[CoA Dehazing GitHub Repository](https://github.com/YanZhang-zy/CoA.git) (请替换为实际使用的 CoA 仓库链接)。
>
> 注意：此步生成的权重是为了提取有效的“去雾特征”，并提供给第三步的双流 YOLO 系统使用。
1. 跳转至 CoA 目录，利用已经生成的 [Clean-Hazy] 成对图片，执行去雾训练操作（可参考 CoA 相关的使用文档说明）。
2. 在该步骤中，模型将训练提取去模糊细节分布的权重矩阵。


### 步骤三：双流融合网络端到端训练 (重点)
当您的 CoA 模型训练完毕并获取到预训练权重（如 `coa_best.pth`）后，即可开启目标检测。
**重要特性：** 您不需要将加雾图片的去雾中间结果提取并保存出来！我们框架支持 **“端到端原图输入”** 的形式，通过联合训练，它在提取去雾特征补偿 YOLO 检测精度的同时，也会 **自动可视化并保存去雾图片** 以供观察！
```bash
python train_feature_fusion_v2.py \
    --data-dir datasets/fusion_training \
    --model-size s \
    --num-classes 10 \
    --do-stage2
```
> 注：模型预训练权重自动寻找（代码内置）。上述 `--data-dir` 需指定刚刚准备好带有`train`和`VisDrone2019-DET-val`等划分的端到端融合数据路径（例如 `datasets/fusion_training`），`--do-stage2` 表示启用阶段2进行全解冻微调。

- **中间验证/去雾结果去哪了？** 
  随着每一个 Epoch 迭代及验证阶段进行，融合架构自动推理出的**中间生成（去雾增强）的影像图、标签框**等结果将自动并集中保存在默认输出路径 `runs/two_stage_training/` 或您指定的输出目录内。

---

## 📈 评估与推理 (Evaluation & Inference)

### 雾天场景直接评测
想要量化我们的方案相较于传统 YOLO 的优势，可以通过如下指令，在有雾环境的验证集上评测 AP、F1 分数等指标：
```bash
python test_fusion_detection.py \
    --checkpoint runs/visdrone_fusion_v4/best.pth \
    --data-dir datasets/fusion_training/VisDrone2019-DET-val/images \
    --model-size s \
    --do-eval
```
*(注：如果需要和单分支 baseline 对比，请传入对应的预训练 YOLO 权重，如 `--yolo-weights yolov11_visdrone.pt`)*

### 端到端图片/视频融合推理
使用融合后的最佳权重文件，直接实现“看穿”雾气的检测显示。支持图片与视频：
```bash
# 图像推理
python inference_fusion.py \
    --checkpoint runs/visdrone_fusion_v4/best.pth \
    --input test_images/sample.jpg \
    --model-size s \
    --output outputs/result.jpg

# 视频推理 (加上 --video 参数)
python inference_fusion.py \
    --checkpoint runs/visdrone_fusion_v4/best.pth \
    --input test_videos/sample.mp4 \
    --model-size s \
    --video \
    --output outputs/result_video.mp4
```
同样，网络跑出来的“去雾+检测框结果”会随输出保存在设定的 `--output` 路径下。

---

## 📦 预训练权重下载 (Pre-trained Weights)

针对嫌麻烦想直接跑通推理的用户，我们提供了已经训练好的权重文件（包含 CoA 去雾预训练权重以及 YOLO 融合网络最优权重）：

| 模型权重 | 参数规模 | 链接 | 提取码 |
| -------- | -------- | ---- | ------ |
| Stage-2 Best (Fusion) | `n` | [Baidu/Google Drive Link](#) | `xxxx` |
| CoA Dehazing Pre-trained| - | [Baidu/Google Drive Link](#) | `xxxx` |

请下载后将其存放在 `weights/` 或者您自定义的目录下，并在执行相关命令时使用对应的参数（如 `--checkpoint` 或 `--dehaze-weights`）指定您放置权重的路径。

---

## 📜 Acknowledgments (致谢)

本项目借鉴与集成了以下优秀的开源成果，特此感谢（无特定排名先后）：

- [Ultralytics (YOLO)](https://github.com/ultralytics/ultralytics) : 提供了极佳的基础目标检测特征提取和易扩展的框架。
- [MiDaS: Robust Depth Estimation](https://github.com/isl-org/MiDaS) : 提供了稳定、多环境泛化的深度图推演能力。
- [VisDrone Dataset](http://aiskyeye.com/) : 提供了用于训练与评估此方案极其优质的低空/无人机视角数据基准。
- [CoA Dehazing](https://github.com/YanZhang-zy/CoA.git) : 为项目提供了强实力的图像解雾恢复基网。