# YOLOsystem - 去雾和目标检测系统——基础模式

这是目标检测系统中的基础模式。系统首先使用暗通道先验(Dark Channel Prior)算法对有雾图像进行去雾处理，然后使用YOLOv8进行目标检测，适用于雾霾天气下的视觉识别任务。

## ✨ 功能特性

- 🌫️ **图像去雾**: 基于暗通道先验(DCP)算法的高质量图像去雾
- 🎯 **目标检测**: 集成YOLOv8系列模型进行实时目标检测
- 🔄 **完整流水线**: 自动化的去雾+检测工作流
- 📹 **视频处理**: 支持视频文件的批量处理
- ⚙️ **灵活配置**: YAML配置文件，方便调整参数
- 📊 **结果可视化**: 自动保存处理结果和检测框

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/qingruxu-dw/YOLOsystem.git
cd YOLOsystem

# 安装依赖
pip install -r requirements.txt
```

### 运行演示

```bash
# 运行所有演示
python demo.py

# 仅演示去雾功能
python demo.py --demo dehazing

# 演示完整流程（去雾+检测）
python demo.py --demo full

# 处理自定义图像
python demo.py --image path/to/your/image.jpg

# 使用自定义配置
python demo.py --image path/to/your/image.jpg --config config.yaml
```

## 📖 使用说明

### 1. 仅使用去雾模块

```python
from yolosystem import DehazingModule
import cv2

# 初始化去雾模块
dehazer = DehazingModule(omega=0.95, t0=0.1, radius=15, eps=0.001)

# 读取图像
img = cv2.imread('hazy_image.jpg')

# 执行去雾
dehazed_img = dehazer.process(img)

# 保存结果
cv2.imwrite('dehazed_image.jpg', dehazed_img)
```

### 2. 仅使用目标检测模块

```python
from yolosystem import YOLODetector
import cv2

# 初始化检测器
detector = YOLODetector(model_path='yolov8n.pt', device='cpu')

# 读取图像
img = cv2.imread('image.jpg')

# 执行检测
detections = detector.detect(img, conf_threshold=0.25)

# 绘制检测框
result_img = detector.draw_detections(img, detections)

# 保存结果
cv2.imwrite('detection_result.jpg', result_img)
```

### 3. 使用完整流水线

```python
from yolosystem import DehazingDetectionPipeline

# 初始化流水线（使用配置文件）
pipeline = DehazingDetectionPipeline(config_path='config.yaml')

# 处理单张图像
results = pipeline.process_image_file('hazy_image.jpg')

# 获取检测统计
stats = pipeline.get_statistics(results)
print(f"检测到 {stats['total_detections']} 个目标")
print(f"各类别数量: {stats['class_counts']}")

# 处理视频
pipeline.process_video('input_video.mp4', 'output_video.mp4')
```

## 📁 项目结构

```
YOLOsystem/
├── yolosystem/              # 核心模块
│   ├── __init__.py         # 模块初始化
│   ├── dehazing.py         # 去雾模块（暗通道先验算法）
│   ├── detection.py        # YOLO检测模块
│   ├── pipeline.py         # 完整流水线
│   └── utils.py            # 工具函数
├── demo.py                 # 演示脚本
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明
```

## ⚙️ 配置说明

配置文件 `config.yaml` 包含以下设置：

```yaml
# 去雾设置
dehazing:
  enabled: true          # 是否启用去雾
  omega: 0.95           # 去雾程度 (0-1)
  t0: 0.1               # 最小透射率
  radius: 15            # 导向滤波半径
  eps: 0.001            # 导向滤波正则化

# 检测设置
detection:
  model: "yolov8n.pt"   # 模型路径
  conf_threshold: 0.25   # 置信度阈值
  iou_threshold: 0.45    # NMS IOU阈值
  max_det: 300          # 最大检测数
  device: "cpu"         # 设备 (cpu/cuda)

# 流水线设置
pipeline:
  save_dehazed: true    # 保存去雾图像
  save_detections: true # 保存检测结果
  output_dir: "outputs" # 输出目录
```

## 🔬 技术细节

### 去雾算法 (Dark Channel Prior)

本系统使用何恺明博士提出的暗通道先验算法：

1. **暗通道计算**: 提取图像的暗通道特征
2. **大气光估计**: 估计全局大气光值
3. **透射率估计**: 计算场景透射率
4. **导向滤波**: 使用导向滤波优化透射率图
5. **图像恢复**: 根据大气散射模型恢复清晰图像

**参考论文**: He, K., Sun, J., & Tang, X. (2010). Single image haze removal using dark channel prior. IEEE TPAMI, 33(12), 2341-2353.

### 目标检测 (YOLOv8)

使用Ultralytics YOLOv8系列模型：

- **YOLOv8n**: 最快速，适合实时应用
- **YOLOv8s**: 速度与精度平衡
- **YOLOv8m**: 中等精度
- **YOLOv8l**: 高精度
- **YOLOv8x**: 最高精度

## 📊 性能优化

- 支持GPU加速 (CUDA)
- 批量处理优化
- 可配置的图像分辨率
- 多线程视频处理

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

## 🙏 致谢

- 暗通道先验算法: He, K., Sun, J., & Tang, X.
- YOLOv8: Ultralytics
- OpenCV: Open Source Computer Vision Library
