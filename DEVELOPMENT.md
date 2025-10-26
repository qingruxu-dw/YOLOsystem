# YOLOsystem 开发指南

## 开发环境设置

### 1. 克隆仓库
```bash
git clone https://github.com/qingruxu-dw/YOLOsystem.git
cd YOLOsystem
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 使用 conda
conda create -n yolosystem python=3.8
conda activate yolosystem
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 项目结构详解

```
YOLOsystem/
├── yolosystem/              # 核心模块包
│   ├── __init__.py         # 包初始化，导出主要类
│   ├── dehazing.py         # 图像去雾模块
│   ├── detection.py        # YOLO目标检测模块
│   ├── pipeline.py         # 完整流水线
│   └── utils.py            # 工具函数
├── demo.py                 # 演示脚本
├── examples.py             # 使用示例代码
├── config.yaml             # 默认配置文件
├── requirements.txt        # 项目依赖
├── .gitignore             # Git忽略文件配置
└── README.md              # 项目文档
```

## 核心模块说明

### 1. DehazingModule (dehazing.py)

实现基于暗通道先验的图像去雾算法。

**主要方法:**
- `get_dark_channel()`: 计算暗通道
- `estimate_atmospheric_light()`: 估计大气光
- `estimate_transmission()`: 估计透射率
- `guided_filter()`: 导向滤波优化
- `recover_image()`: 图像恢复
- `dehaze()`: 完整去雾流程
- `process()`: 简化接口

**参数调优:**
- `omega`: 控制去雾程度，范围[0.85, 0.95]，值越大去雾越强
- `t0`: 最小透射率，防止除零，通常0.1
- `radius`: 导向滤波半径，影响边缘保持，通常10-20
- `eps`: 导向滤波正则化，控制平滑度，通常0.0001-0.001

### 2. YOLODetector (detection.py)

封装Ultralytics YOLO模型的目标检测功能。

**主要方法:**
- `detect()`: 执行目标检测
- `draw_detections()`: 绘制检测框
- `get_model_info()`: 获取模型信息

**支持的模型:**
- yolov8n.pt: 最轻量，速度最快
- yolov8s.pt: 小型，速度和精度平衡
- yolov8m.pt: 中型，较高精度
- yolov8l.pt: 大型，高精度
- yolov8x.pt: 超大型，最高精度

### 3. DehazingDetectionPipeline (pipeline.py)

整合去雾和检测的完整工作流。

**主要方法:**
- `process_image()`: 处理单张图像
- `process_image_file()`: 从文件处理图像
- `process_video()`: 处理视频
- `get_statistics()`: 获取检测统计

## 添加新功能

### 1. 添加新的去雾算法

在 `dehazing.py` 中创建新类:

```python
class NewDehazingMethod:
    def __init__(self, **params):
        # 初始化参数
        pass
    
    def process(self, img: np.ndarray) -> np.ndarray:
        # 实现去雾逻辑
        return dehazed_img
```

### 2. 支持其他检测模型

在 `detection.py` 中扩展或创建新的检测器类:

```python
class CustomDetector:
    def __init__(self, model_path, **kwargs):
        # 加载模型
        pass
    
    def detect(self, img, **kwargs):
        # 执行检测
        return detections
```

### 3. 添加后处理模块

创建新文件 `postprocessing.py`:

```python
def enhance_image(img):
    # 图像增强
    return enhanced_img

def filter_detections(detections, min_size=None, min_conf=None):
    # 过滤检测结果
    return filtered_detections
```

## 测试

### 运行基本测试
```bash
# 测试去雾功能
python demo.py --demo dehazing

# 测试完整流程
python demo.py --demo full

# 处理自定义图像
python demo.py --image test_image.jpg
```

### 性能测试
```python
import time
from yolosystem import DehazingDetectionPipeline

pipeline = DehazingDetectionPipeline()

start = time.time()
results = pipeline.process_image_file('test.jpg')
elapsed = time.time() - start

print(f"处理时间: {elapsed:.2f}秒")
```

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 常见问题

### Q: 如何使用GPU加速？
A: 在配置文件中设置 `device: "cuda"` 或在创建检测器时指定:
```python
detector = YOLODetector(model_path='yolov8n.pt', device='cuda')
```

### Q: 如何调整去雾效果？
A: 调整 `omega` 参数：
- 增大 omega (0.95 -> 0.98): 去雾更强，但可能过度
- 减小 omega (0.95 -> 0.90): 去雾较弱，保留更多细节

### Q: 检测速度太慢怎么办？
A: 
1. 使用更小的模型 (yolov8n)
2. 降低输入图像分辨率
3. 使用GPU加速
4. 减少max_det参数

### Q: 如何只检测特定类别？
A: 在配置文件或代码中设置classes参数:
```python
detector.detect(img, classes=[0, 2, 3])  # 只检测人、汽车、摩托车
```

## 参考资料

- [暗通道先验论文](https://ieeexplore.ieee.org/document/5567108)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [OpenCV 文档](https://docs.opencv.org/)
