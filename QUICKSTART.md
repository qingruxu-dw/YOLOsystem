# 快速开始指南 (Quick Start Guide)

## 系统要求

- Python 3.7+
- 操作系统: Windows, Linux, macOS
- 内存: 建议 4GB+ (使用大模型时需要更多)

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/qingruxu-dw/YOLOsystem.git
cd YOLOsystem
```

### 2. 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**依赖说明:**
- `numpy`: 数值计算
- `opencv-python`: 图像处理
- `torch`, `torchvision`: PyTorch深度学习框架
- `ultralytics`: YOLOv8模型
- `scipy`: 科学计算
- `matplotlib`: 可视化
- `pyyaml`: 配置文件解析

## 快速使用

### 方式1: 运行演示程序

```bash
# 运行所有演示（包括去雾和检测）
python demo.py

# 仅运行去雾演示
python demo.py --demo dehazing

# 仅运行完整流程演示
python demo.py --demo full
```

演示程序会：
1. 创建示例有雾图像
2. 执行去雾处理
3. 进行目标检测
4. 保存结果到 `outputs/` 目录

### 方式2: 处理自己的图像

```bash
# 处理单张图像
python demo.py --image path/to/your/image.jpg

# 使用自定义配置
python demo.py --image path/to/your/image.jpg --config custom_config.yaml
```

### 方式3: 在代码中使用

#### 仅使用去雾功能

```python
from yolosystem import DehazingModule
import cv2

# 初始化去雾模块
dehazer = DehazingModule()

# 读取图像
img = cv2.imread('hazy_image.jpg')

# 执行去雾
dehazed = dehazer.process(img)

# 保存结果
cv2.imwrite('dehazed.jpg', dehazed)
```

#### 使用完整流水线

```python
from yolosystem import DehazingDetectionPipeline

# 创建流水线
pipeline = DehazingDetectionPipeline()

# 处理图像
results = pipeline.process_image_file('input.jpg')

# 查看检测结果
stats = pipeline.get_statistics(results)
print(f"检测到 {stats['total_detections']} 个目标")
```

## 配置说明

编辑 `config.yaml` 来自定义系统行为:

```yaml
# 去雾配置
dehazing:
  enabled: true      # 启用/禁用去雾
  omega: 0.95        # 去雾强度 (0.85-0.98)
  
# 检测配置
detection:
  model: "yolov8n.pt"     # 模型选择
  conf_threshold: 0.25    # 置信度阈值
  device: "cpu"           # cpu 或 cuda
```

### 模型选择

- `yolov8n.pt`: 最快 (推荐测试用)
- `yolov8s.pt`: 快速
- `yolov8m.pt`: 平衡
- `yolov8l.pt`: 高精度
- `yolov8x.pt`: 最高精度 (需要更多资源)

## 输出结果

所有结果默认保存在 `outputs/` 目录:

- `*_dehazed.jpg`: 去雾后的图像
- `*_detection.jpg`: 带检测框的图像
- `*_comparison.jpg`: 对比图 (仅演示模式)

## 常见问题

### Q: 运行时报 "No module named 'ultralytics'"

**A:** 安装依赖: `pip install ultralytics`

### Q: 如何使用GPU加速?

**A:** 
1. 确保安装了支持CUDA的PyTorch
2. 在配置文件中设置 `device: "cuda"`

### Q: 去雾效果不理想怎么办?

**A:** 调整 `omega` 参数:
- 增大 (0.95 -> 0.98): 去雾更强
- 减小 (0.95 -> 0.90): 去雾较弱

### Q: 检测速度太慢

**A:** 
1. 使用更小的模型 (yolov8n)
2. 使用GPU (`device: "cuda"`)
3. 禁用去雾 (`enabled: false`)

## 下一步

- 查看 [README.md](README.md) 了解详细功能
- 查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解开发指南
- 查看 [examples.py](examples.py) 获取更多代码示例

## 获取帮助

- 提交 Issue: https://github.com/qingruxu-dw/YOLOsystem/issues
- 查看文档: 见项目 README

## 许可证

MIT License - 可自由使用和修改
