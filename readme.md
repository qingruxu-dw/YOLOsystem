📖 项目简介（旧的，未更新，谨慎参考）

一个基于暗通道先验算法和YOLOv8的雾天图像目标检测系统。该系统能够对雾天图像进行自动去雾处理，并在去雾后的图像上进行目标检测识别。



✨ 主要功能

🖼️ 图像去雾：使用暗通道先验算法去除图像中的雾气



🔍 目标检测：基于YOLOv8检测图像中的各种目标



🌐 Web界面：提供友好的浏览器操作界面



💻 命令行工具：支持批量处理和多模式操作



📊 结果管理：自动保存处理结果和历史记录



📁 项目结构：
Debaze\_Project/

├── backend/                 # 后端服务

│   ├── models/             # 模型文件目录

│   │   └── yolov8n.pt     # YOLOv8预训练模型

│   ├── app.py             # Flask后端主程序

│   ├── dehaze\_api.py      # 去雾检测核心算法

│   └── requirements.txt   # Python依赖包

├── frontend/              # 前端界面

│   └── index.html         # Web主页面

├── test\_images/           # 测试图像集

├── outputs/               # 处理结果输出目录

├── weights/               # 模型权重备份

├── main.py               # 命令行主程序

└── README.md             # 项目说明文档



🚀 快速开始

环境要求

Python 3.8+

pip 包管理器



