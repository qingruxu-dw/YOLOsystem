📖 项目简介（旧的，未更新，谨慎参考）

一个基于暗通道先验算法和YOLOv8的雾天图像目标检测系统。该系统能够对雾天图像进行自动去雾处理，并在去雾后的图像上进行目标检测识别。



✨ 主要功能
基础模式：
图像去雾：使用暗通道先验算法去除图像中的雾气
目标检测：基于YOLOv8检测图像中的各种目标



🌐 Web界面：提供友好的浏览器操作界面




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

└── README.md             # 项目说明文档



🚀 快速开始

环境要求

Python 3.8+

pip 包管理器


依赖安装：

- 在项目根目录创建虚拟环境： python -m venv .venv
- 激活： .\.venv\Scripts\activate
- 安装后端依赖： pip install -r backend\requirements.txt

运行 后端：
- 进入后端目录： cd backend
- 启动后端： python app.py
运行前端：
- 进入前端目录： cd frontend
- 启动前端： npm run dev