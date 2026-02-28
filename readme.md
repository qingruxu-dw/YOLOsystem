# 雾天图像目标检测系统 (YOLOsystem)

这是一个基于深度学习的雾天图像目标检测系统，旨在提高低能见度环境下的目标识别准确率。系统提供了两种检测模式：基础模式（DCP + YOLOv11）和多模态融合模式（CoADehazer + 特征融合 YOLO）。

## ✨ 主要功能

*   **基础模式 (Basic Mode)**:
    *   **图像去雾**: 使用暗通道先验 (DCP) 算法快速去除图像中的雾气。
    *   **目标检测**: 使用最新的 **YOLOv11n** 模型进行高效目标识别。
    *   **特点**: 速度快，适用于实时性要求较高的场景。

*   **多模态融合模式 (Fusion Mode)**:
    *   **高级去雾**: 集成 **CoADehazer** (CLIP-oriented Awareness Dehazer) 算法，利用 CLIP 模型感知语义信息，支持文本提示 (Text Prompt) 增强去雾效果。
    *   **特征融合**: 采用双分支特征融合架构 (Dual-Branch Feature Fusion)，同时提取原图和去雾图的特征进行融合，显著提升在浓雾下的检测鲁棒性。
    *   **特点**: 精度高，适用于复杂雾天环境。

*   **Web 可视化界面**:
    *   基于 **Vue 3 + Vite + Element Plus** 构建的现代化 UI。
    *   支持图像上传、模式切换、去雾强度调节。
    *   实时展示检测结果、去雾前后对比及历史记录管理。

## � 项目结构

```text
YOLOsystem/
├── backend/                # Flask 后端服务
│   ├── app.py              # 后端主程序入口
│   ├── models/             # 模型文件 (如 yolo11n.pt)
│   ├── clip_model/         # CLIP 预训练模型
│   ├── dehaze_api.py       # 基础模式去雾逻辑 (DCP)
│   ├── fusion_api.py       # 多模态融合模式接口
│   └── requirements.txt    # Python 依赖清单
├── frontend/               # Vue 3 前端项目
│   ├── src/                # 前端源代码
│   ├── public/             # 静态资源
│   ├── package.json        # Node.js 依赖配置
│   └── vite.config.js      # Vite 构建配置
├── multibackend/           # 高级融合算法核心 (Research & Core Logic)
    ├── yolosystem/         # 核心算法库 (包含 CoADehazer, Fusion 网络定义)
    ├── inference_fusion.py # 融合推理引擎实现
    └── ...                 # 训练、测试脚本及说明文档
```

## 🚀 快速开始

### 1. 环境准备

请确保您的系统已安装以下环境：
*   **Python 3.8+** (建议 3.9 或 3.10)
*   **Node.js 16+** (用于运行前端)
*   **CUDA (可选)**: 如果需要使用 GPU 加速 (推荐用于 Fusion 模式)，请确保安装了对应的 CUDA Toolkit。

### 2. 后端部署 (Backend)

1.  进入后端目录：
    ```bash
    cd backend
    ```

2.  创建并激活虚拟环境 (推荐)：
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

4.  启动后端服务：
    ```bash
    python app.py
    ```
    后端服务默认运行在 `http://localhost:5000`。

### 3. 前端部署 (Frontend)

1.  打开新的终端窗口，进入前端目录：
    ```bash
    cd frontend
    ```

2.  安装依赖：
    ```bash
    npm install
    ```

3.  启动开发服务器：
    ```bash
    npm run dev
    ```
    前端页面通常会自动打开，或访问 `http://localhost:3000`。

## 📝 使用说明

1.  **访问系统**: 打开浏览器访问前端地址 (如 `http://localhost:3000`)。
2.  **注册/登录**: 为了管理历史记录，请先注册一个账号并登录。
3.  **上传图像**: 点击上传区域或拖拽雾天图像文件。
4.  **选择模式**:
    *   **Basic**: 默认模式，速度较快。
    *   **Fusion**: 高级模式，点击切换，可输入文本提示辅助去雾。
5.  **查看结果**: 系统处理完成后，将展示：
    *   **Original**: 原始雾天图像。
    *   **Dehazed**: 去雾后的图像。
    *   **Detected**: 包含目标检测框的结果图像。
6.  **历史记录**: 在侧边栏或历史面板查看之前的处理记录。

## ⚠️ 注意事项

*   **模型文件**: 首次运行 Fusion 模式时，系统可能会尝试加载或下载必要的权重文件 (如 `stage2_best.pth`)，请确保网络连接正常或手动放置权重文件。
*   **性能**: Fusion 模式涉及复杂的深度学习推理，在无 GPU 环境下运行速度可能较慢，属于正常现象。
