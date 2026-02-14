import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path
import cv2  # Add this import

# 添加 CoA 目录到系统路径，以便可以导入其中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
coa_root = os.path.join(current_dir, 'CoA')
if coa_root not in sys.path:
    sys.path.insert(0, coa_root)

# 尝试导入模型类
try:
    from model.Student_x import Student_x
except ImportError:
    # 备选导入方案
    try:
        from .CoA.model.Student_x import Student_x
    except ImportError:
        print("Error: 无法导入 Student_x 模型。请确认 yolosystem/CoA 目录结构正确。")
        raise

class CoADehazer:
    """
    CoA 去雾模型适配器
    封装了加载模型和推理的逻辑，使其可以像普通函数一样被调用
    """
    def __init__(self, weights_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 默认权重路径
        if weights_path is None:
            weights_path = os.path.join(coa_root, 'model', 'EMA_model', 'EMA_r.pth')
        
        self.weights_path = weights_path
        
        print(f"Loading CoA Dehazing Model from: {weights_path}")
        print(f"Device: {self.device}")

        # 初始化模型架构
        try:
            self.model = Student_x().to(self.device)
            # 加载权重
            checkpoint = torch.load(weights_path, map_location=self.device)
            # 处理可能的 state_dict 键匹配问题
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("CoA Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load CoA model: {e}")
            raise e

        # 定义预处理转换（参考 Eval.py）
        self.transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])

    def process_opencv(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        处理 OpenCV 格式的图像 (BGR numpy array)
        返回: 去雾后的 BGR numpy array
        """
        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 推理
        dehazed_pil = self.process_pil(pil_img)
        
        # PIL RGB -> OpenCV BGR
        res_rgb = np.array(dehazed_pil)
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
        
        return res_bgr

    def process_pil(self, pil_img: Image.Image) -> Image.Image:
        """
        处理 PIL 图像
        """
        w, h = pil_img.size
        
        # 预处理
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # 为了适应网络结构（通常需要是16的倍数），进行 Resize
        # 参考 Eval.py 中的逻辑
        h_pad = (h // 16) * 16
        w_pad = (w // 16) * 16
        
        input_tensor = Resize((h_pad, w_pad), 
                            interpolation=InterpolationMode.BICUBIC, 
                            antialias=True)(img_tensor)

        # 推理
        with torch.no_grad():
            out_tensor = self.model(input_tensor)[0]

        # 恢复原始尺寸
        out_tensor = Resize((h, w), 
                          interpolation=InterpolationMode.BICUBIC, 
                          antialias=True)(out_tensor)
        
        # 后处理：Tensor -> Image
        # 注意: Eval.py 中没有显式的反归一化，直接 save_image 会自动处理范围吗？
        # torchvision.utils.save_image 会将 tensor 归一化到 [0,1] 如果 normalize=True
        # 这里我们手动处理一下，使其变为 uint8 图像
        
        out_tensor = out_tensor.squeeze(0).cpu().clamp(0, 1) # 假设输出已经是 0-1 范围
        # 如果模型输出不是 0-1，可能需要调整。通常去雾模型输出是重建的图像。
        
        from torchvision.transforms import ToPILImage
        res_img = ToPILImage()(out_tensor)
        
        return res_img

# 为了兼容性，也可以保留 process 方法
    def process(self, img):
        if isinstance(img, np.ndarray):
            import cv2
            return self.process_opencv(img)
        elif isinstance(img, Image.Image):
            return self.process_pil(img)
        else:
            raise ValueError("Input type not supported. Use numpy array (OpenCV) or PIL Image.")
