import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path
import cv2  # Add this import
import torch.optim as optim
import copy


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
    def __init__(self, weights_path=None, device=None, load_clip=False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.clip_model = None

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
        
        # CLIP 专用转换 (Resize to 224 for CLIP)
        self.clip_resize = Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
        
        if load_clip:
            self._load_clip_model()

    def _load_clip_model(self):
        """加载 CLIP 模型用于文本指导"""
        if self.clip_model is not None:
            return

        print("正在加载 CLIP 模型用于文本指导去雾...")
        try:
            # 尝试导入 CLIP
            try:
                from .CoA.CLIP import clip
            except ImportError:
                # Fallback: make sure 'CoA' is on sys processing path
                if coa_root not in sys.path:
                    sys.path.insert(0, coa_root)
                try: 
                     from CLIP import clip
                except ImportError:
                     # One more try: look in current dir's CLIP
                     clip_local = os.path.join(current_dir, 'CoA', 'CLIP')
                     sys.path.insert(0, os.path.dirname(clip_local))
                     import CLIP.clip as clip
                
            # 加载模型
            model_name = "ViT-B/32"
            # 尝试定位 clip_model 目录
            clip_root = os.path.join(coa_root, 'clip_model')
            
            # 检查是否有本地模型，如果没有，CLIP库会自动下载到 download_root 或默认缓存
            # 为了避免用户困惑，我们可以允许 download_root 为 None (使用默认 ~/.cache/clip)
            # 或者如果 clip_root 不存在，就新建一个，让它下载到项目里
            
            if not os.path.exists(clip_root):
                 # 尝试在上级目录寻找 (兼容不同的项目结构)
                 alt_root = os.path.join(os.path.dirname(coa_root), 'clip_model')
                 if os.path.exists(alt_root):
                     clip_root = alt_root
                 else:
                     # 如果都不存在，创建一个目录用于存放下载的模型 (可选，或者直接让 clip 使用默认缓存)
                     # 这里我们保持原样指向 clip_root，download 函数会自动处理下载
                     # 但最好确认目录可以被创建
                     try:
                         os.makedirs(clip_root, exist_ok=True)
                     except:
                         pass

            print(f"CLIP model download root: {clip_root}")
            self.clip_model, _ = clip.load(model_name, device=self.device, download_root=clip_root)
            self.clip_model.eval()
            print(f"CLIP ({model_name}) 加载成功！")
            
        except Exception as e:
            print(f"Warning: 无法加载 CLIP 模型，文本指导功能将不可用。错误: {e}")
            self.clip_model = None

    def process_opencv(self, img_bgr: np.ndarray, prompt: str = None, steps: int = 25) -> np.ndarray:
        """
        处理 OpenCV 格式的图像 (BGR numpy array)
        Args:
            img_bgr: 输入图像
            prompt: (可选) 文本指导描述，例如 "Clear, sharp photo without haze"
            steps: 如果有 prompt，优化的迭代次数
        返回: 去雾后的 BGR numpy array
        """
        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 推理
        if prompt:
            dehazed_pil = self.process_pil_with_text(pil_img, prompt, steps=steps)
        else:
            dehazed_pil = self.process_pil(pil_img)
        
        # PIL RGB -> OpenCV BGR
        res_rgb = np.array(dehazed_pil)
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
        
        return res_bgr
        
    def process_pil_with_text(self, pil_img: Image.Image, prompt: str, steps: int = 15, lr: float = 1e-4) -> Image.Image:
        """
        使用文本 Prompt 指导去雾过程 (Test-Time Adaptation)
        """
        # 确保 CLIP 已加载
        if self.clip_model is None:
            self._load_clip_model()
            if self.clip_model is None:
                print("CLIP 模型未加载，回退到普通去雾模式。")
                return self.process_pil(pil_img)

        w, h = pil_img.size
        
        # 1. 预处理输入
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        h_pad = (h // 16) * 16
        w_pad = (w // 16) * 16
        input_tensor = Resize((h_pad, w_pad), 
                            interpolation=InterpolationMode.BICUBIC, 
                            antialias=True)(img_tensor)

        # 2. 准备文本特征
        import CLIP.clip as clip
        text_token = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            target_text_features = self.clip_model.encode_text(text_token)
            target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)

        # 3. 保存模型原始状态 (必须这样做，因为我们是针对单张图片微调)
        original_state = copy.deepcopy(self.model.state_dict())
        
        # 4. 配置优化器 (微调去雾模型)
        # 开启梯度计算
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 使用 Eval 模式保持 BN 统计量不变，避免单张图片导致 BN 崩溃
        self.model.eval() 
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"正在使用文本 '{prompt}' 优化去雾 (steps={steps})...")
        
        # 5. 优化循环
        try:
            for i in range(steps):
                optimizer.zero_grad()
                
                # 前向传播 (去雾)
                # Student_x 输出通常是 [1, 3, H, W]
                # 注意：Student_x 返回 (out, features_list) 或 out，取决于具体 forward 实现
                # 在原本的 process_pil 中使用的是 out_tensor = self.model(input_tensor)[0]
                # 假设 forward 返回 tuple
                output = self.model(input_tensor)
                if isinstance(output, (tuple, list)):
                    dehazed_img_tensor = output[0]
                else:
                    dehazed_img_tensor = output

                # CLIP 视觉编码 (需 resize 到 224x224)
                # 注意：dehazed_img_tensor 已经是归一化后的数据吗？
                # Student_x 的输入做了 CLIP normalize
                # 如果 Student_x 输出的是像素值 (0-1 或 0-255 并归一化)，我们需要重新做 CLIP normalize
                # 但这里假设输入输出均在类似域内。
                # 为了准确，最好将输出 clamp 到 0-1 然后再 normalize 到 CLIP 空间
                # 但为简化，直接 resize 送入 CLIP (假设输出适配 CLIP 输入分布)
                
                features_in = self.clip_resize(dehazed_img_tensor)
                image_features = self.clip_model.encode_image(features_in)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                # 计算损失: 1 - Cosine Similarity
                # 我们希望 image_features 接近 text_features
                similarity = (image_features @ target_text_features.T)
                loss = 1.0 - similarity.mean()
                
                loss.backward()
                optimizer.step()
                
                if (i+1) % 5 == 0:
                    print(f"Step {i+1}/{steps}, Loss: {loss.item():.4f}, Sim: {similarity.mean().item():.4f}")

            # 6. 生成最终结果
            with torch.no_grad():
                final_out = self.model(input_tensor)
                if isinstance(final_out, (tuple, list)):
                    final_tensor = final_out[0]
                else:
                    final_tensor = final_out

        finally:
            # 7. 无论如何恢复模型原始权重
            self.model.load_state_dict(original_state)
            # 恢复梯度设置 (可选，如果其他地方需要 eval)
            for param in self.model.parameters():
                param.requires_grad = False
                
        # 恢复原始尺寸
        out_tensor = Resize((h, w), 
                          interpolation=InterpolationMode.BICUBIC, 
                          antialias=True)(final_tensor)
        
        out_tensor = out_tensor.squeeze(0).cpu().clamp(0, 1)
        
        from torchvision.transforms import ToPILImage
        res_img = ToPILImage()(out_tensor)
        
        return res_img

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
