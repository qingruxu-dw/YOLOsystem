import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

# ==========================================
# 创新点 3：无参数约束的高频纹理引导分支
# ==========================================
class HighFreqStem(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 使用固定的 Laplacian 算子提取高频边缘 (无参数，不开销显存)
        kernel = torch.tensor([[[[-1., -1., -1.],
                                 [-1.,  8., -1.],
                                 [-1., -1., -1.]]]], dtype=torch.float32)
        self.register_buffer('edge_kernel', kernel) 
        
        # 将 4通道(原图3 + 边缘1) 压缩回 3通道，完美接入原生 YOLO 主干
        self.proj = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1, bias=False)
        
        # 巧妙的初始化：让网络一开始几乎完全忽略边缘，随着蒸馏慢慢吸收边缘特征
        nn.init.dirac_(self.proj.weight[:in_channels, :in_channels]) 
        nn.init.zeros_(self.proj.weight[:, in_channels])

    def forward(self, x):
        # 提取灰度并算边缘
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge = F.conv2d(gray, self.edge_kernel, padding=1)
        # 拼接并降维
        x_cat = torch.cat([x, edge], dim=1)
        return self.proj(x_cat)


# ==========================================
# 创新点 1：环境自适应雾气感知门控
# ==========================================
class HazeAwareGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 极轻量级 MLP，算力开销几近于 0
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 8), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(channels // 4, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        haze_condition = self.pool(x).view(b, c)
        gate_weight = self.mlp(haze_condition).view(b, c, 1, 1)
        return x * gate_weight.expand_as(x)


# ==========================================
# 创新点 2：特征投影对齐适配器 (跨模态桥梁)
# ==========================================
class FeatureAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 用 1x1 卷积完成模态间的维度投影转换，带残差保护
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        # 初始化为0，一开始不干预，完全靠主干
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        return x + self.adapter(x)


# ==========================================
# 完整的带创新点学生网络架构
# ==========================================
class StudentKDYOLO(nn.Module):
    def __init__(self, model_size='n', num_classes=10, fusion_layers=[2, 4, 6]):
        super().__init__()
        # 1. 基础模型
        self.base_model = YOLO(f'yolo11{model_size}.pt')
        self.fusion_layers = fusion_layers
        self.num_classes = num_classes
        
        # 2. 插入高频前端
        self.hf_stem = HighFreqStem()
        
        # 获取特征层通道数字典
        channels_list = self._get_channels(fusion_layers)
        
        # 3. 在 P2 (最浅融合层) 挂载雾气感知门控
        self.haze_gate = HazeAwareGate(channels_list[0])
        
        # 4. 在 所有融合层 挂载对齐适配器
        self.adapters = nn.ModuleDict({
            f'adapter_{layer_idx}': FeatureAdapter(ch) 
            for layer_idx, ch in zip(fusion_layers, channels_list)
        })
        
        # 5. 替换类别检测头 (复用老师的机制)
        self._update_detect_head(num_classes)

    def _get_channels(self, fusion_layers):
        dummy = torch.zeros(1, 3, 256, 256)
        channels = []
        x = dummy
        feats = []
        with torch.no_grad():
            for i, layer in enumerate(self.base_model.model.model[:10]):
                x = layer(x)
                feats.append(x)
        for idx in fusion_layers:
            channels.append(feats[idx].shape[1])
        return channels

    def _update_detect_head(self, num_classes):
        for module in self.base_model.model.model[10:]:
            if isinstance(module, Detect):
                module.nc = num_classes
                module.no = num_classes + module.reg_max * 4
                for i, conv_module in enumerate(module.cv3):
                    last_layer = conv_module[-1]
                    if isinstance(last_layer, nn.Conv2d):
                        in_channels = last_layer.in_channels
                        new_conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)
                        b = new_conv.bias.view(1, -1)
                        b.data.fill_(-math.log((1 - 0.01) / 0.01))
                        new_conv.bias = torch.nn.Parameter(b.view(-1))
                        conv_module[-1] = new_conv
                break

    def forward(self, img_hazy: torch.Tensor):
        x = self.hf_stem(img_hazy)
        backbone_feats = []
        kd_feats = {}
        
        for i, layer in enumerate(self.base_model.model.model[:10]):
            x = layer(x)
            
            # 使用环境门控过滤大雾
            if i == self.fusion_layers[0]:
                x = self.haze_gate(x)
                
            # 经过适配器对齐特征，并存储用于KD蒸馏
            if i in self.fusion_layers:
                mapped_x = self.adapters[f'adapter_{i}'](x)
                kd_feats[i] = mapped_x
                
            backbone_feats.append(x)

        x = list(backbone_feats)
        for layer in self.base_model.model.model[10:]:
            if hasattr(layer, 'f') and layer.f != -1:
                if isinstance(layer.f, int):
                    idx = layer.f if layer.f >= 0 else layer.f + len(x)
                    layer_out = layer(x[idx])
                else:
                    inputs = [x[j if j >= 0 else j + len(x)] for j in layer.f]
                    layer_out = layer(inputs)
            else:
                layer_out = layer(x[-1])
            x.append(layer_out)

        return x[-1], kd_feats

if __name__ == "__main__":
    print("正在实例化 KD 学生模型...")
    model = StudentKDYOLO(model_size='n', num_classes=10)
    dummy_in = torch.randn(2, 3, 640, 640)
    out, kd_feats = model(dummy_in)
    print(f"学生网络检测头输出形状: {out[0].shape if isinstance(out, list) else out.shape}")
    print(f"提取的 KD 蒸馏特征层(准备与老师对齐): {list(kd_feats.keys())}")
    print("模型验证成功，参数全部接通！")
