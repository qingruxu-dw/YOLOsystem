"""
Feature-level Fusion YOLO Network
基于YOLOv11的特征级融合网络，用于雾天目标检测

核心思想：
1. 双路输入：原图 + 去雾图
2. 特征提取：分别提取两路特征
3. 特征融合：在backbone的多个层级进行特征融合
4. 联合检测：使用融合后的特征进行目标检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Add math import
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from typing import List, Tuple, Optional


class FeatureFusionModule(nn.Module):
    """
    特征融合模块

    融合策略：
    1. Concatenation + 1x1 Conv（简单有效）
    2. Attention-based Fusion（自适应权重）
    """

    def __init__(self, in_channels: int, fusion_type: str = 'attention'):
        """
        Args:
            in_channels: 输入特征通道数
            fusion_type: 融合类型 ('concat', 'attention', 'weighted')
        """
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 简单拼接 + 1x1卷积降维
            self.fusion = nn.Sequential(
                Conv(in_channels * 2, in_channels, k=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU()
            )

        elif fusion_type == 'attention':
            # 基于注意力的自适应融合
            self.attention = nn.Sequential(
                Conv(in_channels * 2, in_channels // 4, k=1),
                nn.SiLU(),
                Conv(in_channels // 4, 2, k=1),
                nn.Softmax(dim=1)
            )
            self.fusion = Conv(in_channels, in_channels, k=1)

        elif fusion_type == 'weighted':
            # 可学习的加权融合
            self.weight_original = nn.Parameter(torch.tensor(0.5))
            self.weight_dehazed = nn.Parameter(torch.tensor(0.5))
            self.fusion = Conv(in_channels, in_channels, k=1)

    def forward(self, feat_original: torch.Tensor, feat_dehazed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_original: 原图特征 [B, C, H, W]
            feat_dehazed: 去雾图特征 [B, C, H, W]

        Returns:
            融合后的特征 [B, C, H, W]
        """
        if self.fusion_type == 'concat':
            # 拼接后降维
            concat_feat = torch.cat([feat_original, feat_dehazed], dim=1)
            return self.fusion(concat_feat)

        elif self.fusion_type == 'attention':
            # 自适应注意力融合
            concat_feat = torch.cat([feat_original, feat_dehazed], dim=1)
            attention_weights = self.attention(concat_feat)  # [B, 2, H, W]

            # 分离两个权重
            w_orig = attention_weights[:, 0:1, :, :]  # [B, 1, H, W]
            w_dehz = attention_weights[:, 1:2, :, :]  # [B, 1, H, W]

            # 加权融合
            fused = w_orig * feat_original + w_dehz * feat_dehazed
            return self.fusion(fused)

        elif self.fusion_type == 'weighted':
            # 可学习权重融合
            w_orig = torch.sigmoid(self.weight_original)
            w_dehz = torch.sigmoid(self.weight_dehazed)

            # 归一化权重
            total = w_orig + w_dehz
            w_orig = w_orig / total
            w_dehz = w_dehz / total

            fused = w_orig * feat_original + w_dehz * feat_dehazed
            return self.fusion(fused)


class DualPathBackbone(nn.Module):
    """
    双路Backbone

    结构：
    - 两个共享权重的特征提取器
    - 在多个层级进行特征融合
    """

    def __init__(self, base_model: YOLO, fusion_layers: List[int] = [3, 6, 9]):
        """
        Args:
            base_model: 基础YOLOv11模型
            fusion_layers: 在哪些层进行特征融合
        """
        super().__init__()

        # 获取YOLOv11的backbone
        self.model = base_model.model
        self.fusion_layers = fusion_layers

        # 创建特征融合模块
        self.fusion_modules = nn.ModuleDict()

        # 根据YOLOv11架构确定各层的通道数
        # YOLOv11n的通道数: [64, 128, 256, 512]
        channel_configs = {
            3: 128,   # P2层
            6: 256,   # P3层
            9: 512    # P4层
        }

        for layer_idx in fusion_layers:
            if layer_idx in channel_configs:
                channels = channel_configs[layer_idx]
                self.fusion_modules[f'fusion_{layer_idx}'] = FeatureFusionModule(
                    channels, fusion_type='attention'
                )

    def forward(self, img_original: torch.Tensor, img_dehazed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_original: 原图 [B, 3, H, W]
            img_dehazed: 去雾图 [B, 3, H, W]

        Returns:
            融合后的单个特征张量（用于后续处理）
        """
        # 存储所有层的输出
        feats_original = []
        feats_dehazed = []

        # 原图特征提取 - 通过完整的backbone
        x_orig = img_original
        for i, layer in enumerate(self.model.model[:10]):  # backbone部分
            x_orig = layer(x_orig)
            feats_original.append(x_orig)

        # 去雾图特征提取（共享权重）
        x_dehz = img_dehazed
        for i, layer in enumerate(self.model.model[:10]):
            x_dehz = layer(x_dehz)
            feats_dehazed.append(x_dehz)

        # 在指定层进行特征融合，其他层使用原图特征
        fused_features = []
        for i in range(len(feats_original)):
            if i in self.fusion_layers and f'fusion_{i}' in self.fusion_modules:
                # 融合层
                fusion_module = self.fusion_modules[f'fusion_{i}']
                fused_feat = fusion_module(feats_original[i], feats_dehazed[i])
                fused_features.append(fused_feat)
            else:
                # 非融合层，直接使用原图特征
                fused_features.append(feats_original[i])

        # 返回最后一层的融合特征
        return fused_features[-1]


class FeatureFusionYOLO(nn.Module):
    """
    Feature-level Fusion YOLO完整网络

    架构：
    输入: 原图 + 去雾图
    ├─> 双路Backbone (共享权重)
    ├─> 多层级特征融合
    ├─> Neck (FPN + PAN)
    └─> Head (检测头)
    """

    def __init__(self,
                 model_size: str = 'n',
                 num_classes: int = 6,
                 fusion_layers: List[int] = [3, 6, 9],
                 pretrained: bool = True):
        """
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            num_classes: 类别数量
            fusion_layers: 特征融合层
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        # 加载基础YOLOv11模型
        model_name = f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml'
        self.base_model = YOLO(model_name)

        # 双路Backbone
        self.dual_backbone = DualPathBackbone(self.base_model, fusion_layers)

        # Neck和Head使用原始YOLO的
        self.neck_head = self.base_model.model.model[10:]  # Neck + Head部分

        self.num_classes = num_classes
        self.fusion_layers = fusion_layers
        
        # 自动适配 Head 的类别数
        self._update_detect_head(num_classes)

    def _update_detect_head(self, num_classes):
        """
        根据指定的类别数自动更新Detect Head
        当预训练模型类别数(80)与当前数据类别数不一致时，
        自动替换最后分类层的卷积通道。
        """
        # 遍历 neck_head 找到 Detect 层
        for module in self.neck_head:
            if isinstance(module, Detect):
                if module.nc != num_classes:
                    print(f"自动适配: Detect Head 类别数从 {module.nc} 更新为 {num_classes}")
                    
                    # 1. 更新属性
                    module.nc = num_classes
                    
                    # 2. 更新分类分支的卷积层 (cv3)
                    # Detect 层的 cv3 是一个 ModuleList，包含针对不同尺度的分类头
                    # 每个元素通常是一个 Sequential
                    for i, conv_module in enumerate(module.cv3):
                        # 在 YOLOv8/11 中, cv3 的最后一层是负责输出类别概率的 Conv2d
                        # 结构: [Conv, Conv, ..., Conv2d(c2, nc, 1)]
                        
                        # 获取最后一层
                        last_layer = conv_module[-1]
                        
                        if isinstance(last_layer, nn.Conv2d):
                            # 获取输入通道数
                            in_channels = last_layer.in_channels
                            
                            # 创建新的卷积层 (输出通道 = num_classes)
                            # 注意：这里不需要放到 device，因为整个 model 之后会被 .to(device)
                            new_conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)
                            
                            # 初始化权重 (参考 Ultralytics 的初始化策略)
                            # 这里的 bias 初始化有助于训练初期的稳定性
                            stride = module.stride[i] if hasattr(module, 'stride') else 32
                            # 这里的 bias 初始化是一个经验值，模拟 YOLO 的初始化
                            b = new_conv.bias.view(1, -1)
                            b.data.fill_(-math.log((1 - 0.01) / 0.01))
                            new_conv.bias = torch.nn.Parameter(b.view(-1))
                            
                            # 替换旧层
                            conv_module[-1] = new_conv
                            
                break

    def train(self, mode: bool = True):
        """
        重写train方法，避免调用YOLO的train方法
        """
        # 只设置dual_backbone和neck_head的训练模式
        self.training = mode
        self.dual_backbone.train(mode)
        for layer in self.neck_head:
            if hasattr(layer, 'train'):
                layer.train(mode)
        return self

    def eval(self):
        """
        重写eval方法
        """
        return self.train(False)

    def forward(self, img_original: torch.Tensor, img_dehazed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_original: 原图 [B, 3, H, W]
            img_dehazed: 去雾图 [B, 3, H, W]

        Returns:
            检测结果
        """
        # 获取融合后的backbone特征
        fused_feature = self.dual_backbone(img_original, img_dehazed)

        # 通过Neck和Head
        x = fused_feature
        for layer in self.neck_head:
            x = layer(x)

        return x

    def predict(self, img_original: torch.Tensor, img_dehazed: torch.Tensor,
                conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        预测接口

        Args:
            img_original: 原图
            img_dehazed: 去雾图
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值

        Returns:
            检测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img_original, img_dehazed)
            # 后处理（NMS等）
            # 这里需要根据YOLOv11的输出格式进行后处理
            return outputs


def create_feature_fusion_yolo(model_size: str = 'n',
                               num_classes: int = 6,
                               pretrained: bool = True) -> FeatureFusionYOLO:
    """
    创建Feature-level Fusion YOLO模型

    Args:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重

    Returns:
        FeatureFusionYOLO模型
    """
    model = FeatureFusionYOLO(
        model_size=model_size,
        num_classes=num_classes,
        fusion_layers=[3, 6, 9],  # 在P2, P3, P4层融合
        pretrained=pretrained
    )

    return model


if __name__ == '__main__':
    # 测试代码
    print("创建Feature-level Fusion YOLO模型...")
    model = create_feature_fusion_yolo(model_size='n', num_classes=6)

    # 测试前向传播
    batch_size = 2
    img_size = 640
    img_original = torch.randn(batch_size, 3, img_size, img_size)
    img_dehazed = torch.randn(batch_size, 3, img_size, img_size)

    print(f"输入形状: {img_original.shape}")

    # 前向传播
    outputs = model(img_original, img_dehazed)
    print(f"输出形状: {outputs.shape if isinstance(outputs, torch.Tensor) else 'Multiple outputs'}")

    print("\n模型创建成功！")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
