"""
Feature-level Fusion YOLO Network - 简化版本
基于YOLOv11的特征级融合网络，用于雾天目标检测

简化方案：
1. 双路输入：原图 + 去雾图
2. 早期融合：在输入层或浅层进行融合
3. 单路检测：使用标准YOLO进行检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from typing import Optional


class InputFusionModule(nn.Module):
    """
    输入层融合模块
    在图像输入层面进行融合，然后送入标准YOLO
    """

    def __init__(self, fusion_type: str = 'learned'):
        """
        Args:
            fusion_type: 融合类型 ('average', 'learned', 'attention')
        """
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'learned':
            # 可学习的权重
            self.weight_original = nn.Parameter(torch.tensor(0.5))
            self.weight_dehazed = nn.Parameter(torch.tensor(0.5))

        elif fusion_type == 'attention':
            # 注意力机制
            self.attention = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )

    def forward(self, img_original: torch.Tensor, img_dehazed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_original: 原图 [B, 3, H, W]
            img_dehazed: 去雾图 [B, 3, H, W]

        Returns:
            融合后的图像 [B, 3, H, W]
        """
        if self.fusion_type == 'average':
            # 简单平均
            return (img_original + img_dehazed) / 2.0

        elif self.fusion_type == 'learned':
            # 可学习权重融合
            w_orig = torch.sigmoid(self.weight_original)
            w_dehz = torch.sigmoid(self.weight_dehazed)
            total = w_orig + w_dehz
            return (w_orig * img_original + w_dehz * img_dehazed) / total

        elif self.fusion_type == 'attention':
            # 注意力融合
            concat = torch.cat([img_original, img_dehazed], dim=1)  # [B, 6, H, W]
            attention_weights = self.attention(concat)  # [B, 2, H, W]

            w_orig = attention_weights[:, 0:1, :, :]  # [B, 1, H, W]
            w_dehz = attention_weights[:, 1:2, :, :]  # [B, 1, H, W]

            return w_orig * img_original + w_dehz * img_dehazed


class FeatureFusionYOLO(nn.Module):
    """
    Feature-level Fusion YOLO - 简化版本

    架构：
    输入: 原图 + 去雾图
    ├─> 输入层融合
    └─> 标准YOLO检测
    """

    def __init__(self,
                 model_size: str = 'n',
                 num_classes: int = 6,
                 fusion_type: str = 'learned',
                 pretrained: bool = True):
        """
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            num_classes: 类别数量
            fusion_type: 融合类型
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        # 输入融合模块
        self.fusion_module = InputFusionModule(fusion_type)

        # 加载标准YOLO模型
        model_name = f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml'
        self.yolo = YOLO(model_name)

        self.num_classes = num_classes
        self.model_size = model_size

    def forward(self, img_original: torch.Tensor, img_dehazed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_original: 原图 [B, 3, H, W]
            img_dehazed: 去雾图 [B, 3, H, W]

        Returns:
            检测结果
        """
        # 融合输入
        fused_img = self.fusion_module(img_original, img_dehazed)

        # 通过YOLO模型
        # 注意：这里直接调用model的forward，不是YOLO的predict
        outputs = self.yolo.model(fused_img)

        return outputs

    def train(self, mode: bool = True):
        """重写train方法"""
        self.training = mode
        self.fusion_module.train(mode)
        # 确保整个 YOLO 模型都切换到正确的模式
        if hasattr(self, 'yolo') and hasattr(self.yolo, 'model'):
            self.yolo.model.train(mode)
        return self

    def eval(self):
        """重写eval方法"""
        return self.train(False)

    def compute_loss(self, preds, batch):
        """
        计算YOLO损失

        Args:
            preds: 模型预测输出
            batch: 批次数据，包含图像和标签

        Returns:
            loss: 总损失
        """
        # 使用YOLO内置的损失计算
        # 需要访问YOLO模型的loss函数
        if hasattr(self.yolo.model, 'loss'):
            # 直接使用YOLO的损失函数
            return self.yolo.model.loss(preds, batch)
        else:
            # 如果没有loss属性，尝试使用trainer的criterion
            from ultralytics.utils.loss import v8DetectionLoss

            # 创建损失计算器
            if not hasattr(self, 'criterion'):
                self.criterion = v8DetectionLoss(self.yolo.model)

            return self.criterion(preds, batch)


def create_feature_fusion_yolo(model_size: str = 'n',
                               num_classes: int = 6,
                               fusion_type: str = 'learned',
                               pretrained: bool = True) -> FeatureFusionYOLO:
    """
    创建Feature-level Fusion YOLO模型

    Args:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        num_classes: 类别数量
        fusion_type: 融合类型 ('average', 'learned', 'attention')
        pretrained: 是否使用预训练权重

    Returns:
        FeatureFusionYOLO模型
    """
    model = FeatureFusionYOLO(
        model_size=model_size,
        num_classes=num_classes,
        fusion_type=fusion_type,
        pretrained=pretrained
    )

    return model


if __name__ == '__main__':
    # 测试代码
    print("创建Feature-level Fusion YOLO模型（简化版）...")
    model = create_feature_fusion_yolo(model_size='n', num_classes=6, fusion_type='learned')

    # 测试前向传播
    batch_size = 2
    img_size = 640
    img_original = torch.randn(batch_size, 3, img_size, img_size)
    img_dehazed = torch.randn(batch_size, 3, img_size, img_size)

    print(f"输入形状: {img_original.shape}")

    model.eval()
    with torch.no_grad():
        outputs = model(img_original, img_dehazed)
        print(f"输出: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"  输出{i}形状: {out.shape}")
        elif isinstance(outputs, torch.Tensor):
            print(f"  输出形状: {outputs.shape}")

    print("\n模型创建成功！")
    print(f"融合模块参数: {sum(p.numel() for p in model.fusion_module.parameters())}")
    print(f"总参数: {sum(p.numel() for p in model.parameters())}")
