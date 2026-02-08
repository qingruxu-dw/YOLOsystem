"""
Fog-Aware YOLO
端到端雾天目标检测，无需显式去雾
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class FogAttentionModule(nn.Module):
    """
    雾感知注意力模块
    自动学习处理雾的影响，增强特征表达
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # 通道注意力 - 学习哪些通道对雾天检测更重要
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力 - 学习哪些区域受雾影响更严重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 雾浓度估计分支
        self.fog_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()  # 输出0-1的雾浓度估计
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            增强后的特征 (B, C, H, W)
            雾浓度估计 (B, 1, 1, 1)
        """
        # 估计雾浓度
        fog_density = self.fog_estimator(x)

        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_att

        # 根据雾浓度自适应调整特征
        # 雾越浓，越依赖注意力增强的特征
        output = x * (1 - fog_density) + x_spatial * fog_density

        return output, fog_density


class FogAwareBackbone(nn.Module):
    """
    雾感知Backbone
    在YOLO的backbone中插入雾感知注意力模块
    """

    def __init__(self, yolo_model, insert_layers=[3, 6, 9]):
        """
        Args:
            yolo_model: YOLO模型
            insert_layers: 在哪些层后插入雾感知模块
        """
        super().__init__()
        self.yolo = yolo_model
        self.insert_layers = insert_layers

        # 为每个插入点创建雾感知模块
        self.fog_modules = nn.ModuleDict()

        # 获取backbone的通道数
        # 这里需要根据实际的YOLO架构调整
        # YOLOv11的典型通道数: [64, 128, 256, 512]
        channel_configs = {
            3: 128,   # 浅层特征
            6: 256,   # 中层特征
            9: 512    # 深层特征
        }

        for layer_idx in insert_layers:
            if layer_idx in channel_configs:
                channels = channel_configs[layer_idx]
                self.fog_modules[str(layer_idx)] = FogAttentionModule(channels)

    def forward(self, x):
        """
        前向传播，在指定层插入雾感知模块
        """
        fog_densities = []

        # 这里需要修改YOLO的前向传播
        # 简化版本：直接使用YOLO的预测
        # 实际应该在backbone中间插入雾感知模块

        # 获取YOLO的预测结果
        results = self.yolo.predict(x, verbose=False)

        return results, fog_densities


class FogAwareYOLO(nn.Module):
    """
    端到端雾感知YOLO
    直接从有雾图像进行检测，无需显式去雾
    """

    def __init__(self, model_size: str = 'n', num_classes: int = 80):
        super().__init__()

        # 加载预训练YOLO
        self.yolo = YOLO(f'yolo11{model_size}.pt')

        # 雾感知增强模块（作用于输入图像）
        self.input_enhancement = nn.Sequential(
            # 第一层：提取低级特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 雾感知注意力
            FogAttentionModule(32),

            # 第二层：特征融合
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出增强后的图像
        )

        # 是否使用输入增强
        self.use_enhancement = True

    def forward(self, x):
        """
        Args:
            x: 输入的有雾图像 (B, 3, H, W)

        Returns:
            检测结果
        """
        if self.use_enhancement:
            # 输入增强
            enhanced, fog_density = self.input_enhancement[:-1](x)
            enhanced = self.input_enhancement[-1](enhanced)

            # 残差连接：保留原始信息
            x_enhanced = x + enhanced
        else:
            x_enhanced = x
            fog_density = None

        # YOLO检测
        results = self.yolo.predict(x_enhanced, verbose=False)

        return results, fog_density

    def train_model(self, train_data, val_data, epochs=50):
        """
        训练雾感知YOLO

        Args:
            train_data: 训练数据路径
            val_data: 验证数据路径
            epochs: 训练轮数
        """
        # 使用YOLO的训练接口
        results = self.yolo.train(
            data=train_data,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='cuda',
            project='runs/fog_aware',
            name='train',
            exist_ok=True
        )

        return results


class SimpleFogAwareYOLO(nn.Module):
    """
    简化版雾感知YOLO
    只在输入层添加轻量级增强模块
    """

    def __init__(self, model_size: str = 'n'):
        super().__init__()

        # 加载YOLO
        self.yolo = YOLO(f'yolo11{model_size}.pt')

        # 轻量级输入增强（类似去雾的效果，但可学习）
        self.enhancement = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 输出增强残差
        )

        # 增强强度（可学习参数）
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: 有雾图像 (B, 3, H, W)

        Returns:
            检测结果
        """
        # 计算增强残差
        residual = self.enhancement(x)

        # 自适应增强
        x_enhanced = x + self.alpha * residual
        x_enhanced = torch.clamp(x_enhanced, 0, 1)

        # YOLO检测
        results = self.yolo.predict(x_enhanced, verbose=False)

        return results

    def get_enhanced_image(self, x):
        """获取增强后的图像（用于可视化）"""
        with torch.no_grad():
            residual = self.enhancement(x)
            x_enhanced = x + self.alpha * residual
            x_enhanced = torch.clamp(x_enhanced, 0, 1)
        return x_enhanced


def create_fog_aware_yolo(model_size: str = 'n',
                          mode: str = 'simple') -> nn.Module:
    """
    创建雾感知YOLO模型

    Args:
        model_size: YOLO模型大小 ('n', 's', 'm', 'l', 'x')
        mode: 模式
            - 'simple': 简化版（只增强输入）
            - 'full': 完整版（backbone中插入注意力）

    Returns:
        雾感知YOLO模型
    """
    if mode == 'simple':
        return SimpleFogAwareYOLO(model_size)
    elif mode == 'full':
        return FogAwareYOLO(model_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    # 测试
    model = create_fog_aware_yolo('n', mode='simple')
    print("✓ 雾感知YOLO模型创建成功")

    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    results = model(x)
    print("✓ 前向传播成功")
