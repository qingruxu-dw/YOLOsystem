"""
Utility functions for the YOLOsystem
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像数组 (BGR格式)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return img


def save_image(img: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    保存图像到文件
    
    Args:
        img: 图像数组
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def create_comparison_image(images: list, titles: list = None) -> np.ndarray:
    """
    创建对比图像
    
    Args:
        images: 图像列表
        titles: 标题列表
        
    Returns:
        拼接后的对比图像
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    # 确保所有图像高度一致
    max_height = max(img.shape[0] for img in images)
    resized_images = []
    
    for img in images:
        if img.shape[0] != max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
        resized_images.append(img)
    
    # 水平拼接
    comparison = np.hstack(resized_images)
    
    # 添加标题（如果提供）
    if titles and len(titles) == len(images):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 255, 255)
        
        x_offset = 0
        for i, (img, title) in enumerate(zip(resized_images, titles)):
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                title, font, font_scale, font_thickness
            )
            
            # 在图像顶部居中绘制标题
            x = x_offset + (img.shape[1] - text_width) // 2
            y = 40
            
            # 绘制文本背景
            cv2.rectangle(
                comparison,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + baseline + 5),
                (0, 0, 0),
                -1
            )
            
            # 绘制文本
            cv2.putText(comparison, title, (x, y), font, font_scale, color, font_thickness)
            
            x_offset += img.shape[1]
    
    return comparison


def resize_image(img: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """
    调整图像大小，保持宽高比
    
    Args:
        img: 输入图像
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        调整后的图像
    """
    h, w = img.shape[:2]
    
    # 计算缩放比例
    scale = min(max_width / w, max_height / h)
    
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    return img


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两张图像之间的PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        PSNR值
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两张图像之间的SSIM (Structural Similarity Index)
    简化版本实现
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        SSIM值
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))
