"""
Dehazing Module
实现基于暗通道先验(Dark Channel Prior)的图像去雾算法
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class DehazingModule:
    """
    图像去雾模块
    使用暗通道先验(Dark Channel Prior)算法进行图像去雾
    
    参考论文: 
    He, K., Sun, J., & Tang, X. (2010). Single image haze removal using dark channel prior.
    IEEE transactions on pattern analysis and machine intelligence, 33(12), 2341-2353.
    """
    
    def __init__(self, omega: float = 0.95, t0: float = 0.1, radius: int = 15, eps: float = 0.001):
        """
        初始化去雾模块
        
        Args:
            omega: 去雾程度参数，保留少量雾以保持自然 (0-1)
            t0: 最小透射率阈值，防止除零
            radius: 导向滤波半径
            eps: 导向滤波正则化参数
        """
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.eps = eps
    
    def get_dark_channel(self, img: np.ndarray, size: int = 15) -> np.ndarray:
        """
        计算图像的暗通道
        
        Args:
            img: 输入图像 (H, W, C)
            size: 局部区域大小
            
        Returns:
            暗通道图像 (H, W)
        """
        # 对每个像素，取RGB三通道的最小值
        min_channel = np.min(img, axis=2)
        
        # 在局部区域内取最小值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def estimate_atmospheric_light(self, img: np.ndarray, dark_channel: np.ndarray, 
                                   percent: float = 0.001) -> np.ndarray:
        """
        估计大气光值
        
        Args:
            img: 输入图像
            dark_channel: 暗通道图像
            percent: 选取的最亮像素比例
            
        Returns:
            大气光值 (3,) RGB
        """
        h, w = dark_channel.shape
        num_pixels = int(h * w * percent)
        
        # 选取暗通道中最亮的像素
        dark_vec = dark_channel.reshape(h * w)
        img_vec = img.reshape(h * w, 3)
        
        # 获取前num_pixels个最亮的像素索引
        indices = np.argsort(dark_vec)[-num_pixels:]
        
        # 在这些像素中，选择原图像中最亮的作为大气光
        atmospheric_light = np.max(img_vec[indices], axis=0)
        
        return atmospheric_light
    
    def estimate_transmission(self, img: np.ndarray, atmospheric_light: np.ndarray, 
                             size: int = 15) -> np.ndarray:
        """
        估计透射率图
        
        Args:
            img: 输入图像
            atmospheric_light: 大气光值
            size: 局部区域大小
            
        Returns:
            透射率图 (H, W)
        """
        # 归一化图像
        normalized_img = img.astype(np.float64) / atmospheric_light
        
        # 计算归一化图像的暗通道
        dark_channel = self.get_dark_channel(normalized_img, size)
        
        # 估计透射率
        transmission = 1 - self.omega * dark_channel
        
        return transmission
    
    def guided_filter(self, guide: np.ndarray, src: np.ndarray, 
                     radius: int, eps: float) -> np.ndarray:
        """
        导向滤波，用于细化透射率图
        
        Args:
            guide: 导向图像
            src: 源图像
            radius: 滤波半径
            eps: 正则化参数
            
        Returns:
            滤波后的图像
        """
        mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        
        mean_guide_guide = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_guide = mean_guide_guide - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        return mean_a * guide + mean_b
    
    def recover_image(self, img: np.ndarray, transmission: np.ndarray, 
                     atmospheric_light: np.ndarray) -> np.ndarray:
        """
        根据透射率和大气光恢复无雾图像
        
        Args:
            img: 有雾图像
            transmission: 透射率图
            atmospheric_light: 大气光值
            
        Returns:
            去雾后的图像
        """
        # 确保透射率不小于t0
        transmission = np.maximum(transmission, self.t0)
        
        # 恢复图像
        recovered = np.empty_like(img, dtype=np.float64)
        for i in range(3):
            recovered[:, :, i] = (img[:, :, i].astype(np.float64) - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
        # 限制到[0, 255]
        recovered = np.clip(recovered, 0, 255)
        
        return recovered.astype(np.uint8)
    
    def dehaze(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        对图像进行去雾处理
        
        Args:
            img: 输入的有雾图像 (H, W, C) BGR格式
            
        Returns:
            去雾后的图像和中间结果字典
        """
        # 转换为float并归一化
        img_float = img.astype(np.float64)
        
        # 1. 计算暗通道
        dark_channel = self.get_dark_channel(img_float)
        
        # 2. 估计大气光
        atmospheric_light = self.estimate_atmospheric_light(img_float, dark_channel)
        
        # 3. 估计透射率
        transmission = self.estimate_transmission(img_float, atmospheric_light)
        
        # 4. 使用导向滤波细化透射率
        # 将图像转换为灰度图作为导向图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission_refined = self.guided_filter(gray, transmission, self.radius, self.eps)
        
        # 5. 恢复图像
        dehazed = self.recover_image(img_float, transmission_refined, atmospheric_light)
        
        # 返回结果和中间数据
        results = {
            'dark_channel': dark_channel,
            'atmospheric_light': atmospheric_light,
            'transmission': transmission,
            'transmission_refined': transmission_refined,
            'dehazed': dehazed
        }
        
        return dehazed, results
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        简化的处理接口，只返回去雾图像
        
        Args:
            img: 输入图像
            
        Returns:
            去雾后的图像
        """
        dehazed, _ = self.dehaze(img)
        return dehazed
