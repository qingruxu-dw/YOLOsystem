"""
Example script showing how to use the YOLOsystem components
"""

import cv2
import numpy as np
from pathlib import Path

# Example 1: Using the dehazing module independently
def example_dehazing():
    """演示独立使用去雾模块"""
    from yolosystem import DehazingModule
    
    # 创建去雾模块
    dehazer = DehazingModule(
        omega=0.95,    # 去雾程度
        t0=0.1,        # 最小透射率
        radius=15,     # 导向滤波半径
        eps=0.001      # 导向滤波正则化
    )
    
    # 读取有雾图像
    img = cv2.imread('input_image.jpg')
    
    # 方法1: 仅获取去雾结果
    dehazed = dehazer.process(img)
    cv2.imwrite('dehazed.jpg', dehazed)
    
    # 方法2: 获取去雾结果和中间数据
    dehazed, results = dehazer.dehaze(img)
    # results包含: dark_channel, atmospheric_light, transmission等
    cv2.imwrite('dark_channel.jpg', results['dark_channel'])
    cv2.imwrite('dehazed_detailed.jpg', results['dehazed'])


# Example 2: Using the detection module independently
def example_detection():
    """演示独立使用检测模块"""
    from yolosystem import YOLODetector
    
    # 创建检测器
    detector = YOLODetector(
        model_path='yolov8n.pt',  # 可以是 yolov8n, yolov8s, yolov8m等
        device='cpu'               # 或 'cuda' for GPU
    )
    
    # 读取图像
    img = cv2.imread('input_image.jpg')
    
    # 执行检测
    detections = detector.detect(
        img,
        conf_threshold=0.25,
        iou_threshold=0.45,
        classes=None,  # [0, 2, 3] 只检测特定类别，None为全部
        max_det=300
    )
    
    # 打印检测结果
    for det in detections:
        print(f"检测到: {det['class_name']}, 置信度: {det['confidence']:.2f}")
        print(f"位置: {det['bbox']}")
    
    # 绘制检测框
    result_img = detector.draw_detections(img, detections)
    cv2.imwrite('detection_result.jpg', result_img)
    
    # 获取模型信息
    info = detector.get_model_info()
    print(f"模型: {info['model_path']}")
    print(f"类别数: {info['num_classes']}")


# Example 3: Using the complete pipeline
def example_pipeline():
    """演示使用完整流水线"""
    from yolosystem import DehazingDetectionPipeline
    
    # 方法1: 使用默认配置
    pipeline = DehazingDetectionPipeline()
    
    # 方法2: 使用自定义配置
    # pipeline = DehazingDetectionPipeline(config_path='config.yaml')
    
    # 处理单张图像
    results = pipeline.process_image_file('input_image.jpg')
    
    # 访问处理结果
    original = results['original']
    dehazed = results['dehazed']  # 如果启用了去雾
    detections = results['detections']
    detection_img = results['detection_img']
    
    # 获取统计信息
    stats = pipeline.get_statistics(results)
    print(f"检测到 {stats['total_detections']} 个目标")
    print(f"各类别数量: {stats['class_counts']}")
    print(f"平均置信度: {stats['average_confidence']:.2f}")


# Example 4: Processing video
def example_video_processing():
    """演示视频处理"""
    from yolosystem import DehazingDetectionPipeline
    
    pipeline = DehazingDetectionPipeline(config_path='config.yaml')
    
    # 处理视频
    pipeline.process_video(
        video_path='input_video.mp4',
        output_path='output_video.mp4'
    )


# Example 5: Batch processing
def example_batch_processing():
    """演示批量处理"""
    from yolosystem import DehazingDetectionPipeline
    from pathlib import Path
    
    pipeline = DehazingDetectionPipeline()
    
    # 获取所有图像文件
    image_dir = Path('input_images')
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    # 批量处理
    all_stats = []
    for img_path in image_files:
        print(f"处理: {img_path.name}")
        results = pipeline.process_image_file(str(img_path))
        stats = pipeline.get_statistics(results)
        all_stats.append(stats)
    
    # 汇总统计
    total_detections = sum(s['total_detections'] for s in all_stats)
    print(f"\n总共检测到 {total_detections} 个目标")


# Example 6: Custom configuration
def example_custom_config():
    """演示自定义配置"""
    from yolosystem import DehazingDetectionPipeline
    import yaml
    
    # 创建自定义配置
    custom_config = {
        'dehazing': {
            'enabled': True,
            'omega': 0.9,      # 更强的去雾
            't0': 0.15,
            'radius': 20,
            'eps': 0.0001
        },
        'detection': {
            'model': 'yolov8s.pt',  # 使用更大的模型
            'conf_threshold': 0.3,   # 更高的置信度阈值
            'iou_threshold': 0.5,
            'max_det': 500,
            'classes': [0, 1, 2],    # 只检测人、自行车、汽车
            'device': 'cuda'         # 使用GPU
        },
        'pipeline': {
            'save_dehazed': True,
            'save_detections': True,
            'output_dir': 'custom_outputs'
        }
    }
    
    # 保存配置
    with open('custom_config.yaml', 'w') as f:
        yaml.dump(custom_config, f)
    
    # 使用自定义配置
    pipeline = DehazingDetectionPipeline(config_path='custom_config.yaml')
    results = pipeline.process_image_file('input_image.jpg')


# Example 7: Using utility functions
def example_utilities():
    """演示工具函数的使用"""
    from yolosystem.utils import (
        load_image, save_image, create_comparison_image,
        resize_image, calculate_psnr, calculate_ssim
    )
    
    # 加载图像
    img1 = load_image('image1.jpg')
    img2 = load_image('image2.jpg')
    
    # 创建对比图
    comparison = create_comparison_image(
        [img1, img2],
        ['原图', '处理后']
    )
    save_image(comparison, 'comparison.jpg')
    
    # 调整图像大小
    resized = resize_image(img1, max_width=800, max_height=600)
    
    # 计算图像质量指标
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


if __name__ == '__main__':
    print("请查看源代码中的各个示例函数")
    print("根据需要取消注释并运行相应的示例")
    
    # 取消注释以运行示例
    # example_dehazing()
    # example_detection()
    # example_pipeline()
    # example_video_processing()
    # example_batch_processing()
    # example_custom_config()
    # example_utilities()
