#!/usr/bin/env python3
"""
YOLOsystem Demo
演示去雾和目标检测系统的使用
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from yolosystem import DehazingDetectionPipeline, MultiModelDetectionPipeline, DehazingModule, YOLODetector
from yolosystem.utils import create_comparison_image, save_image


def create_demo_image():
    """
    创建一个示例有雾图像用于演示
    """
    # 创建一个简单的场景图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # 添加一些"物体"
    cv2.rectangle(img, (100, 150), (200, 350), (0, 0, 200), -1)  # 红色矩形
    cv2.rectangle(img, (300, 100), (450, 300), (200, 0, 0), -1)  # 蓝色矩形
    cv2.circle(img, (500, 300), 80, (0, 200, 0), -1)  # 绿色圆形
    
    # 添加雾效果（简单的白色叠加）
    haze = np.ones_like(img, dtype=np.uint8) * 180
    hazy_img = cv2.addWeighted(img, 0.5, haze, 0.5, 0)
    
    return hazy_img


def demo_dehazing_only():
    """
    演示去雾功能
    """
    print("=" * 50)
    print("演示1: 图像去雾功能")
    print("=" * 50)
    
    # 创建示例图像
    hazy_img = create_demo_image()
    
    # 创建去雾模块
    dehazer = DehazingModule(omega=0.95, t0=0.1, radius=15, eps=0.001)
    
    # 执行去雾
    print("正在进行去雾处理...")
    dehazed_img, results = dehazer.dehaze(hazy_img)
    
    # 创建对比图像
    comparison = create_comparison_image(
        [hazy_img, dehazed_img],
        ['有雾图像', '去雾图像']
    )
    
    # 保存结果
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    save_image(hazy_img, output_dir / 'demo_hazy.jpg')
    save_image(dehazed_img, output_dir / 'demo_dehazed.jpg')
    save_image(comparison, output_dir / 'demo_comparison.jpg')
    
    print(f"✓ 去雾完成！结果已保存到 {output_dir}")
    print(f"  - 有雾图像: demo_hazy.jpg")
    print(f"  - 去雾图像: demo_dehazed.jpg")
    print(f"  - 对比图像: demo_comparison.jpg")
    print()


def demo_full_pipeline():
    """
    演示完整的去雾+检测流程
    """
    print("=" * 50)
    print("演示2: 完整去雾+目标检测流程")
    print("=" * 50)
    
    # 创建示例图像
    hazy_img = create_demo_image()
    
    # 保存示例图像
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    save_image(hazy_img, output_dir / 'demo_input.jpg')
    
    print("正在初始化系统...")
    try:
        # 创建流水线（使用默认配置）
        pipeline = DehazingDetectionPipeline()
        
        print("正在处理图像...")
        # 处理图像
        results = pipeline.process_image(hazy_img, str(output_dir / 'demo_result.jpg'))
        
        # 获取统计信息
        stats = pipeline.get_statistics(results)
        
        print(f"✓ 处理完成！")
        print(f"\n检测统计:")
        print(f"  - 检测到的目标总数: {stats['total_detections']}")
        print(f"  - 各类别数量: {stats['class_counts']}")
        if stats['total_detections'] > 0:
            print(f"  - 平均置信度: {stats['average_confidence']:.2f}")
        
        print(f"\n结果已保存到 {output_dir}:")
        print(f"  - 输入图像: demo_input.jpg")
        if pipeline.config['pipeline']['save_dehazed']:
            print(f"  - 去雾图像: demo_result_dehazed.jpg")
        if pipeline.config['pipeline']['save_detections']:
            print(f"  - 检测结果: demo_result_detection.jpg")
        
    except ImportError as e:
        print(f"⚠ 警告: {e}")
        print("请安装 ultralytics: pip install ultralytics")
        print("或者使用 demo_dehazing_only 仅演示去雾功能")
    
    print()


def demo_multi_model():
    """
    演示多模型对比检测功能 (YOLOv8 vs YOLOv11)
    """
    print("=" * 50)
    print("演示3: 多模型对比检测 (YOLOv8 vs YOLOv11)")
    print("=" * 50)

    # 创建示例图像
    hazy_img = create_demo_image()

    # 保存示例图像
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    save_image(hazy_img, output_dir / 'demo_multi_input.jpg')

    print("正在初始化多模型系统...")
    try:
        # 创建多模型流水线（使用默认配置）
        pipeline = MultiModelDetectionPipeline()

        print("正在使用 YOLOv8 和 YOLOv11 同时检测...")
        # 处理图像
        results = pipeline.process_image(hazy_img, str(output_dir / 'demo_multi_result.jpg'))

        # 获取统计信息
        stats = pipeline.get_statistics(results)

        print(f"✓ 处理完成！")
        print(f"\n多模型检测统计对比:")

        for model_name, model_stats in stats.items():
            print(f"\n【{model_name}】")
            print(f"  - 检测到的目标总数: {model_stats['total_detections']}")
            print(f"  - 各类别数量: {model_stats['class_counts']}")
            if model_stats['total_detections'] > 0:
                print(f"  - 平均置信度: {model_stats['average_confidence']:.2f}")

        print(f"\n结果已保存到 {output_dir}:")
        print(f"  - 输入图像: demo_multi_input.jpg")
        if pipeline.config['pipeline']['save_dehazed']:
            print(f"  - 去雾图像: demo_multi_result_dehazed.jpg")
        if pipeline.config['pipeline']['save_detections']:
            for model_config in pipeline.config['multi_detection']['models']:
                model_name = model_config['name']
                print(f"  - {model_name}检测结果: demo_multi_result_detection_{model_name}.jpg")
        if pipeline.config['pipeline']['save_comparison']:
            print(f"  - 并排对比图像: demo_multi_result_comparison.jpg")

    except ImportError as e:
        print(f"⚠ 警告: {e}")
        print("请安装 ultralytics: pip install ultralytics")
        print("确保已安装 YOLOv8 和 YOLOv11 模型")

    print()


def process_custom_image_multi_model(image_path: str, config_path: str = None):
    """
    使用多模型处理自定义图像

    Args:
        image_path: 图像路径
        config_path: 配置文件路径
    """
    print("=" * 50)
    print(f"使用多模型处理图像: {image_path}")
    print("=" * 50)

    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return

    try:
        # 创建多模型流水线
        pipeline = MultiModelDetectionPipeline(config_path)

        print("正在使用多个模型同时处理...")
        # 处理图像
        results = pipeline.process_image_file(image_path)

        # 获取统计信息
        stats = pipeline.get_statistics(results)

        print(f"✓ 处理完成！")
        pipeline.print_statistics(stats)

        print(f"\n结果已保存到: {pipeline.output_dir}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def process_custom_image(image_path: str, config_path: str = None):
    """
    处理自定义图像
    
    Args:
        image_path: 图像路径
        config_path: 配置文件路径
    """
    print("=" * 50)
    print(f"处理图像: {image_path}")
    print("=" * 50)
    
    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    try:
        # 创建流水线
        pipeline = DehazingDetectionPipeline(config_path)
        
        print("正在处理...")
        # 处理图像
        results = pipeline.process_image_file(image_path)
        
        # 获取统计信息
        stats = pipeline.get_statistics(results)
        
        print(f"✓ 处理完成！")
        print(f"\n检测统计:")
        print(f"  - 检测到的目标总数: {stats['total_detections']}")
        print(f"  - 各类别数量: {stats['class_counts']}")
        if stats['total_detections'] > 0:
            print(f"  - 平均置信度: {stats['average_confidence']:.2f}")
        
        print(f"\n结果已保存到: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOsystem - 去雾和目标检测系统演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有演示
  python demo.py

  # 仅运行去雾演示
  python demo.py --demo dehazing

  # 运行完整流程演示
  python demo.py --demo full

  # 运行多模型对比演示
  python demo.py --demo multi

  # 处理自定义图像（单模型）
  python demo.py --image path/to/image.jpg

  # 处理自定义图像（多模型对比）
  python demo.py --image path/to/image.jpg --multi-model

  # 使用自定义配置处理图像
  python demo.py --image path/to/image.jpg --config config.yaml
        """
    )

    parser.add_argument(
        '--demo',
        type=str,
        choices=['dehazing', 'full', 'multi', 'all'],
        default='all',
        help='选择要运行的演示 (默认: all)'
    )

    parser.add_argument(
        '--image',
        type=str,
        help='要处理的图像路径'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径 (可选)'
    )

    parser.add_argument(
        '--multi-model',
        action='store_true',
        help='使用多模型对比模式处理图像'
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("YOLOsystem - 去雾和目标检测系统")
    print("Dehazing and Object Detection System")
    print("=" * 50 + "\n")

    # 如果指定了图像，处理该图像
    if args.image:
        if args.multi_model:
            process_custom_image_multi_model(args.image, args.config)
        else:
            process_custom_image(args.image, args.config)
        return

    # 否则运行演示
    if args.demo in ['dehazing', 'all']:
        demo_dehazing_only()

    if args.demo in ['full', 'all']:
        demo_full_pipeline()

    if args.demo in ['multi', 'all']:
        demo_multi_model()

    print("=" * 50)
    print("演示完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
