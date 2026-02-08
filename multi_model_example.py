#!/usr/bin/env python3
"""
多模型检测使用示例
演示如何使用 YOLOv8 和 YOLOv11 同时进行目标检测并对比结果
"""

import cv2
from pathlib import Path
from yolosystem import MultiModelDetectionPipeline, MultiModelDetector


def example_1_basic_usage():
    """
    示例1: 基本用法 - 使用默认配置
    """
    print("\n" + "="*60)
    print("示例1: 基本用法 - 使用默认配置")
    print("="*60)

    # 创建多模型流水线（会自动加载 YOLOv8 和 YOLOv11）
    pipeline = MultiModelDetectionPipeline()

    # 处理图像文件
    results = pipeline.process_image_file('your_image.jpg')

    # 获取统计信息
    stats = pipeline.get_statistics(results)

    # 打印统计信息
    pipeline.print_statistics(stats)

    print("\n结果文件已保存到 outputs/ 目录")
    print("- 各模型独立检测结果")
    print("- 并排对比图像")


def example_2_custom_config():
    """
    示例2: 使用自定义配置文件
    """
    print("\n" + "="*60)
    print("示例2: 使用自定义配置文件")
    print("="*60)

    # 使用自定义配置文件
    pipeline = MultiModelDetectionPipeline(config_path='config.yaml')

    # 处理图像
    results = pipeline.process_image_file('your_image.jpg')

    # 打印各模型的检测统计
    stats = pipeline.get_statistics(results)
    pipeline.print_statistics(stats)


def example_3_process_numpy_array():
    """
    示例3: 直接处理 numpy 数组（适用于视频处理）
    """
    print("\n" + "="*60)
    print("示例3: 处理 numpy 数组")
    print("="*60)

    # 读取图像
    img = cv2.imread('your_image.jpg')

    # 创建流水线
    pipeline = MultiModelDetectionPipeline()

    # 处理图像数组
    results = pipeline.process_image(img, save_path='outputs/result.jpg')

    # 访问各模型的检测结果
    for model_name, detections in results['all_detections'].items():
        print(f"\n{model_name} 检测到 {len(detections)} 个目标:")
        for det in detections[:5]:  # 只显示前5个
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")


def example_4_multi_detector_only():
    """
    示例4: 仅使用多模型检测器（不含去雾）
    """
    print("\n" + "="*60)
    print("示例4: 仅使用多模型检测器")
    print("="*60)

    # 配置多个模型
    model_configs = [
        {"name": "YOLOv8-nano", "model_path": "yolov8n.pt"},
        {"name": "YOLOv11-nano", "model_path": "yolo11n.pt"},
        # 可以添加更多模型
        # {"name": "YOLOv8-small", "model_path": "yolov8s.pt"},
    ]

    # 创建多模型检测器
    detector = MultiModelDetector(model_configs, device='cpu')

    # 读取图像
    img = cv2.imread('your_image.jpg')

    # 使用所有模型检测
    all_detections = detector.detect_all(
        img,
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    # 获取统计对比
    stats = detector.get_comparison_statistics(all_detections)

    # 打印结果
    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        print(f"  总检测数: {model_stats['total_detections']}")
        print(f"  平均置信度: {model_stats['average_confidence']:.3f}")
        print(f"  类别分布: {model_stats['class_counts']}")

    # 绘制检测结果
    detection_images = detector.draw_detections_comparison(img, all_detections)

    # 创建并排对比图
    comparison = detector.create_side_by_side_comparison(detection_images)

    # 保存结果
    cv2.imwrite('outputs/comparison.jpg', comparison)
    print("\n对比图已保存到 outputs/comparison.jpg")


def example_5_custom_colors():
    """
    示例5: 自定义检测框颜色
    """
    print("\n" + "="*60)
    print("示例5: 自定义检测框颜色")
    print("="*60)

    model_configs = [
        {"name": "YOLOv8", "model_path": "yolov8n.pt"},
        {"name": "YOLOv11", "model_path": "yolo11n.pt"},
    ]

    detector = MultiModelDetector(model_configs, device='cpu')

    img = cv2.imread('your_image.jpg')

    # 检测
    all_detections = detector.detect_all(img, conf_threshold=0.3)

    # 自定义颜色
    colors = {
        "YOLOv8": (0, 255, 0),    # 绿色
        "YOLOv11": (255, 0, 0),   # 蓝色
    }

    # 绘制（使用自定义颜色）
    detection_images = detector.draw_detections_comparison(
        img,
        all_detections,
        colors=colors,
        thickness=3
    )

    # 保存各模型的检测结果
    for model_name, detection_img in detection_images.items():
        output_path = f'outputs/{model_name}_detection.jpg'
        cv2.imwrite(output_path, detection_img)
        print(f"{model_name} 检测结果已保存: {output_path}")


def example_6_comparison_analysis():
    """
    示例6: 详细对比分析
    """
    print("\n" + "="*60)
    print("示例6: 详细对比分析")
    print("="*60)

    pipeline = MultiModelDetectionPipeline()

    # 处理图像
    results = pipeline.process_image_file('your_image.jpg')

    # 详细分析各模型的检测结果
    all_detections = results['all_detections']

    print("\n详细对比分析:")
    print("-" * 60)

    for model_name, detections in all_detections.items():
        print(f"\n{model_name}:")
        print(f"  检测总数: {len(detections)}")

        if detections:
            # 按类别统计
            class_stats = {}
            for det in detections:
                cls = det['class_name']
                if cls not in class_stats:
                    class_stats[cls] = {
                        'count': 0,
                        'confidences': []
                    }
                class_stats[cls]['count'] += 1
                class_stats[cls]['confidences'].append(det['confidence'])

            # 打印各类别统计
            for cls, stats in class_stats.items():
                avg_conf = sum(stats['confidences']) / len(stats['confidences'])
                print(f"  {cls}:")
                print(f"    数量: {stats['count']}")
                print(f"    平均置信度: {avg_conf:.3f}")
                print(f"    置信度范围: {min(stats['confidences']):.3f} - {max(stats['confidences']):.3f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("YOLOv8 vs YOLOv11 多模型检测示例")
    print("Multi-Model Detection Examples")
    print("="*60)

    # 确保输出目录存在
    Path('outputs').mkdir(exist_ok=True)

    # 运行你想要的示例
    print("\n选择要运行的示例:")
    print("1. 基本用法")
    print("2. 使用自定义配置")
    print("3. 处理 numpy 数组")
    print("4. 仅使用多模型检测器")
    print("5. 自定义检测框颜色")
    print("6. 详细对比分析")

    # 这里你可以取消注释你想运行的示例
    # example_1_basic_usage()
    # example_2_custom_config()
    # example_3_process_numpy_array()
    # example_4_multi_detector_only()
    # example_5_custom_colors()
    # example_6_comparison_analysis()

    print("\n提示: 取消注释相应的函数调用来运行示例")
    print("注意: 请将 'your_image.jpg' 替换为实际的图像路径")
