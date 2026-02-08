"""
融合检测示例脚本
演示如何使用双路检测+智能融合策略
"""

import cv2
import argparse
from pathlib import Path
from yolosystem import FusionDetectionPipeline


def main():
    parser = argparse.ArgumentParser(description='融合检测示例')
    parser.add_argument('--image', type=str, default='test_images/hazy_road2.jpg',
                       help='输入图像路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    parser.add_argument('--strategy', type=str, default='adaptive',
                       choices=['adaptive', 'confidence', 'quality', 'both'],
                       help='融合策略: adaptive(自适应), confidence(置信度), quality(质量), both(保留两者)')
    parser.add_argument('--output-dir', type=str, default='outputs/fusion',
                       help='输出目录')
    parser.add_argument('--show', action='store_true',
                       help='是否显示结果')

    args = parser.parse_args()

    print("="*80)
    print("双路检测+智能融合 演示")
    print("="*80)
    print(f"输入图像: {args.image}")
    print(f"融合策略: {args.strategy}")
    print(f"输出目录: {args.output_dir}")
    print("="*80 + "\n")

    # 初始化融合检测流水线
    if args.config:
        pipeline = FusionDetectionPipeline(config_path=args.config)
    else:
        # 使用默认配置，但覆盖部分参数
        pipeline = FusionDetectionPipeline()
        pipeline.config['fusion']['strategy'] = args.strategy
        pipeline.config['pipeline']['output_dir'] = args.output_dir
        pipeline.config['pipeline']['show_results'] = args.show
        pipeline.output_dir = Path(args.output_dir)
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)

        # 重新初始化融合检测器以应用新策略
        from yolosystem.fusion import FusionDetector
        pipeline.fusion_detector = FusionDetector(
            detector=pipeline.detector,
            dehazer=pipeline.dehazer,
            fusion_strategy=args.strategy,
            iou_threshold=pipeline.config['fusion']['iou_threshold']
        )

    # 检查输入图像是否存在
    if not Path(args.image).exists():
        print(f"错误: 找不到图像文件 {args.image}")
        print("提示: 请提供有效的图像路径，或使用 --image 参数指定图像")
        return

    # 处理图像
    print("正在处理图像...")
    results = pipeline.process_image_file(args.image)

    # 获取并打印统计信息
    stats = pipeline.get_statistics(results)
    pipeline.print_statistics(stats)

    # 打印保存的文件
    image_name = Path(args.image).stem
    print(f"\n结果已保存到: {args.output_dir}/")
    print(f"  - {image_name}_original_detection.jpg (原图检测)")
    print(f"  - {image_name}_dehazed_detection.jpg (去雾图检测)")
    print(f"  - {image_name}_fused_detection.jpg (融合检测)")
    print(f"  - {image_name}_comparison.jpg (三路对比)")

    # 分析融合效果
    print("\n" + "="*80)
    print("融合效果分析")
    print("="*80)

    original_count = stats['original']['total_detections']
    dehazed_count = stats['dehazed']['total_detections']
    fused_count = stats['fused']['total_detections']

    print(f"原图检测数量: {original_count}")
    print(f"去雾图检测数量: {dehazed_count}")
    print(f"融合后检测数量: {fused_count}")

    if 'source_distribution' in stats['fusion_info']:
        source_dist = stats['fusion_info']['source_distribution']
        print(f"\n融合结果来源分析:")
        for source, count in source_dist.items():
            percentage = (count / fused_count * 100) if fused_count > 0 else 0
            print(f"  来自{source}: {count} ({percentage:.1f}%)")

    # 质量改善分析
    quality_imp = stats['fusion_info']['quality_improvement']
    print(f"\n去雾质量改善:")
    print(f"  清晰度提升: {quality_imp['sharpness']:.2f}x")
    print(f"  对比度提升: {quality_imp['contrast']:.2f}x")
    print(f"  信息熵提升: {quality_imp['entropy']:.2f}x")

    # 给出建议
    print(f"\n建议:")
    avg_improvement = sum(quality_imp.values()) / len(quality_imp)
    if avg_improvement > 1.2:
        print("  ✓ 去雾显著改善了图像质量，建议继续使用融合策略")
    elif avg_improvement > 0.9:
        print("  ✓ 去雾对图像质量有一定改善，融合策略是合理的选择")
    else:
        print("  ⚠ 去雾可能降低了图像质量，建议调整去雾参数或直接使用原图")

    if args.strategy == 'adaptive':
        print("  ✓ 当前使用自适应策略，系统会自动根据质量和置信度选择最佳结果")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
