"""
批量测试融合检测效果
用于评估在真实有雾图像数据集上的表现
"""

import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from yolosystem import FusionDetectionPipeline


def batch_test(image_dir: str, output_dir: str, strategy: str = 'adaptive', pattern: str = '*'):
    """
    批量测试融合检测效果

    Args:
        image_dir: 输入图像目录
        output_dir: 输出目录
        strategy: 融合策略
        pattern: 文件名模式（如 '*beta_0.01.png' 用于Foggy Cityscapes）
    """
    # 初始化流水线
    pipeline = FusionDetectionPipeline()
    pipeline.config['fusion']['strategy'] = strategy
    pipeline.config['pipeline']['output_dir'] = output_dir
    pipeline.output_dir = Path(output_dir)
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)

    # 重新初始化融合检测器
    from yolosystem.fusion import FusionDetector
    pipeline.fusion_detector = FusionDetector(
        detector=pipeline.detector,
        dehazer=pipeline.dehazer,
        fusion_strategy=strategy,
        iou_threshold=pipeline.config['fusion']['iou_threshold']
    )

    # 获取所有图像文件
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    # 支持文件名模式匹配（如 *beta_0.01.png）
    for ext in image_extensions:
        # 如果pattern不包含扩展名，添加扩展名
        if not pattern.endswith(ext) and not pattern.endswith(ext.upper()):
            search_pattern = f"{pattern}{ext}"
        else:
            search_pattern = pattern

        # 递归搜索（支持多层目录结构，如Foggy Cityscapes）
        image_files.extend(list(image_dir.rglob(search_pattern)))
        if ext.upper() != ext:
            search_pattern_upper = search_pattern.replace(ext, ext.upper())
            image_files.extend(list(image_dir.rglob(search_pattern_upper)))

    # 去重并排序
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"错误: 在 {image_dir} 中没有找到匹配 '{pattern}' 的图像文件")
        print(f"提示: 请检查目录路径和文件名模式")
        print(f"示例: --pattern '*beta_0.01.png' (Foggy Cityscapes)")
        return

    print(f"找到 {len(image_files)} 张图像 (模式: {pattern})")
    print(f"融合策略: {strategy}")
    print(f"输出目录: {output_dir}")
    print("="*80)

    # 存储所有统计信息
    all_stats = []
    summary = {
        'total_images': len(image_files),
        'strategy': strategy,
        'pattern': pattern,
        'avg_quality_improvement': {
            'sharpness': [],
            'contrast': [],
            'entropy': []
        },
        'total_detections': {
            'original': 0,
            'dehazed': 0,
            'fused': 0
        },
        'avg_confidence': {
            'original': [],
            'dehazed': [],
            'fused': []
        },
        'source_distribution': {
            'original': 0,
            'dehazed': 0
        }
    }

    # 批量处理
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # 读取图像
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"警告: 无法读取 {image_file}")
                continue

            # 准备保存路径
            save_path = pipeline.output_dir / f"{image_file.stem}.jpg"

            # 处理图像
            results = pipeline.process_image(img, str(save_path))

            # 获取统计信息
            stats = pipeline.get_statistics(results)
            stats['image_name'] = image_file.name
            all_stats.append(stats)

            # 累计统计
            summary['avg_quality_improvement']['sharpness'].append(
                stats['fusion_info']['quality_improvement']['sharpness']
            )
            summary['avg_quality_improvement']['contrast'].append(
                stats['fusion_info']['quality_improvement']['contrast']
            )
            summary['avg_quality_improvement']['entropy'].append(
                stats['fusion_info']['quality_improvement']['entropy']
            )

            summary['total_detections']['original'] += stats['original']['total_detections']
            summary['total_detections']['dehazed'] += stats['dehazed']['total_detections']
            summary['total_detections']['fused'] += stats['fused']['total_detections']

            if stats['original']['average_confidence'] > 0:
                summary['avg_confidence']['original'].append(stats['original']['average_confidence'])
            if stats['dehazed']['average_confidence'] > 0:
                summary['avg_confidence']['dehazed'].append(stats['dehazed']['average_confidence'])
            if stats['fused']['average_confidence'] > 0:
                summary['avg_confidence']['fused'].append(stats['fused']['average_confidence'])

            # 统计来源分布
            if 'source_distribution' in stats['fusion_info']:
                for source, count in stats['fusion_info']['source_distribution'].items():
                    summary['source_distribution'][source] = \
                        summary['source_distribution'].get(source, 0) + count

        except Exception as e:
            print(f"错误: 处理 {image_file} 时出错: {e}")
            continue

    # 计算平均值
    for metric in ['sharpness', 'contrast', 'entropy']:
        if summary['avg_quality_improvement'][metric]:
            summary['avg_quality_improvement'][metric] = np.mean(
                summary['avg_quality_improvement'][metric]
            )
        else:
            summary['avg_quality_improvement'][metric] = 0.0

    for phase in ['original', 'dehazed', 'fused']:
        if summary['avg_confidence'][phase]:
            summary['avg_confidence'][phase] = np.mean(summary['avg_confidence'][phase])
        else:
            summary['avg_confidence'][phase] = 0.0

    # 保存详细统计
    stats_file = Path(output_dir) / 'detailed_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # 保存汇总统计
    summary_file = Path(output_dir) / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总报告
    print("\n" + "="*80)
    print("批量测试汇总报告")
    print("="*80)

    print(f"\n处理图像数量: {len(all_stats)}/{summary['total_images']}")
    print(f"融合策略: {strategy}")
    if pattern != '*':
        print(f"文件名模式: {pattern}")

    print("\n【平均质量改善】")
    for metric, value in summary['avg_quality_improvement'].items():
        print(f"  {metric}: {value:.3f}x")

    print("\n【检测数量统计】")
    print(f"  原图总检测数: {summary['total_detections']['original']}")
    print(f"  去雾图总检测数: {summary['total_detections']['dehazed']}")
    print(f"  融合后总检测数: {summary['total_detections']['fused']}")

    print("\n【平均置信度】")
    print(f"  原图: {summary['avg_confidence']['original']:.3f}")
    print(f"  去雾图: {summary['avg_confidence']['dehazed']:.3f}")
    print(f"  融合后: {summary['avg_confidence']['fused']:.3f}")

    print("\n【融合来源分布】")
    total_source = sum(summary['source_distribution'].values())
    if total_source > 0:
        for source, count in summary['source_distribution'].items():
            percentage = (count / total_source) * 100
            print(f"  来自{source}: {count} ({percentage:.1f}%)")

    # 融合效果评估
    print("\n【融合效果评估】")
    avg_improvement = np.mean([
        summary['avg_quality_improvement']['sharpness'],
        summary['avg_quality_improvement']['contrast'],
        summary['avg_quality_improvement']['entropy']
    ])

    if avg_improvement > 1.2:
        print("  ✅ 去雾显著改善了图像质量")
        improvement_level = "显著"
    elif avg_improvement > 1.0:
        print("  ✅ 去雾适度改善了图像质量")
        improvement_level = "适度"
    elif avg_improvement > 0.9:
        print("  ⚠️  去雾对图像质量改善有限")
        improvement_level = "有限"
    else:
        print("  ❌ 去雾可能降低了图像质量")
        improvement_level = "负面"

    # 检测性能评估
    detection_improvement = (
        (summary['total_detections']['fused'] - summary['total_detections']['original']) /
        max(summary['total_detections']['original'], 1) * 100
    )

    if detection_improvement > 10:
        print(f"  ✅ 融合检测数量提升 {detection_improvement:.1f}%")
    elif detection_improvement > 0:
        print(f"  ✅ 融合检测数量略有提升 {detection_improvement:.1f}%")
    elif detection_improvement > -10:
        print(f"  ⚠️  融合检测数量略有下降 {detection_improvement:.1f}%")
    else:
        print(f"  ❌ 融合检测数量明显下降 {detection_improvement:.1f}%")

    # 置信度评估
    conf_improvement = (
        (summary['avg_confidence']['fused'] - summary['avg_confidence']['original']) /
        max(summary['avg_confidence']['original'], 0.01) * 100
    )

    if conf_improvement > 5:
        print(f"  ✅ 平均置信度提升 {conf_improvement:.1f}%")
    elif conf_improvement > -5:
        print(f"  ➖ 平均置信度基本持平")
    else:
        print(f"  ❌ 平均置信度下降 {abs(conf_improvement):.1f}%")

    print("\n【结论】")
    if improvement_level in ["显著", "适度"] and detection_improvement > 0:
        print("  ✅ 融合策略有效，建议在实际应用中使用")
    elif improvement_level in ["显著", "适度"]:
        print("  ⚠️  图像质量改善，但检测数量变化不明显")
    else:
        print("  ⚠️  建议调整去雾参数或尝试其他融合策略")

    print("\n" + "="*80)
    print(f"详细统计已保存到: {stats_file}")
    print(f"汇总报告已保存到: {summary_file}")
    print(f"结果图像已保存到: {output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='批量测试融合检测效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试普通数据集
  python batch_test_fusion.py --input-dir datasets/DAWN/images/fog

  # 测试Foggy Cityscapes（中雾）
  python batch_test_fusion.py --input-dir datasets/foggy_cityscapes/leftImg8bit_foggy/val --pattern "*beta_0.01.png"

  # 测试Foggy Cityscapes（浓雾）
  python batch_test_fusion.py --input-dir datasets/foggy_cityscapes/leftImg8bit_foggy/val --pattern "*beta_0.02.png"
        """
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output-dir', type=str, default='outputs/batch_test',
                       help='输出目录')
    parser.add_argument('--strategy', type=str, default='adaptive',
                       choices=['adaptive', 'confidence', 'quality', 'both'],
                       help='融合策略')
    parser.add_argument('--pattern', type=str, default='*',
                       help='文件名模式（如 "*beta_0.01.png" 用于Foggy Cityscapes特定雾浓度）')

    args = parser.parse_args()

    # 检查输入目录
    if not Path(args.input_dir).exists():
        print(f"错误: 输入目录不存在: {args.input_dir}")
        print("\n提示: 请下载有雾图像数据集")
        print("推荐数据集（包含车辆和行人）:")
        print("  1. Foggy Cityscapes: https://www.cityscapes-dataset.com/")
        print("     (最推荐，城市道路场景，包含大量车辆和行人)")
        print("  2. DAWN: https://github.com/Yvette01/DAWN-Dataset")
        print("     (真实拍摄，快速下载)")
        print("  3. ACDC: https://acdc.vision.ee.ethz.ch/")
        print("     (多种恶劣天气条件)")
        print("\n详细说明请查看: ROAD_DATASET_GUIDE.md")
        return

    batch_test(args.input_dir, args.output_dir, args.strategy, args.pattern)


if __name__ == '__main__':
    main()
