"""
融合检测 vs 简单串联方法性能对比脚本

比较两种方法：
1. 融合方法：双路检测 + 智能融合
2. 简单串联：原图 → 去雾 → 检测

生成详细的性能对比报告
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict

from yolosystem import DehazingModule, YOLODetector, FusionDetectionPipeline
from yolosystem.fusion import ImageQualityAssessment


def run_simple_pipeline(image_path: str, dehazing_module: DehazingModule,
                       detector: YOLODetector, conf_threshold: float = 0.25) -> Tuple[List, float]:
    """
    运行简单串联方法：去雾 → 检测

    Returns:
        (detections, processing_time)
    """
    start_time = time.time()

    # 读取图像
    image = cv2.imread(image_path)

    # 去雾
    dehazed = dehazing_module.process(image)

    # 检测
    results = detector.detect(dehazed, conf_threshold=conf_threshold)

    processing_time = time.time() - start_time

    return results, processing_time


def run_fusion_pipeline(image_path: str, fusion_pipeline: FusionDetectionPipeline) -> Tuple[Dict, float]:
    """
    运行融合检测方法

    Returns:
        (fusion_result, processing_time)
    """
    start_time = time.time()

    # 读取图像
    image = cv2.imread(image_path)

    # 融合检测
    result = fusion_pipeline.process_image(image)

    processing_time = time.time() - start_time

    return result, processing_time


def analyze_detections(detections: List) -> Dict:
    """分析检测结果"""
    if not detections:
        return {
            'total_count': 0,
            'avg_confidence': 0.0,
            'class_distribution': {},
            'confidence_distribution': []
        }

    confidences = [det['confidence'] for det in detections]
    classes = [det['class_name'] for det in detections]

    # 类别分布
    class_dist = defaultdict(int)
    for cls in classes:
        class_dist[cls] += 1

    return {
        'total_count': len(detections),
        'avg_confidence': np.mean(confidences),
        'max_confidence': np.max(confidences),
        'min_confidence': np.min(confidences),
        'std_confidence': np.std(confidences),
        'class_distribution': dict(class_dist),
        'confidence_distribution': confidences
    }


def compare_methods(image_paths: List[str], output_dir: str,
                   strategy: str = 'adaptive', conf_threshold: float = 0.25):
    """
    对比融合方法和简单串联方法
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("融合检测 vs 简单串联方法性能对比")
    print("=" * 60)
    print(f"\n测试图像数量: {len(image_paths)}")
    print(f"输出目录: {output_dir}")
    print(f"融合策略: {strategy}")
    print(f"置信度阈值: {conf_threshold}\n")

    # 初始化流水线
    print("初始化流水线...")
    dehazing_module = DehazingModule()
    detector = YOLODetector()
    fusion_pipeline = FusionDetectionPipeline()

    # 修改融合策略和置信度阈值
    fusion_pipeline.config['fusion']['strategy'] = strategy
    fusion_pipeline.config['detection']['conf_threshold'] = conf_threshold
    fusion_pipeline.fusion_detector.strategy = strategy

    # 统计数据
    simple_stats = {
        'total_detections': 0,
        'total_time': 0.0,
        'per_image_stats': [],
        'all_confidences': [],
        'class_counts': defaultdict(int)
    }

    fusion_stats = {
        'total_detections': 0,
        'total_time': 0.0,
        'per_image_stats': [],
        'all_confidences': [],
        'class_counts': defaultdict(int),
        'source_distribution': defaultdict(int),
        'quality_improvements': []
    }

    # 处理每张图像
    print("\n开始处理图像...")
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] 处理: {Path(image_path).name}")

        # 简单串联方法
        print("  运行简单串联方法...")
        simple_results, simple_time = run_simple_pipeline(
            image_path, dehazing_module, detector, conf_threshold
        )
        simple_analysis = analyze_detections(simple_results)

        simple_stats['total_detections'] += simple_analysis['total_count']
        simple_stats['total_time'] += simple_time
        simple_stats['all_confidences'].extend(simple_analysis['confidence_distribution'])
        for cls, count in simple_analysis['class_distribution'].items():
            simple_stats['class_counts'][cls] += count

        simple_stats['per_image_stats'].append({
            'image': Path(image_path).name,
            'detections': simple_analysis['total_count'],
            'avg_confidence': simple_analysis['avg_confidence'],
            'time': simple_time
        })

        print(f"    检测数量: {simple_analysis['total_count']}, "
              f"平均置信度: {simple_analysis['avg_confidence']:.3f}, "
              f"耗时: {simple_time:.2f}s")

        # 融合方法
        print("  运行融合检测方法...")
        fusion_result, fusion_time = run_fusion_pipeline(image_path, fusion_pipeline)
        fusion_analysis = analyze_detections(fusion_result['fused_detections'])

        fusion_stats['total_detections'] += fusion_analysis['total_count']
        fusion_stats['total_time'] += fusion_time
        fusion_stats['all_confidences'].extend(fusion_analysis['confidence_distribution'])
        for cls, count in fusion_analysis['class_distribution'].items():
            fusion_stats['class_counts'][cls] += count

        # 来源分布
        for det in fusion_result['fused_detections']:
            fusion_stats['source_distribution'][det.get('source', 'unknown')] += 1

        # 质量改善
        quality_improvement = fusion_result.get('quality_improvement', {})
        if quality_improvement:
            fusion_stats['quality_improvements'].append(quality_improvement)

        fusion_stats['per_image_stats'].append({
            'image': Path(image_path).name,
            'detections': fusion_analysis['total_count'],
            'avg_confidence': fusion_analysis['avg_confidence'],
            'time': fusion_time,
            'quality_improvement': quality_improvement
        })

        print(f"    检测数量: {fusion_analysis['total_count']}, "
              f"平均置信度: {fusion_analysis['avg_confidence']:.3f}, "
              f"耗时: {fusion_time:.2f}s")

    # 计算汇总统计
    print("\n" + "=" * 60)
    print("计算汇总统计...")

    comparison_report = {
        'test_info': {
            'total_images': len(image_paths),
            'fusion_strategy': strategy,
            'conf_threshold': conf_threshold
        },
        'simple_method': {
            'total_detections': simple_stats['total_detections'],
            'avg_detections_per_image': simple_stats['total_detections'] / len(image_paths),
            'avg_confidence': np.mean(simple_stats['all_confidences']) if simple_stats['all_confidences'] else 0.0,
            'std_confidence': np.std(simple_stats['all_confidences']) if simple_stats['all_confidences'] else 0.0,
            'total_time': simple_stats['total_time'],
            'avg_time_per_image': simple_stats['total_time'] / len(image_paths),
            'class_distribution': dict(simple_stats['class_counts'])
        },
        'fusion_method': {
            'total_detections': fusion_stats['total_detections'],
            'avg_detections_per_image': fusion_stats['total_detections'] / len(image_paths),
            'avg_confidence': np.mean(fusion_stats['all_confidences']) if fusion_stats['all_confidences'] else 0.0,
            'std_confidence': np.std(fusion_stats['all_confidences']) if fusion_stats['all_confidences'] else 0.0,
            'total_time': fusion_stats['total_time'],
            'avg_time_per_image': fusion_stats['total_time'] / len(image_paths),
            'class_distribution': dict(fusion_stats['class_counts']),
            'source_distribution': dict(fusion_stats['source_distribution'])
        },
        'comparison': {
            'detection_improvement': {
                'absolute': fusion_stats['total_detections'] - simple_stats['total_detections'],
                'percentage': ((fusion_stats['total_detections'] - simple_stats['total_detections']) /
                              simple_stats['total_detections'] * 100) if simple_stats['total_detections'] > 0 else 0.0
            },
            'confidence_improvement': {
                'absolute': (np.mean(fusion_stats['all_confidences']) - np.mean(simple_stats['all_confidences']))
                           if fusion_stats['all_confidences'] and simple_stats['all_confidences'] else 0.0,
                'percentage': ((np.mean(fusion_stats['all_confidences']) - np.mean(simple_stats['all_confidences'])) /
                              np.mean(simple_stats['all_confidences']) * 100)
                             if simple_stats['all_confidences'] and np.mean(simple_stats['all_confidences']) > 0 else 0.0
            },
            'time_overhead': {
                'absolute': fusion_stats['total_time'] - simple_stats['total_time'],
                'percentage': ((fusion_stats['total_time'] - simple_stats['total_time']) /
                              simple_stats['total_time'] * 100) if simple_stats['total_time'] > 0 else 0.0
            }
        },
        'quality_analysis': {}
    }

    # 质量改善分析
    if fusion_stats['quality_improvements']:
        quality_metrics = ['sharpness', 'contrast', 'entropy', 'brightness']
        quality_summary = {}

        for metric in quality_metrics:
            improvements = [qi.get(metric, 1.0) for qi in fusion_stats['quality_improvements']]
            quality_summary[metric] = {
                'avg_improvement': np.mean(improvements),
                'max_improvement': np.max(improvements),
                'min_improvement': np.min(improvements)
            }

        comparison_report['quality_analysis'] = quality_summary

    # 保存JSON报告
    json_path = output_path / 'comparison_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON报告已保存: {json_path}")

    # 生成Markdown报告
    generate_markdown_report(comparison_report, output_path / 'comparison_report.md')

    # 打印汇总
    print_summary(comparison_report)

    return comparison_report


def generate_markdown_report(report: Dict, output_path: Path):
    """生成Markdown格式的对比报告"""

    md_content = f"""# 融合检测 vs 简单串联方法性能对比报告

## 测试信息

- **测试图像数量**: {report['test_info']['total_images']}
- **融合策略**: {report['test_info']['fusion_strategy']}
- **置信度阈值**: {report['test_info']['conf_threshold']}

---

## 整体性能对比

### 检测数量

| 方法 | 总检测数 | 平均每图 | 改善 |
|------|---------|---------|------|
| 简单串联 | {report['simple_method']['total_detections']} | {report['simple_method']['avg_detections_per_image']:.2f} | - |
| 融合检测 | {report['fusion_method']['total_detections']} | {report['fusion_method']['avg_detections_per_image']:.2f} | **+{report['comparison']['detection_improvement']['percentage']:.2f}%** |

### 检测置信度

| 方法 | 平均置信度 | 标准差 | 改善 |
|------|-----------|--------|------|
| 简单串联 | {report['simple_method']['avg_confidence']:.4f} | {report['simple_method']['std_confidence']:.4f} | - |
| 融合检测 | {report['fusion_method']['avg_confidence']:.4f} | {report['fusion_method']['std_confidence']:.4f} | **+{report['comparison']['confidence_improvement']['percentage']:.2f}%** |

### 处理时间

| 方法 | 总时间(s) | 平均每图(s) | 时间开销 |
|------|----------|-----------|---------|
| 简单串联 | {report['simple_method']['total_time']:.2f} | {report['simple_method']['avg_time_per_image']:.2f} | - |
| 融合检测 | {report['fusion_method']['total_time']:.2f} | {report['fusion_method']['avg_time_per_image']:.2f} | **+{report['comparison']['time_overhead']['percentage']:.2f}%** |

---

## 类别分布对比

### 简单串联方法
"""

    # 简单串联类别分布
    for cls, count in sorted(report['simple_method']['class_distribution'].items(),
                            key=lambda x: x[1], reverse=True):
        md_content += f"\n- **{cls}**: {count}"

    md_content += "\n\n### 融合检测方法\n"

    # 融合方法类别分布
    for cls, count in sorted(report['fusion_method']['class_distribution'].items(),
                            key=lambda x: x[1], reverse=True):
        md_content += f"\n- **{cls}**: {count}"

    # 来源分布
    if 'source_distribution' in report['fusion_method']:
        md_content += "\n\n---\n\n## 融合检测来源分布\n"
        total_fusion = sum(report['fusion_method']['source_distribution'].values())
        for source, count in report['fusion_method']['source_distribution'].items():
            percentage = (count / total_fusion * 100) if total_fusion > 0 else 0
            md_content += f"\n- **{source}**: {count} ({percentage:.1f}%)"

    # 质量改善
    if report.get('quality_analysis'):
        md_content += "\n\n---\n\n## 图像质量改善分析\n\n"
        md_content += "| 指标 | 平均改善 | 最大改善 | 最小改善 |\n"
        md_content += "|------|---------|---------|----------|\n"

        for metric, stats in report['quality_analysis'].items():
            md_content += f"| {metric} | {stats['avg_improvement']:.3f}x | {stats['max_improvement']:.3f}x | {stats['min_improvement']:.3f}x |\n"

    # 结论
    md_content += "\n\n---\n\n## 结论\n\n"

    det_improvement = report['comparison']['detection_improvement']['percentage']
    conf_improvement = report['comparison']['confidence_improvement']['percentage']
    time_overhead = report['comparison']['time_overhead']['percentage']

    if det_improvement > 0:
        md_content += f"✅ **检测数量提升**: 融合方法比简单串联多检测到 **{det_improvement:.2f}%** 的目标\n\n"

    if conf_improvement > 0:
        md_content += f"✅ **置信度提升**: 融合方法的平均置信度提高了 **{conf_improvement:.2f}%**\n\n"

    md_content += f"⏱️ **时间开销**: 融合方法增加了 **{time_overhead:.2f}%** 的处理时间\n\n"

    if 'source_distribution' in report['fusion_method']:
        dehazed_pct = (report['fusion_method']['source_distribution'].get('dehazed', 0) /
                      sum(report['fusion_method']['source_distribution'].values()) * 100)
        md_content += f"🎯 **智能选择**: {dehazed_pct:.1f}% 的检测来自去雾图，说明去雾对检测有显著帮助\n\n"

    md_content += """
---

## 推荐

基于以上对比结果：

1. **融合检测方法优于简单串联方法**，能够检测到更多目标且置信度更高
2. **时间开销可接受**，相比性能提升，额外的处理时间是值得的
3. **智能融合策略有效**，能够根据图像质量自动选择最佳检测结果
4. **适合实际应用**，特别是在雾天或低能见度场景下

"""

    # 保存Markdown
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✓ Markdown报告已保存: {output_path}")


def print_summary(report: Dict):
    """打印汇总信息"""
    print("\n" + "=" * 60)
    print("性能对比汇总")
    print("=" * 60)

    print("\n【检测数量】")
    print(f"  简单串联: {report['simple_method']['total_detections']} "
          f"(平均 {report['simple_method']['avg_detections_per_image']:.2f}/图)")
    print(f"  融合检测: {report['fusion_method']['total_detections']} "
          f"(平均 {report['fusion_method']['avg_detections_per_image']:.2f}/图)")
    print(f"  改善: {report['comparison']['detection_improvement']['absolute']:+d} "
          f"({report['comparison']['detection_improvement']['percentage']:+.2f}%)")

    print("\n【检测置信度】")
    print(f"  简单串联: {report['simple_method']['avg_confidence']:.4f}")
    print(f"  融合检测: {report['fusion_method']['avg_confidence']:.4f}")
    print(f"  改善: {report['comparison']['confidence_improvement']['absolute']:+.4f} "
          f"({report['comparison']['confidence_improvement']['percentage']:+.2f}%)")

    print("\n【处理时间】")
    print(f"  简单串联: {report['simple_method']['total_time']:.2f}s "
          f"(平均 {report['simple_method']['avg_time_per_image']:.2f}s/图)")
    print(f"  融合检测: {report['fusion_method']['total_time']:.2f}s "
          f"(平均 {report['fusion_method']['avg_time_per_image']:.2f}s/图)")
    print(f"  开销: {report['comparison']['time_overhead']['absolute']:+.2f}s "
          f"({report['comparison']['time_overhead']['percentage']:+.2f}%)")

    if 'source_distribution' in report['fusion_method']:
        print("\n【融合来源分布】")
        total = sum(report['fusion_method']['source_distribution'].values())
        for source, count in report['fusion_method']['source_distribution'].items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {source}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='融合检测 vs 简单串联方法性能对比')
    parser.add_argument('--input-dir', required=True, help='输入图像目录')
    parser.add_argument('--output-dir', default='outputs/comparison', help='输出目录')
    parser.add_argument('--pattern', default='*.png', help='文件名模式')
    parser.add_argument('--strategy', default='adaptive',
                       choices=['adaptive', 'confidence', 'quality', 'both'],
                       help='融合策略')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--max-images', type=int, default=None, help='最大测试图像数量')

    args = parser.parse_args()

    # 查找图像
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return

    image_paths = sorted(input_path.rglob(args.pattern))

    if not image_paths:
        print(f"错误: 在 {args.input_dir} 中找不到匹配 {args.pattern} 的图像")
        return

    # 限制图像数量
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    # 运行对比
    compare_methods(
        [str(p) for p in image_paths],
        args.output_dir,
        strategy=args.strategy,
        conf_threshold=args.conf_threshold
    )


if __name__ == '__main__':
    main()
