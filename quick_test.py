"""
使用本地测试图像快速验证融合检测
"""

import cv2
from pathlib import Path
from yolosystem import FusionDetectionPipeline


def quick_test():
    """使用本地测试图像快速验证"""

    # 检查是否有测试图像
    test_dirs = [
        Path('test_images'),
        Path('outputs'),
        Path('.'),
    ]

    image_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(test_dir.glob(f'**/*{ext}')))
                image_files.extend(list(test_dir.glob(f'**/*{ext.upper()}')))

    # 去重
    image_files = list(set(image_files))

    if not image_files:
        print("错误: 未找到测试图像")
        print("\n请将一些测试图像放到以下目录之一:")
        print("  - test_images/")
        print("  - 当前目录")
        print("\n或者从网上下载一些有雾的道路图像：")
        print("  Google Images: 'foggy road' 或 'hazy street'")
        return

    print(f"找到 {len(image_files)} 张测试图像")
    print("="*80)

    # 初始化流水线
    print("初始化融合检测流水线...")
    pipeline = FusionDetectionPipeline()

    # 测试前几张图像
    test_count = min(3, len(image_files))
    print(f"测试前 {test_count} 张图像...\n")

    for i, img_file in enumerate(image_files[:test_count], 1):
        print(f"[{i}/{test_count}] 处理: {img_file.name}")

        try:
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  ⚠️  无法读取图像")
                continue

            # 处理
            results = pipeline.process_image(
                img,
                save_path=str(Path('outputs/quick_test') / img_file.name)
            )

            # 获取统计
            stats = pipeline.get_statistics(results)

            # 打印结果
            print(f"  ✅ 完成")
            print(f"     原图检测: {stats['original']['total_detections']} 个")
            print(f"     去雾图检测: {stats['dehazed']['total_detections']} 个")
            print(f"     融合后检测: {stats['fused']['total_detections']} 个")

            # 质量改善
            quality_imp = stats['fusion_info']['quality_improvement']
            avg_imp = sum(quality_imp.values()) / len(quality_imp)
            print(f"     图像质量改善: {avg_imp:.2f}x")

            # 来源分布
            if 'source_distribution' in stats['fusion_info']:
                dist = stats['fusion_info']['source_distribution']
                total = sum(dist.values())
                if total > 0:
                    print(f"     来源分布: ", end="")
                    for source, count in dist.items():
                        pct = count / total * 100
                        print(f"{source}={pct:.0f}% ", end="")
                    print()
            print()

        except Exception as e:
            print(f"  ❌ 处理失败: {e}\n")

    print("="*80)
    print("快速测试完成！")
    print("\n结果保存在: outputs/quick_test/")
    print("  - *_original_detection.jpg (原图检测)")
    print("  - *_dehazed_detection.jpg (去雾图检测)")
    print("  - *_fused_detection.jpg (融合检测)")
    print("  - *_comparison.jpg (三路对比)")
    print("\n✅ 代码运行正常！")
    print("\n下一步:")
    print("  1. 查看生成的对比图像")
    print("  2. 下载Foggy Cityscapes进行完整测试")
    print("     https://www.cityscapes-dataset.com/")
    print("="*80)


if __name__ == '__main__':
    quick_test()
