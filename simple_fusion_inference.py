"""
简单的融合检测推理脚本
适合快速测试和部署
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
# from yolosystem.dehazing import DehazingModule
from yolosystem.coa_adapter import CoADehazer


class SimpleFusionDetector:
    """简化的融合检测器（使用预训练YOLO）"""

    def __init__(self, model_path: str = 'yolo11n.pt'):
        """
        初始化检测器

        Args:
            model_path: YOLO模型路径
        """
        print(f"加载YOLO模型: {model_path}")
        self.yolo = YOLO(model_path)
        # self.dehazer = DehazingModule()
        print("加载 CoA 去雾模型 (支持文本指导)...")
        self.dehazer = CoADehazer(load_clip=True)
        print("✓ 模型加载完成")

    def detect_image(self, img_path: str, output_dir: str = 'outputs/inference',
                    conf_threshold: float = 0.25, fusion_weight: float = 0.7, 
                    dehaze_prompt: str = None):
        """
        对单张图像进行融合检测

        Args:
            img_path: 输入图像路径
            output_dir: 输出目录
            conf_threshold: 置信度阈值
            fusion_weight: 去雾图权重（0-1），有雾图权重为1-fusion_weight
            dehaze_prompt: (可选) 用于指导去雾的文本描述

        Returns:
            检测结果统计
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 读取图像
        print(f"\n处理图像: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 去雾
        print("正在去雾...")
        
        # 使用 CoA 去雾 (支持 prompt)
        if dehaze_prompt:
             print(f"应用文本指导: '{dehaze_prompt}'")
             img_dehazed = self.dehazer.process_opencv(img, prompt=dehaze_prompt)
        else:
             img_dehazed = self.dehazer.process_opencv(img)

        # 融合图像（简单的加权融合）
        print(f"融合图像（去雾权重: {fusion_weight:.1%}）...")
        img_fusion = (img_rgb * (1 - fusion_weight) + img_dehazed * fusion_weight).astype(np.uint8)

        # 三种检测
        print("正在检测...")
        results_foggy = self.yolo(img_rgb, conf=conf_threshold, verbose=False)[0]
        results_dehazed = self.yolo(img_dehazed, conf=conf_threshold, verbose=False)[0]
        results_fusion = self.yolo(img_fusion, conf=conf_threshold, verbose=False)[0]

        # 统计
        stats = {
            'foggy': len(results_foggy.boxes),
            'dehazed': len(results_dehazed.boxes),
            'fusion': len(results_fusion.boxes)
        }

        print(f"\n检测结果:")
        print(f"  有雾图: {stats['foggy']} 个目标")
        print(f"  去雾图: {stats['dehazed']} 个目标")
        print(f"  融合图: {stats['fusion']} 个目标")

        # 保存结果
        img_name = Path(img_path).stem

        # 保存有雾图检测结果
        vis_foggy = results_foggy.plot()
        cv2.imwrite(str(output_path / f'{img_name}_foggy.jpg'),
                   cv2.cvtColor(vis_foggy, cv2.COLOR_RGB2BGR))

        # 保存去雾图检测结果
        vis_dehazed = results_dehazed.plot()
        cv2.imwrite(str(output_path / f'{img_name}_dehazed.jpg'),
                   cv2.cvtColor(vis_dehazed, cv2.COLOR_RGB2BGR))

        # 保存融合图检测结果
        vis_fusion = results_fusion.plot()
        cv2.imwrite(str(output_path / f'{img_name}_fusion.jpg'),
                   cv2.cvtColor(vis_fusion, cv2.COLOR_RGB2BGR))

        # 保存对比图
        self.save_comparison(img_rgb, img_dehazed, img_fusion,
                           results_foggy, results_dehazed, results_fusion,
                           output_path / f'{img_name}_comparison.jpg')

        print(f"\n✓ 结果已保存到: {output_path}")

        return stats

    def save_comparison(self, img_foggy, img_dehazed, img_fusion,
                       results_foggy, results_dehazed, results_fusion,
                       save_path):
        """保存对比图"""
        # 绘制检测结果
        vis_foggy = results_foggy.plot()
        vis_dehazed = results_dehazed.plot()
        vis_fusion = results_fusion.plot()

        # 调整大小
        h, w = 400, 600
        vis_foggy = cv2.resize(vis_foggy, (w, h))
        vis_dehazed = cv2.resize(vis_dehazed, (w, h))
        vis_fusion = cv2.resize(vis_fusion, (w, h))

        # 添加标题
        def add_title(img, title, count):
            img_with_title = np.ones((h + 60, w, 3), dtype=np.uint8) * 255
            img_with_title[60:, :] = img
            cv2.putText(img_with_title, title, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(img_with_title, f"Detections: {count}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return img_with_title

        vis_foggy = add_title(vis_foggy, "Foggy Image", len(results_foggy.boxes))
        vis_dehazed = add_title(vis_dehazed, "Dehazed Image", len(results_dehazed.boxes))
        vis_fusion = add_title(vis_fusion, "Fusion Image", len(results_fusion.boxes))

        # 拼接
        comparison = np.hstack([vis_foggy, vis_dehazed, vis_fusion])

        # 保存
        cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    def detect_video(self, video_path: str, output_path: str = 'output.mp4',
                    conf_threshold: float = 0.25, fusion_weight: float = 0.7,
                    dehaze_prompt: str = None):
        """
        对视频进行融合检测

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
            fusion_weight: 去雾图权重
            dehaze_prompt: (可选) 用于指导去雾的文本描述
        """
        cap = cv2.VideoCapture(video_path)

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"\n处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")

        frame_idx = 0
        total_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 去雾
            # frame_dehazed, _ = self.dehazer.dehaze(frame_rgb)
            # 视频暂不支持逐帧文本优化，太慢
            if dehaze_prompt and frame_idx == 1:
                print(f"\n注意：视频模式下暂不支持逐帧文本优化，仅使用默认 CoA 去雾。")
            
            # 使用 CoA 适配器 (输入必须是 BGR for process_opencv, or RGB)
            # self.dehazer.process_opencv 接受 BGR
            frame_dehazed_bgr = self.dehazer.process_opencv(frame)
            frame_dehazed = cv2.cvtColor(frame_dehazed_bgr, cv2.COLOR_BGR2RGB)

            # 融合
            frame_fusion = (frame_rgb * (1 - fusion_weight) +
                          frame_dehazed * fusion_weight).astype(np.uint8)

            # 检测
            results = self.yolo(frame_fusion, conf=conf_threshold, verbose=False)[0]
            num_detections = len(results.boxes)
            total_detections += num_detections

            # 可视化
            vis_frame = results.plot()
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            # 添加信息
            cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Detections: {num_detections}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 写入
            out.write(vis_frame)

            # 进度
            if frame_idx % 30 == 0:
                print(f"处理进度: {frame_idx}/{total_frames} "
                     f"({frame_idx/total_frames*100:.1f}%)", end='\r')

        cap.release()
        out.release()

        print(f"\n✓ 视频处理完成: {output_path}")
        print(f"  总检测数: {total_detections}")
        print(f"  平均每帧: {total_detections/frame_idx:.1f} 个目标")

    def detect_folder(self, folder_path: str, output_dir: str = 'outputs/batch',
                     conf_threshold: float = 0.25, fusion_weight: float = 0.7,
                     dehaze_prompt: str = None):
        """
        批量处理文件夹中的图像

        Args:
            folder_path: 输入文件夹路径
            output_dir: 输出目录
            conf_threshold: 置信度阈值
            fusion_weight: 去雾图权重
            dehaze_prompt: (可选) 用于指导去雾的文本描述
        """
        folder = Path(folder_path)
        img_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))

        if not img_files:
            print(f"❌ 未找到图像文件: {folder_path}")
            return

        print(f"\n找到 {len(img_files)} 张图像")

        all_stats = []
        for img_file in img_files:
            try:
                stats = self.detect_image(
                    str(img_file),
                    output_dir=output_dir,
                    conf_threshold=conf_threshold,
                    fusion_weight=fusion_weight,
                    dehaze_prompt=dehaze_prompt
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"❌ 处理失败 {img_file.name}: {e}")

        # 总结
        if all_stats:
            avg_foggy = np.mean([s['foggy'] for s in all_stats])
            avg_dehazed = np.mean([s['dehazed'] for s in all_stats])
            avg_fusion = np.mean([s['fusion'] for s in all_stats])

            print(f"\n=== 批量处理总结 ===")
            print(f"处理图像数: {len(all_stats)}")
            print(f"平均检测数:")
            print(f"  有雾图: {avg_foggy:.1f}")
            print(f"  去雾图: {avg_dehazed:.1f}")
            print(f"  融合图: {avg_fusion:.1f}")
            print(f"融合提升: {(avg_fusion - avg_foggy) / avg_foggy * 100:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='简单融合检测推理')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像/视频/文件夹路径')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='输出路径')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--fusion-weight', type=float, default=0.7,
                       help='去雾图权重（0-1）')
    parser.add_argument('--prompt', type=str, default=None,
                       help='(可选) 用于指导图像去雾的文本描述 (支持英文)')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'video', 'folder'],
                       help='处理模式')

    args = parser.parse_args()

    # 创建检测器
    detector = SimpleFusionDetector(model_path=args.model)

    # 推理
    if args.mode == 'image':
        detector.detect_image(
            img_path=args.input,
            output_dir=args.output,
            conf_threshold=args.conf,
            fusion_weight=args.fusion_weight,
            dehaze_prompt=args.prompt
        )
    elif args.mode == 'video':
        detector.detect_video(
            video_path=args.input,
            output_path=args.output,
            conf_threshold=args.conf,
            fusion_weight=args.fusion_weight,
            dehaze_prompt=args.prompt
        )
    elif args.mode == 'folder':
        detector.detect_folder(
            folder_path=args.input,
            output_dir=args.output,
            conf_threshold=args.conf,
            fusion_weight=args.fusion_weight,
            dehaze_prompt=args.prompt
        )


if __name__ == '__main__':
    main()
