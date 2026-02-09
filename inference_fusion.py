"""
Feature Fusion YOLO 推理脚本
用于测试和部署融合模型
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo
from yolosystem.dehazing import DehazingModule


class FusionInference:
    """融合模型推理器"""

    def __init__(self, checkpoint_path: str, model_size: str = 's'):
        """
        初始化推理器

        Args:
            checkpoint_path: 训练好的模型权重路径
            model_size: YOLO模型大小 ('n', 's', 'm', 'l', 'x')
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 加载融合模型
        print(f"加载融合模型: {checkpoint_path}")
        self.model = create_feature_fusion_yolo(
            model_size=model_size,
            num_classes=80,  # COCO数据集类别数
            pretrained=True
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 去雾模块
        self.dehazer = DehazingModule()

        # COCO类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        print("✓ 模型加载完成")

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        img_resized = cv2.resize(img, (640, 640))
        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        # 转换为tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def detect(self, img_path: str, conf_threshold: float = 0.25, save_path: str = None):
        """
        对单张图像进行检测

        Args:
            img_path: 输入图像路径
            conf_threshold: 置信度阈值
            save_path: 保存结果的路径（可选）

        Returns:
            检测结果
        """
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 去雾
        print("正在去雾...")
        img_dehazed, _ = self.dehazer.dehaze(img_rgb)

        # 预处理
        img_foggy_tensor = self.preprocess(img_rgb)
        img_dehazed_tensor = self.preprocess(img_dehazed)

        # 推理
        print("正在检测...")
        with torch.no_grad():
            results = self.model(img_foggy_tensor, img_dehazed_tensor)

        # 解析结果
        detections = self.parse_results(results, conf_threshold)

        print(f"检测到 {len(detections)} 个目标")

        # 可视化
        if save_path:
            vis_img = self.visualize(img, detections)
            cv2.imwrite(save_path, vis_img)
            print(f"✓ 结果已保存到: {save_path}")

        return detections

    def parse_results(self, results, conf_threshold: float = 0.25):
        """解析YOLO输出"""
        detections = []

        # YOLO输出格式: [batch, num_boxes, 4+num_classes]
        # 4: x, y, w, h
        # num_classes: 类别概率

        if isinstance(results, (list, tuple)):
            results = results[0]  # 取第一个输出

        # 简化处理：假设results已经是处理好的检测结果
        # 实际使用时需要根据YOLO的具体输出格式调整

        return detections

    def visualize(self, img: np.ndarray, detections: list) -> np.ndarray:
        """可视化检测结果"""
        vis_img = img.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class']
            label = self.class_names[cls] if cls < len(self.class_names) else str(cls)

            # 绘制边界框
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 绘制标签
            text = f"{label}: {conf:.2f}"
            cv2.putText(vis_img, text, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_img

    def detect_video(self, video_path: str, output_path: str, conf_threshold: float = 0.25):
        """
        对视频进行检测

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
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

        print(f"处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            print(f"处理帧 {frame_idx}/{total_frames}", end='\r')

            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 去雾
            frame_dehazed, _ = self.dehazer.dehaze(frame_rgb)

            # 预处理
            frame_foggy_tensor = self.preprocess(frame_rgb)
            frame_dehazed_tensor = self.preprocess(frame_dehazed)

            # 推理
            with torch.no_grad():
                results = self.model(frame_foggy_tensor, frame_dehazed_tensor)

            # 解析并可视化
            detections = self.parse_results(results, conf_threshold)
            vis_frame = self.visualize(frame, detections)

            # 写入输出视频
            out.write(vis_frame)

        cap.release()
        out.release()

        print(f"\n✓ 视频处理完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Feature Fusion YOLO 推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像或视频路径')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='输出路径')
    parser.add_argument('--model-size', type=str, default='s',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO模型大小')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--video', action='store_true',
                       help='输入是否为视频')

    args = parser.parse_args()

    # 创建推理器
    inferencer = FusionInference(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size
    )

    # 推理
    if args.video:
        inferencer.detect_video(
            video_path=args.input,
            output_path=args.output,
            conf_threshold=args.conf
        )
    else:
        inferencer.detect(
            img_path=args.input,
            conf_threshold=args.conf,
            save_path=args.output
        )


if __name__ == '__main__':
    main()
