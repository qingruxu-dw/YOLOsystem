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
from yolosystem.coa_adapter import CoADehazer


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
            # num_classes=80,  # COCO数据集类别数
            num_classes=10,  # VisDrone数据集类别数
            pretrained=True
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 去雾模块
        # self.dehazer = DehazingModule()
        print("加载 CoA 去雾模型...")
        self.dehazer = CoADehazer()

        # VisDrone类别名称
        self.class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]

        # 新增：生成随机颜色表，用于不同类别的可视化
        np.random.seed(42)  # 固定种子保证每次运行颜色一致
        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(self.class_names))]

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


    def detect(self, img_path: str, conf_threshold: float = 0.25, save_path: str = None, dehaze_prompt: str = None):
        """
        对单张图像进行检测
        Args:
            img_path: 图像路径
            conf_threshold: 置信度阈值
            save_path: 结果保存路径
            dehaze_prompt: (可选) 用于指导去雾的文本描述，如 "Clear street scene without fog"
        """
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        h_orig, w_orig = img.shape[:2]

        # 1. 去雾 (BGR)
        print("正在去雾 (CoA)...")
        if dehaze_prompt:
             print(f"应用文本指导: '{dehaze_prompt}'")
             img_dehazed_bgr = self.dehazer.process_opencv(img, prompt=dehaze_prompt)
        else:
             img_dehazed_bgr = self.dehazer.process_opencv(img)
        
        # 2. 转换颜色空间为 RGB (供 YOLO 推理)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_dehazed_rgb = cv2.cvtColor(img_dehazed_bgr, cv2.COLOR_BGR2RGB)

        # 预处理
        img_foggy_tensor = self.preprocess(img_rgb)
        img_dehazed_tensor = self.preprocess(img_dehazed_rgb)

        # 推理
        print("正在检测...")
        with torch.no_grad():
            results = self.model(img_foggy_tensor, img_dehazed_tensor)

        # 解析结果
        detections = self.parse_results(results, conf_threshold, (h_orig, w_orig))

        print(f"检测到 {len(detections)} 个目标")

        # 可视化改进：仅保存最终融合后的结果图
        if save_path:
            # 这里的 vis_img 是在“去雾图”的基础上绘制了不同颜色的检测框
            vis_img = self.visualize(img_dehazed_bgr, detections)
            cv2.imwrite(save_path, vis_img)
            print(f"✓ 融合检测结果已保存到: {save_path}")

        return detections


    def parse_results(self, results, conf_threshold: float = 0.25, orig_shape=None):
        """解析YOLO输出"""
        # 尝试从不同的位置导入 NMS，兼容不同版本的 Ultralytics
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            try:
                from ultralytics.utils.ops import non_max_suppression
            except ImportError:
                raise ImportError("无法从 ultralytics.utils.nms 或 ultralytics.utils.ops 导入 non_max_suppression")
        
        detections = []

        if isinstance(results, (list, tuple)):
            results = results[0]  # 取第一个输出 (推理张量)

        # 执行 NMS
        preds = non_max_suppression(
            results, 
            conf_thres=conf_threshold, 
            iou_thres=0.45
        )
        
        det = preds[0]
        if len(det) == 0:
            return []

        # 获取尺寸用于缩放
        if orig_shape:
            h_orig, w_orig = orig_shape
            det[:, 0] *= (w_orig / 640)
            det[:, 1] *= (h_orig / 640)
            det[:, 2] *= (w_orig / 640)
            det[:, 3] *= (h_orig / 640)

        for *xyxy, conf, cls in det:
            detections.append({
                'bbox': [float(x) for x in xyxy],
                'confidence': float(conf),
                'class': int(cls)
            })

        return detections

    def visualize(self, img: np.ndarray, detections: list) -> np.ndarray:
        """可视化检测结果 (带颜色区分)"""
        vis_img = img.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            cls = det['class']
            
            # 获取该类别的颜色
            color = self.colors[cls % len(self.colors)]
            label_name = self.class_names[cls] if cls < len(self.class_names) else f"ID:{cls}"

            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景和文字
            label = f"{label_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_img


    def detect_video(self, video_path: str, output_path: str, conf_threshold: float = 0.25, dehaze_prompt: str = None):
        """
        对视频进行检测
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

            # 去雾和检测
            # 注意：这里我们复用 detect() 的核心逻辑，但为了效率直接处理 frame
            # 视频帧通常不建议逐帧做 TTA (太慢)，除非非常有必要
            if dehaze_prompt and frame_idx == 1:
                print(f"\n注意：视频模式下暂不支持逐帧文本优化（速度过慢），仅使用默认去雾。")
                
            img_dehazed_bgr = self.dehazer.process_opencv(frame)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_dehazed_rgb = cv2.cvtColor(img_dehazed_bgr, cv2.COLOR_BGR2RGB)

            # 预处理
            t1 = self.preprocess(frame_rgb)
            t2 = self.preprocess(frame_dehazed_rgb)

            # 推理
            with torch.no_grad():
                results = self.model(t1, t2)

            # 解析并将坐标还原到原图尺寸 (height, width)
            detections = self.parse_results(results, conf_threshold, (height, width))
            
            # 可视化结果并写入视频
            vis_frame = self.visualize(frame, detections)
            out.write(vis_frame)

        cap.release()
        out.release()
        print(f"\n✓ 视频处理完成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Feature Fusion YOLO 推理与测试脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型权重的路径 (.pth)')
    parser.add_argument('--input', type=str, required=True, help='输入图片或视频的路径')
    parser.add_argument('--output', type=str, default='output_result.jpg', help='输出结果保存路径')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLO模型规模')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--video', action='store_true', help='如果输入是视频请加上此参数')
    parser.add_argument('--prompt', type=str, default=None, help='(可选) 用于指导图像去雾的文本描述')

    args = parser.parse_args()

    # 初始化推理器
    inferencer = FusionInference(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size
    )

    # 执行检测任务
    if args.video:
        # 如果没有指定视频后缀，默认mp4
        output_path = args.output if args.output.endswith('.mp4') else 'output_result.mp4'
        inferencer.detect_video(
            video_path=args.input,
            output_path=output_path,
            conf_threshold=args.conf,
            dehaze_prompt=args.prompt
        )
    else:
        inferencer.detect(
            img_path=args.input,
            conf_threshold=args.conf,
            save_path=args.output,
            dehaze_prompt=args.prompt
        )

if __name__ == '__main__':
    main()