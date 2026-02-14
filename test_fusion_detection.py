"""
测试Feature-level Fusion的检测性能

对比三种方案：
1. 原图检测
2. 去雾图检测
3. 融合检测
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from yolosystem.feature_fusion_yolo_simple import create_feature_fusion_yolo


class DetectionComparison:
    """检测性能对比"""

    def __init__(self, fusion_checkpoint: str, yolo_checkpoint: str = None, model_size: str = 's', conf_threshold: float = 0.25):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold

        print(f"使用设备: {self.device}")
        
        # 类别名称 - 修正：不再硬编码，而是支持从yaml加载或手动设置
        # 这是 VisDrone/你的自定义数据集的类别
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]

        # 加载融合模型
        print("\n加载融合模型...")
        
        # 智能加载策略：尝试匹配权重的结构
        try:
            print(f"尝试加载模型 (类别数: {len(self.class_names)})...")
            self._load_model_safe(fusion_checkpoint, len(self.class_names), model_size)
            print("✓ 模型加载成功 (匹配目标类别数)")
            
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠ 警告: 权重形状不匹配 ({e})")
                print("尝试使用默认结构 (80类) 加载旧版权重...")
                # 重新尝试使用默认COCO类别数(80)加载
                self._load_model_safe(fusion_checkpoint, 80, model_size)
                print("✓ 旧版模型加载成功 (已兼容运行)")
            else:
                raise e

        # 加载标准YOLO（用于对比）
        print("\n加载标准YOLO模型...")
        # 优先使用指定的权重文件，否则回退到官方预训练权重
        yolo_path = yolo_checkpoint if yolo_checkpoint else f'yolo11{model_size}.pt'
        print(f"权重路径: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        print("✓ 标准YOLO加载成功")

        # 类别名称 - 修正：不再硬编码，而是支持从yaml加载或手动设置
        # 这是 VisDrone/你的自定义数据集的类别
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]

        # 生成不同类别的颜色表 (RGB)
        np.random.seed(42)  # 固定种子以保证颜色一致
        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)]
        
        print(f"检测类别数: {len(self.class_names)}")

    def _load_model_safe(self, checkpoint_path, num_classes, model_size):
        """辅助函数：创建指定类别数的模型并加载权重"""
        self.fusion_model = create_feature_fusion_yolo(
            model_size=model_size,
            num_classes=num_classes,
            fusion_type='learned',
            pretrained=False # 加载此checkpoint不需要预训练权重，我们马上覆盖
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()

    def detect_with_fusion(self, img_orig, img_dehz):
        """使用融合模型检测"""
        # 预处理
        h_orig, w_orig = img_orig.shape[:2]
        img_orig_padded = cv2.resize(img_orig, (640, 640))
        img_dehz_padded = cv2.resize(img_dehz, (640, 640))

        img_orig_tensor = torch.from_numpy(img_orig_padded).float().permute(2, 0, 1) / 255.0
        img_dehz_tensor = torch.from_numpy(img_dehz_padded).float().permute(2, 0, 1) / 255.0

        img_orig_tensor = img_orig_tensor.unsqueeze(0).to(self.device)
        img_dehz_tensor = img_dehz_tensor.unsqueeze(0).to(self.device)
        
        # 1. 前向传播
        self.fusion_model.eval()
        with torch.no_grad():
             # 双输入模型
             results = self.fusion_model(img_orig_tensor, img_dehz_tensor)
             
             # 如果模型没能正确切换到推理模式，手动触发 Detect 模块的合并逻辑
             # YOLOv8/v11 的输出在训练/验证混淆模式下可能是 list
            #  if isinstance(results, list):
            #      # 这种情况下通常是 3 个尺度的预测结果列表
            #      # 我们可以尝试使用结果中的某些部分或者检查模型状态
            #      print(f"DEBUG: results is list of len {len(results)}")
            #      # 这里通常说明 Detect Head 还在输出 Raw Feature Maps
            #      # 强制模型重新设置为推理模式并确保属性正确
            #      pass
                         # 重要：YOLOv11/v8 的原生 model(x) 可能返回 (tensor, list) 
             # 我们只需要第一个推理张量
             if isinstance(results, (list, tuple)):
                results = results[0]

             # 尝试再次调用，但在调用前确保底层子模块也是 eval 模式
             # 这里再次尝试 NMS
             preds = non_max_suppression(
                 results, 
                 conf_thres=self.conf_threshold, 
                 iou_thres=0.45,
                 classes=None
             )
             
             det = preds[0]  # 获取第一张图的结果 [N, 6] (x1, y1, x2, y2, conf, cls)
             
             if len(det) == 0:
                 return []
             
             # 将坐标从 640x640 缩放回原图尺寸
             # det[:, :4] 是 xyxy 格式
             det[:, 0] *= (w_orig / 640)
             det[:, 1] *= (h_orig / 640)
             det[:, 2] *= (w_orig / 640)
             det[:, 3] *= (h_orig / 640)
             
             formatted_results = []
             for *xyxy, conf, cls in det:
                 cls = int(cls)
                 formatted_results.append({
                     'box': [float(x) for x in xyxy],
                     'confidence': float(conf),
                     'class': cls,
                     'class_name': self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                 })
                 
             return formatted_results

    def detect_with_yolo(self, img):
        """使用标准YOLO检测"""
        results = self.yolo_model(img, conf=self.conf_threshold, verbose=False)
        return self.parse_results(results[0])

    def parse_results(self, result):
        """解析标准YOLO的结果"""
        detections = []
        if result.boxes:
            for box in result.boxes:
                # 获取数据并处理成简单 python 类型
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'box': xyxy,
                    'confidence': conf,
                    'class': cls,
                    'class_name': self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                })
        return detections

    def draw_detections(self, img, detections, title):
        """绘制检测框"""
        img_draw = img.copy()
        # h, w = img.shape[:2]

        # # 缩放比例（从640到原始尺寸）
        # scale_x = w / 640
        # scale_y = h / 640

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制框
            cls_id = int(det['class'])
            # 获取对应颜色，使用取模防止越界，并确保是整数元组
            color = self.colors[cls_id % len(self.colors)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # 计算文字背景框大小
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # 绘制文字背景，使文字更清晰
            cv2.rectangle(img_draw, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            
            # 文字使用白色，画在彩色背景上
            cv2.putText(img_draw, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加标题
        cv2.putText(img_draw, f"{title} ({len(detections)} objects)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img_draw

    def test_on_dataset(self, data_dir: str, output_dir: Path):
        """在数据集上测试"""
        data_path = Path(data_dir)
        val_orig_dir = data_path / 'val' / 'images_original'
        val_dehz_dir = data_path / 'val' / 'images_dehazed'

        img_files = list(val_orig_dir.glob('*.jpg'))

        print(f"\n测试图像数量: {len(img_files)}")

        results = {
            'original': {'total_detections': 0, 'avg_confidence': [], 'per_image': []},
            'dehazed': {'total_detections': 0, 'avg_confidence': [], 'per_image': []},
            'fusion': {'total_detections': 0, 'avg_confidence': [], 'per_image': []}
        }

        # 创建可视化目录
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_file in enumerate(tqdm(img_files, desc="检测中")):
            # 读取图像
            img_orig = cv2.imread(str(img_file))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            img_dehz_path = val_dehz_dir / img_file.name
            img_dehz = cv2.imread(str(img_dehz_path))
            img_dehz = cv2.cvtColor(img_dehz, cv2.COLOR_BGR2RGB)

            # 三种检测
            det_orig = self.detect_with_yolo(img_orig)
            det_dehz = self.detect_with_yolo(img_dehz)
            # 注意：这里我们使用特征级融合进行检测
            # 但为了可视化更清晰，我们将检测框画在【去雾图】上
            detections_fusion = self.detect_with_fusion(img_orig, img_dehz)
            results['fusion']['total_detections'] += len(detections_fusion)
            if detections_fusion:
                confs = [d['confidence'] for d in detections_fusion]
                results['fusion']['avg_confidence'].extend(confs)
            results['fusion']['per_image'].append(len(detections_fusion))

            # 可视化前5张
            if idx < 5:
                # 绘制结果
                # 修改：这里的 vis_orig, vis_dehz, vis_fusion 都是 OpenCV BGR 格式
                # 而我们的输入是 RGB 格式，所以需要转换一下颜色
                
                # 重新读取图片以保证干净的背景（或者拷贝一份并转回 BGR）
                # 这里我们复用 draw_detections，它内部会copy
                # 但需要注意 draw_detections 修改的是 RGB 图像
                # 所以我们先转回 BGR 再保存，或者直接用 RGB 画图最后保存时转 RGB->BGR
                
                # 简单做法：这里我们统一在 RGB 上画，最后统一转 BGR 保存
                vis_orig_rgb = self.draw_detections(img_orig, det_orig, "ORIGINAL")
                vis_dehz_rgb = self.draw_detections(img_dehz, det_dehz, "DEHAZED")
                vis_fusion_rgb = self.draw_detections(img_dehz, detections_fusion, "FUSION (Feature-level)")

                # 拼接
                h, w = img_orig.shape[:2]
                vis_combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
                vis_combined[:, :w] = vis_orig_rgb
                vis_combined[:, w:w*2] = vis_dehz_rgb
                vis_combined[:, w*2:] = vis_fusion_rgb

                # RGB -> BGR for saving
                vis_combined_bgr = cv2.cvtColor(vis_combined, cv2.COLOR_RGB2BGR)

                output_path = vis_dir / f"{img_file.stem}_comparison.jpg"
                cv2.imwrite(str(output_path), vis_combined_bgr)
            
            # 统计
            results['original']['total_detections'] += len(det_orig)
            if det_orig:
                 results['original']['avg_confidence'].extend([d['confidence'] for d in det_orig])
            results['original']['per_image'].append(len(det_orig))

            results['dehazed']['total_detections'] += len(det_dehz)
            if det_dehz:
                 results['dehazed']['avg_confidence'].extend([d['confidence'] for d in det_dehz])
            results['dehazed']['per_image'].append(len(det_dehz))

        return results

    def print_results(self, results):
        """打印结果表格"""
        print("\n" + "="*60)
        print("检测性能对比")
        print("="*60)

        for name in ['original', 'dehazed', 'fusion']:
            data = results[name]
            total = data['total_detections']
            avg_conf = np.mean(data['avg_confidence']) if data['avg_confidence'] else 0
            per_img = np.mean(data['per_image']) if data['per_image'] else 0

            print(f"\n{name.upper()}:")
            print(f"  总检测数: {total}")
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  平均每张图: {per_img:.1f}")

        print("\n" + "="*60)
        print("性能提升")
        print("="*60)

        orig_total = results['original']['total_detections']
        dehz_total = results['dehazed']['total_detections']
        fusion_total = results['fusion']['total_detections']

        print("\n融合 vs 原图:")
        if orig_total > 0:
            diff = fusion_total - orig_total
            print(f"  检测数提升: {diff} ({diff / orig_total * 100:+.1f}%)")
        else:
            print(f"  检测数提升: {fusion_total} (原图未检测到目标)")

        print("\n融合 vs 去雾图:")
        if dehz_total > 0:
            diff = fusion_total - dehz_total
            print(f"  检测数提升: {diff} ({diff / dehz_total * 100:+.1f}%)")
        else:
            print(f"  检测数提升: {fusion_total} (去雾图未检测到目标)")

        # 判断
        if fusion_total > max(orig_total, dehz_total):
            print("\n✓ 融合检测效果最好！")
        elif fusion_total > min(orig_total, dehz_total):
            print("\n⚠ 融合检测效果中等")
        else:
            print("\n✗ 融合检测效果不如单独检测")


def main():
    parser = argparse.ArgumentParser(description='测试Feature-level Fusion检测性能')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型检查点')
    parser.add_argument('--yolo-weights', type=str, default=None, help='标准YOLO权重路径 (可选，用于对比)')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--model-size', type=str, default='s', help='模型大小')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--output-dir', type=str, default='detection_results', help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Feature-level Fusion 检测性能测试")
    print("="*60)

    # 创建对比器
    comparator = DetectionComparison(
        fusion_checkpoint=args.checkpoint,
        yolo_checkpoint=args.yolo_weights,
        model_size=args.model_size,
        conf_threshold=args.conf_threshold
    )

    # 测试
    results = comparator.test_on_dataset(args.data_dir, output_dir)

    # 打印结果
    comparator.print_results(results)

    # 保存结果
    results_file = output_dir / 'detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {results_file}")
    print(f"可视化已保存: {output_dir / 'visualizations'}")


if __name__ == '__main__':
    main()
