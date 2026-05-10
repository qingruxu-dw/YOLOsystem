"""
测试Feature-level Fusion的检测性能

对比三种方案：
1. 原图检测
2. 去雾图检测
3. 融合检测
"""
#利用自己拍摄的图像进行测试
# python test_fusion_detection.py --data-dir datasets/fusion_training/testschool/images --checkpoint runs/visdrone_fusion_v2/best.pth --yolo-weights yolov11_visdrone.pt --model-size n --output-dir detection_school --conf-threshold 0.5

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

try:
    from yolosystem.coa_adapter import CoADehazer
except ImportError:
    try:
        from coa_adapter import CoADehazer
    except ImportError:
        CoADehazer = None

try:
    from yolosystem.feature_fusion_yolo import FeatureFusionYOLO
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from yolosystem.feature_fusion_yolo import FeatureFusionYOLO


class DetectionComparison:
    """检测性能对比"""

    def __init__(self, fusion_checkpoint: str, yolo_checkpoint: str = None, model_size: str = 's', conf_threshold: float = 0.25, auto_dehaze: bool = False, dehaze_weights: str = None, map_to_rtts: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.auto_dehaze = auto_dehaze
        self.map_to_rtts = map_to_rtts

        print(f"使用设备: {self.device}")
        
        if self.auto_dehaze:
            if CoADehazer is None:
                print("⚠ 警告: 无法导入 CoADehazer，无法自动去雾。请确保 coa_adapter.py 存在。")
                self.auto_dehaze = False
            else:
                print("\n加载去雾模型 (CoA)...")
                try:
                    self.dehazer = CoADehazer(weights_path=dehaze_weights, device=self.device)
                    print("✓ 去雾模型加载成功")
                except Exception as e:
                    print(f"⚠ 警告: 加载去雾模型失败: {e}")
                    self.auto_dehaze = False
                    self.dehazer = None

        # 类别名称 - 修正：不再硬编码，而是支持从yaml加载或手动设置
        # 这是 VisDrone/你的自定义数据集的类别
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        
        self.rtts_classes = ['person', 'bicycle', 'car', 'bus', 'motor']

        self.vis_to_rtts_map = {
            0: 0, # pedestrian -> person
            1: 0, # people -> person
            2: 1, # bicycle -> bicycle
            3: 2, # car -> car
            4: 2, # van -> car
            5: 2, # truck -> car
            6: -1, # tricycle -> skip
            7: -1, # awning-tricycle -> skip
            8: 3, # bus -> bus
            9: 4  # motor -> motor
        }

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
        if self.map_to_rtts:
            print(f"✅ 开启类名映射：VisDrone(10类) -> RTTS(5类) : {self.rtts_classes}")

    def _load_model_safe(self, checkpoint_path, num_classes, model_size):
        """辅助函数：创建指定类别数的模型并加载权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            if 'weights_only' in str(e) or 'Unsupported global' in str(e):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            else:
                raise

        # 尝试从 args 中提取 fusion_layers，未找到则默认 [2, 4, 6] 或 [3, 6, 9] (如果是后面发现的)
        fusion_layers = [3, 6, 9]  # 强制改为匹配训练的层数
        if 'args' in checkpoint and hasattr(checkpoint['args'], 'fusion_layers'):
            fusion_layers = checkpoint['args'].fusion_layers
        elif 'args' in checkpoint and isinstance(checkpoint['args'], dict) and 'fusion_layers' in checkpoint['args']:
            fusion_layers = checkpoint['args']['fusion_layers']
            
        self.fusion_model = FeatureFusionYOLO(
            model_size=model_size,
            num_classes=num_classes,
            fusion_layers=fusion_layers,
            pretrained=False # 加载此checkpoint不需要预训练权重，我们马上覆盖
        )
        
        # 兼容单独保存 state_dict 或保存了整个 checkpoint 字典的情况
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 使用 strict=False 容忍 Detect 头部动态形状的警告
        load_result = self.fusion_model.load_state_dict(state_dict, strict=False)
        print(f"  [Info] 模型权重加载完毕，missing_keys={len(load_result.missing_keys)}")
        
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()

    def detect_with_fusion(self, img_orig, img_dehz):
        """使用融合模型检测"""
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import scale_boxes
        
        # 预处理 - 使用与官方一致的 LetterBox (打黑边) 保留长宽比
        h_orig, w_orig = img_orig.shape[:2]
        
        # 实例化官方 LetterBox 预处理器
        letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
        
        # 处理两张图像，确保缩放和 padding 的行为完全一致
        img_orig_padded = letterbox(image=img_orig)
        img_dehz_padded = letterbox(image=img_dehz)

        # 转换为 tensor
        img_orig_tensor = torch.from_numpy(img_orig_padded).float().permute(2, 0, 1) / 255.0
        img_dehz_tensor = torch.from_numpy(img_dehz_padded).float().permute(2, 0, 1) / 255.0

        img_orig_tensor = img_orig_tensor.unsqueeze(0).to(self.device)
        img_dehz_tensor = img_dehz_tensor.unsqueeze(0).to(self.device)
        
        # 1. 前向传播
        self.fusion_model.eval()
        with torch.no_grad():
             # 双输入模型
             results = self.fusion_model(img_orig_tensor, img_dehz_tensor)
             
             # 我们只需要第一个推理张量
             if isinstance(results, (list, tuple)):
                results = results[0]

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
             
             # 使用官方方法将带有 Padded 的坐标精准还原回原始图像的长宽比坐标中去
             det[:, :4] = scale_boxes((640, 640), det[:, :4], (h_orig, w_orig))
             
             formatted_results = []
             for *xyxy, conf, cls in det:
                 cls = int(cls)
                 if self.map_to_rtts:
                     cls = self.vis_to_rtts_map.get(cls, -1)
                     if cls == -1:
                         continue # skip this class
                     class_name = self.rtts_classes[cls]
                 else:
                     class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                     
                 formatted_results.append({
                     'box': [float(x) for x in xyxy],
                     'confidence': float(conf),
                     'class': cls,
                     'class_name': class_name
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
                
                if self.map_to_rtts:
                    cls = self.vis_to_rtts_map.get(cls, -1)
                    if cls == -1:
                        continue # skip this class
                    class_name = self.rtts_classes[cls]
                else:
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                
                detections.append({
                    'box': xyxy,
                    'confidence': conf,
                    'class': cls,
                    'class_name': class_name
                })
        return detections

    def draw_detections(self, img, detections, title):
        """绘制检测框"""
        img_draw = img.copy()
        
        # 为了避免框堆叠，进行简单的置信度过滤和 NMS（如果需要），但传入的应该已经做了 NMS。
        # 如果框依然太多，可能 conf_threshold 太低 (0.001)
        # 这里我们在画图时只画置信度高于 0.25 的，让画出来的图好看，但不影响 mAP 计算
        plot_threshold = 0.25
        
        for det in detections:
            conf = det['confidence']
            if conf < plot_threshold:
                continue
                
            bbox = det['box']
            cls = det['class']
            class_name = det['class_name']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[cls % len(self.colors)]
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # 再算一下实际画出来的目标数 (在 RGB 空间下，(255, 0, 0) 为红色)
        drawn_count = sum(1 for d in detections if d['confidence'] >= plot_threshold)
        cv2.putText(img_draw, f"{title} ({drawn_count} objects)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img_draw

    def evaluate_dataset(self, orig_dir, dehz_dir, label_dir, output_dir):
        """对整个数据集基于当前前向传播方法计算 mAP@0.5，并绘制图表与对比图"""
        print(f"\n开始在数据集上进行定量评测 mAP@0.5 ...")
        print(f"原图目录: {orig_dir}")
        print(f"去雾目录: {dehz_dir}")
        print(f"标签目录: {label_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        img_out_dir = output_path / "comparison_images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        orig_paths = list(Path(orig_dir).glob('*.jpg')) + list(Path(orig_dir).glob('*.png'))
        if len(orig_paths) == 0:
            print("未找到测试图像！")
            return
            
        all_preds_hazy = []
        all_preds_dehz = []
        all_preds_fusion = []
        all_gts = {}
        
        import random
        max_comparisons = 5
        save_indices = set(random.sample(range(len(orig_paths)), min(max_comparisons, len(orig_paths))))
        
        # 1. 搜集所有 Ground Truth 和预测框
        for idx, img_path in enumerate(tqdm(orig_paths, desc='推理与读取标签')):
            img_id = img_path.stem
            
            # 读取标签
            label_path = Path(label_dir) / f"{img_id}.txt"
            img_gts = []
            
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue
                
            h_orig, w_orig = orig_img.shape[:2]
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c = int(parts[0])
                            x_c, y_c, w, h = map(float, parts[1:5])
                            # YOLO归一化坐标(center_x, center_y, w, h)转(x1, y1, x2, y2)实际像素
                            x1 = (x_c - w/2) * w_orig
                            y1 = (y_c - h/2) * h_orig
                            x2 = (x_c + w/2) * w_orig
                            y2 = (y_c + h/2) * h_orig
                            img_gts.append({'bbox': [x1, y1, x2, y2], 'class': c, 'matched': False})
            all_gts[img_id] = img_gts
            
            dehz_path = Path(dehz_dir) / (img_id + img_path.suffix)
            if not dehz_path.exists():
                dehz_path_alt = Path(dehz_dir) / (img_id + '_d' + img_path.suffix) # 兼容一些去雾后缀命名
                if dehz_path_alt.exists():
                    dehz_path = dehz_path_alt
            
            if dehz_path.exists():
                dehz_img = cv2.imread(str(dehz_path))
            else:
                if self.auto_dehaze and hasattr(self, 'dehazer') and self.dehazer is not None:
                    dehz_img = self.dehazer.process_opencv(orig_img)
                    Path(dehz_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dehz_path), dehz_img)
                else:
                    dehz_img = orig_img
            
            # 分别获取三种模式的预测
            preds_hazy = self.detect_with_yolo(orig_img)
            preds_dehz = self.detect_with_yolo(dehz_img)
            preds_fus = self.detect_with_fusion(orig_img, dehz_img)
            
            # 随机保存5张对比图
            if idx in save_indices:
                # 为了防止背景混乱，分别对它们各自的底图进行绘画
                img_hazy_drawn = self.draw_detections(orig_img, preds_hazy, "ORIGINAL")
                img_dehz_drawn = self.draw_detections(dehz_img, preds_dehz, "DEHAZED")
                # 融合图用去雾图做底图，更清楚，但根据需要可以使用原图也可以使用去雾图，这里使用 dehz_img
                img_fus_drawn = self.draw_detections(dehz_img, preds_fus, "FUSION (Feature-level)")
                
                # 确保高度一致
                h1, w1 = img_hazy_drawn.shape[:2]
                img_dehz_drawn = cv2.resize(img_dehz_drawn, (w1, h1))
                img_fus_drawn = cv2.resize(img_fus_drawn, (w1, h1))
                
                concat_img = np.concatenate([img_hazy_drawn, img_dehz_drawn, img_fus_drawn], axis=1)
                save_path = img_out_dir / f"comparison_{img_id}.jpg"
                cv2.imwrite(str(save_path), concat_img)

            def format_preds(preds_list, target_list):
                for p in preds_list:
                    p_copy = p.copy()
                    p_copy['image_id'] = img_id
                    if 'bbox' not in p_copy and 'box' in p_copy:
                        p_copy['bbox'] = p_copy['box']
                    if 'confidence' not in p_copy and 'conf' in p_copy:
                        p_copy['confidence'] = p_copy['conf']
                    target_list.append(p_copy)
                    
            format_preds(preds_hazy, all_preds_hazy)
            format_preds(preds_dehz, all_preds_dehz)
            format_preds(preds_fus, all_preds_fusion)
                
        # 2. 计算评估指标 (P, R, mAP50) 
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter / (area1 + area2 - inter + 1e-16)
            
        def compute_ap(recall, precision):
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
        def evaluate_preds(preds_list, name="Model", plot=False):
            aps, ps, rs = [], [], []
            aps_50_95 = []
            all_precisions, all_recalls, all_confs = [], [], []

            eval_classes = len(self.rtts_classes) if self.map_to_rtts else len(self.class_names)
            iou_thresholds = np.linspace(0.5, 0.95, 10)
            
            for c in range(eval_classes):
                c_preds = [p for p in preds_list if p['class'] == c]
                c_preds.sort(key=lambda x: x['confidence'], reverse=True)
                
                nd = len(c_preds)
                num_gt = sum(1 for gts in all_gts.values() for gt in gts if gt['class'] == c)
                
                if num_gt == 0:
                    continue
                if nd == 0:
                    aps.append(0.0); ps.append(0.0); rs.append(0.0); aps_50_95.append(0.0)
                    if plot:
                        all_precisions.append(np.array([0.0]))
                        all_recalls.append(np.array([0.0]))
                        all_confs.append(np.array([0.0]))
                    continue
                    
                tp_all = np.zeros((len(iou_thresholds), nd))
                fp_all = np.zeros((len(iou_thresholds), nd))
                confs = np.array([p['confidence'] for p in c_preds])
                
                # 为不同IoU阈值存储GT匹配状态
                gt_matched_all = {iou_idx: {img_id: [False]*len(img_gts) for img_id, img_gts in all_gts.items()} for iou_idx in range(len(iou_thresholds))}
                        
                for i, pred in enumerate(c_preds):
                    img_id = pred['image_id']
                    pred_bbox = pred['bbox']
                    best_iou = 0
                    best_gt_idx = -1
                    
                    img_gts = all_gts.get(img_id, [])
                    # Find GT with best IOU
                    for j, gt in enumerate(img_gts):
                        if gt['class'] == c:
                            iou = compute_iou(pred_bbox, gt['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                                
                    # Assign TP/FP for each IoU threshold
                    for iou_idx, iou_thresh in enumerate(iou_thresholds):
                        if best_iou >= iou_thresh and not gt_matched_all[iou_idx][img_id][best_gt_idx]:
                            gt_matched_all[iou_idx][img_id][best_gt_idx] = True
                            tp_all[iou_idx, i] = 1
                        else:
                            fp_all[iou_idx, i] = 1
                        
                # Compute AP for 0.5
                fpc = np.cumsum(fp_all[0])
                tpc = np.cumsum(tp_all[0])
                rec = tpc / num_gt
                prec = tpc / (fpc + tpc + np.finfo(float).eps)
                ap_50 = compute_ap(rec, prec)
                
                # Compute AP for 0.5:0.95
                ap_thresh = []
                for iou_idx in range(len(iou_thresholds)):
                    fpc_t = np.cumsum(fp_all[iou_idx])
                    tpc_t = np.cumsum(tp_all[iou_idx])
                    rec_t = tpc_t / num_gt
                    prec_t = tpc_t / (fpc_t + tpc_t + np.finfo(float).eps)
                    ap_thresh.append(compute_ap(rec_t, prec_t))
                ap_50_95 = np.mean(ap_thresh)
                
                if plot:
                    all_precisions.append(prec)
                    all_recalls.append(rec)
                    all_confs.append(confs)

                # 在最大 F1 处计算 P 和 R (基于 IoU=0.5)
                f1 = 2 * prec * rec / (prec + rec + np.finfo(float).eps)
                max_f1_idx = np.argmax(f1)
                best_p = prec[max_f1_idx]
                best_r = rec[max_f1_idx]
                
                aps.append(ap_50)
                aps_50_95.append(ap_50_95)
                ps.append(best_p)
                rs.append(best_r)
                
            mean_ap = np.mean(aps) if aps else 0.0
            mean_ap_50_95 = np.mean(aps_50_95) if aps_50_95 else 0.0
            mean_p = np.mean(ps) if ps else 0.0
            mean_r = np.mean(rs) if rs else 0.0
            
            print(f"[{name.ljust(15)}] P: {mean_p:.4f} | R: {mean_r:.4f} | mAP@0.5: {mean_ap:.4f} | mAP@0.5:0.95: {mean_ap_50_95:.4f}")
            
            if plot and all_precisions:
                # 为了绘制平滑的类别平均曲线，我们在固定的 recall 轴上插值
                recall_ticks = np.linspace(0, 1, 101)
                precisions_interp = []
                confs_interp = []
                f1s_interp = []
                for p, r, conf in zip(all_precisions, all_recalls, all_confs):
                    # flip因为rec是递增的
                    r_clean = np.concatenate(([0.0], r, [1.0]))
                    p_clean = np.concatenate(([1.0], p, [0.0]))
                    conf_clean = np.concatenate(([1.0], conf, [0.0]))
                    
                    # 使其单调
                    for i in range(len(p_clean) - 1, 0, -1):
                        p_clean[i - 1] = np.maximum(p_clean[i - 1], p_clean[i])

                    p_interp = np.interp(recall_ticks, r_clean, p_clean)
                    c_interp = np.interp(recall_ticks, r_clean, conf_clean)
                    f1_interp = 2 * p_interp * recall_ticks / (p_interp + recall_ticks + 1e-16)
                    
                    precisions_interp.append(p_interp)
                    confs_interp.append(c_interp)
                    f1s_interp.append(f1_interp)

                mean_precisions = np.mean(precisions_interp, axis=0)
                mean_f1s = np.mean(f1s_interp, axis=0)
                # 使用固定的置信度轴来插值F1更准确，这里简化处理，绘制 F1-Confidence
                conf_ticks = np.linspace(0, 1, 101)
                f1_by_conf = []
                for conf, f1 in zip(all_confs, [2 * p * r / (p + r + 1e-16) for p, r in zip(all_precisions, all_recalls)]):
                    # 处理前缀和
                    c_clean = conf[::-1]
                    f_clean = f1[::-1]
                    idx = np.argsort(c_clean)
                    f1_c_interp = np.interp(conf_ticks, c_clean[idx], f_clean[idx])
                    f1_by_conf.append(f1_c_interp)
                mean_f1_by_conf = np.mean(f1_by_conf, axis=0)
                
                return mean_p, mean_r, mean_ap, recall_ticks, mean_precisions, conf_ticks, mean_f1_by_conf, aps, mean_ap_50_95, aps_50_95

            return mean_p, mean_r, mean_ap, None, None, None, None, aps, mean_ap_50_95, aps_50_95

        print("\n================== 评测结果 ==================")
        metrics_hazy = evaluate_preds(all_preds_hazy, "基线模型(Hazy)", plot=True)
        metrics_dehz = evaluate_preds(all_preds_dehz, "基线模型(Dehazed)", plot=True)
        metrics_fus  = evaluate_preds(all_preds_fusion, "融合模型(Fusion)", plot=True)
        
        map_hazy = metrics_hazy[2]
        map_dehz = metrics_dehz[2]
        map_fus = metrics_fus[2]
        
        map_50_95_hazy = metrics_hazy[8]
        map_50_95_dehz = metrics_dehz[8]
        map_50_95_fus = metrics_fus[8]
        
        aps_hazy = metrics_hazy[7]
        aps_dehz = metrics_dehz[7]
        aps_fus = metrics_fus[7]

        result_txt = (
            "================== 评测结果 ==================\n"
            f"[基线模型(Hazy)     ] P: {metrics_hazy[0]:.4f} | R: {metrics_hazy[1]:.4f} | mAP@0.5: {map_hazy:.4f} | mAP@0.5:0.95: {map_50_95_hazy:.4f}\n"
            f"[基线模型(Dehazed)  ] P: {metrics_dehz[0]:.4f} | R: {metrics_dehz[1]:.4f} | mAP@0.5: {map_dehz:.4f} | mAP@0.5:0.95: {map_50_95_dehz:.4f}\n"
            f"[融合模型(Fusion)   ] P: {metrics_fus[0]:.4f}  | R: {metrics_fus[1]:.4f}  | mAP@0.5: {map_fus:.4f} | mAP@0.5:0.95: {map_50_95_fus:.4f}\n"
            "==============================================\n"
            f"融合 vs Hazy (mAP@0.5)   : 绝对提升 {map_fus - map_hazy:+.4f}\n"
            f"融合 vs Dehazed (mAP@0.5): 绝对提升 {map_fus - map_dehz:+.4f}\n"
            "==============================================\n"
        )
        print(result_txt)
        
        # 统计各类别 AP 并追加到文本中
        eval_classes = len(self.rtts_classes) if self.map_to_rtts else len(self.class_names)
        class_names_eval = self.rtts_classes if self.map_to_rtts else self.class_names
        
        class_ap_txt = "\n================== 各类别 AP@0.5 统计 ==================\n"
        class_ap_txt += f"{'Class':<15} | {'Hazy AP':<10} | {'Dehazed AP':<10} | {'Fusion AP':<10}\n"
        class_ap_txt += "-" * 55 + "\n"
        
        for i in range(eval_classes):
            c_name = class_names_eval[i]
            ap_h = aps_hazy[i] if i < len(aps_hazy) else 0.0
            ap_d = aps_dehz[i] if i < len(aps_dehz) else 0.0
            ap_f = aps_fus[i] if i < len(aps_fus) else 0.0
            class_ap_txt += f"{c_name:<15} | {ap_h:<10.4f} | {ap_d:<10.4f} | {ap_f:<10.4f}\n"
        class_ap_txt += "=" * 55 + "\n"
        
        print(class_ap_txt)
        result_txt += class_ap_txt
        
        with open(output_path / "evaluation_results.txt", "w") as f:
            f.write(result_txt)
            
        # 统计各类别的检测数量与 GT 数量
        eval_classes = len(self.rtts_classes) if self.map_to_rtts else len(self.class_names)
        class_names_eval = self.rtts_classes if self.map_to_rtts else self.class_names
        
        counts_gt = [0] * eval_classes
        counts_hazy = [0] * eval_classes
        counts_dehz = [0] * eval_classes
        counts_fus = [0] * eval_classes
        
        for gts in all_gts.values():
            for gt in gts:
                if 0 <= gt['class'] < eval_classes:
                    counts_gt[gt['class']] += 1
                    
        for p in all_preds_hazy:
            if 0 <= p['class'] < eval_classes:
                counts_hazy[p['class']] += 1
                
        for p in all_preds_dehz:
            if 0 <= p['class'] < eval_classes:
                counts_dehz[p['class']] += 1
                
        for p in all_preds_fusion:
            if 0 <= p['class'] < eval_classes:
                counts_fus[p['class']] += 1

        counts_txt = "\n================== 类别数量统计 ==================\n"
        counts_txt += f"{'Class':<15} | {'GT':<6} | {'Hazy':<6} | {'Dehazed':<8} | {'Fusion':<6}\n"
        counts_txt += "-" * 60 + "\n"
        for i in range(eval_classes):
            c_name = class_names_eval[i]
            counts_txt += f"{c_name:<15} | {counts_gt[i]:<6} | {counts_hazy[i]:<6} | {counts_dehz[i]:<8} | {counts_fus[i]:<6}\n"
        counts_txt += "-" * 60 + "\n"
        counts_txt += f"{'Total':<15} | {sum(counts_gt):<6} | {sum(counts_hazy):<6} | {sum(counts_dehz):<8} | {sum(counts_fus):<6}\n"
        counts_txt += "=" * 60 + "\n"
        
        print(counts_txt)
        with open(output_path / "evaluation_results.txt", "a") as f:
            f.write(counts_txt)
            
        print("绘制并保存各个类别的 AP 对比柱状图...")
        # 绘制类别 AP 柱状图
        x = np.arange(eval_classes)
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rects1 = ax.bar(x - width, [aps_hazy[i] if i < len(aps_hazy) else 0 for i in range(eval_classes)], width, label='Hazy', color='blue')
        rects2 = ax.bar(x, [aps_dehz[i] if i < len(aps_dehz) else 0 for i in range(eval_classes)], width, label='Dehazed', color='green')
        rects3 = ax.bar(x + width, [aps_fus[i] if i < len(aps_fus) else 0 for i in range(eval_classes)], width, label='Fusion', color='red')
        
        ax.set_ylabel('Average Precision (AP@0.5)')
        ax.set_title('AP by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_eval, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / "Class_AP_comparison.png", dpi=300)
        plt.close()
            
        print("绘制并保存 PR 曲线和 F1-Confidence 曲线...")
        # 绘制 PR 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(metrics_hazy[3], metrics_hazy[4], label=f'Hazy mAP@0.5: {map_hazy:.4f}', color='blue')
        plt.plot(metrics_dehz[3], metrics_dehz[4], label=f'Dehazed mAP@0.5: {map_dehz:.4f}', color='green')
        plt.plot(metrics_fus[3], metrics_fus[4], label=f'Fusion mAP@0.5: {map_fus:.4f}', color='red', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (all classes)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(output_path / "PR_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制 F1-Confidence 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(metrics_hazy[5], metrics_hazy[6], label='Hazy', color='blue')
        plt.plot(metrics_dehz[5], metrics_dehz[6], label='Dehazed', color='green')
        plt.plot(metrics_fus[5], metrics_fus[6], label='Fusion', color='red', linewidth=2)
        plt.xlabel('Confidence')
        plt.ylabel('F1 Score')
        plt.title('F1-Confidence Curve (all classes)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(output_path / "F1_Confidence_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"评测结果和图表已保存到目录: {output_path}")

        return map_fus

def parse_args():
    parser = argparse.ArgumentParser(description='测试Feature-level Fusion检测性能')
    parser.add_argument('--checkpoint', type=str, required=True, help='融合模型检查点')
    parser.add_argument('--yolo-weights', type=str, default=None, help='标准YOLO权重路径 (可选，用于对比)')
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--model-size', type=str, default='n', help='模型大小')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--do-eval', action='store_true', help='是否计算数据集 mAP')
    parser.add_argument('--map-to-rtts', action='store_true', help='将在VisDrone上的10类预测映射为RTTS的5类评估')
    parser.add_argument('--auto-dehaze', action='store_true', help='是否在未找到去雾图像时自动进行去雾并保存')
    parser.add_argument('--dehaze-weights', type=str, default='yolosystem/CoA/model/EMA_model/EMA_r.pth', help='去雾模型权重路径')
    parser.add_argument('--dehz-dir', type=str, default='', help='去雾图像目录(评测时)')
    parser.add_argument('--label-dir', type=str, default='', help='标签文件目录(评测时)')
    parser.add_argument('--output-dir', type=str, default='', help='输出文件夹')
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparer = DetectionComparison(
        fusion_checkpoint=args.checkpoint,
        yolo_checkpoint=args.yolo_weights,
        model_size=args.model_size,
        conf_threshold=args.conf_threshold,
        auto_dehaze=args.auto_dehaze,
        dehaze_weights=args.dehaze_weights,
        map_to_rtts=args.map_to_rtts
    )

    if args.do_eval:
        labels = args.label_dir if args.label_dir else str(Path(args.data_dir).parent / 'labels')
        dehzs = args.dehz_dir if args.dehz_dir else str(Path(args.data_dir).parent / 'images_dehazed')
        comparer.evaluate_dataset(args.data_dir, dehzs, labels, args.output_dir)
        print("评测计算完毕，相关图表图像已保存。跳过单张处理。如果要分别处理单张，请不要加 --do-eval。")
        return

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    labels = args.label_dir if args.label_dir else str(Path(args.data_dir).parent / 'labels')
    dehzs = args.dehz_dir if args.dehz_dir else str(Path(args.data_dir).parent / 'images_dehazed')
    
    # 评测全量数据集
    comparer.evaluate_dataset(str(data_dir), dehzs, labels, str(output_dir))
    print(f"\n结果和可视化已保存到: {output_dir}")

if __name__ == '__main__':
    main()
