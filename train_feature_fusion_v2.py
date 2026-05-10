"""
Feature-level Fusion YOLO 训练脚本 - 两阶段训练版本

阶段1: 冻结YOLO，只训练融合模块
阶段2: 解冻YOLO，端到端微调
"""

import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import math
import numpy as np
from ultralytics.utils.nms import non_max_suppression

# 处理路径，确保可以从子文件夹导入
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 将 yolosystem 外部路径加入以支持 yolosystem.xxx 的写法
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 优先直接从当前路径导入具体模块
try:
    from yolosystem.feature_fusion_yolo import FeatureFusionYOLO
    from yolosystem.fusion_dataset import FusionDataset, collate_fn
except ImportError:
    try:
        from feature_fusion_yolo import FeatureFusionYOLO
        from fusion_dataset import FusionDataset, collate_fn
    except ImportError as e:
        print(f"导入错误: {e}")
        print(f"当前 Python 搜索路径: {sys.path}")
        raise

try:
    from ultralytics.utils.loss import v8DetectionLoss
    HAS_YOLO_LOSS = True
except ImportError:
    v8DetectionLoss = None
    HAS_YOLO_LOSS = False

# 模拟对象，以便通过 .box 访问
class Object(dict):
    def __init__(self, *args, **kwargs):
        super(Object, self).__init__(*args, **kwargs)
        self.__dict__ = self

# 模拟列表，让 model.model[-1] 返回 Detect 层
class ModelListEmulator:
    def __init__(self, detect_module):
        self.detect = detect_module
    def __getitem__(self, index):
        return self.detect
    def __len__(self):
        return 1

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        
        # 强制配置 Ultralytics v8DetectionLoss 需要的参数对象
        self.args = Object({
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'reg_max': 16,
            'n_channels': 3,
            'iou_type': 'iou' 
        })
        self.pt = True
        
        # 提取 stride 和 nc
        # 在 FeatureFusionYOLO 中，检测头在 neck_head 的最后一个模块中
        detect_module = None
        if hasattr(model, 'neck_head'):
            for m in model.neck_head:
                if 'Detect' in str(type(m)):
                    detect_module = m
                    break
        
        if detect_module is not None and hasattr(detect_module, 'stride'):
            self.stride = detect_module.stride
            self.nc = getattr(detect_module, 'nc', 10)
        else:
            self.stride = torch.tensor([8., 16., 32.])
            self.nc = getattr(model, 'num_classes', 10)
            
        self.names = {i: f'class_{i}' for i in range(self.nc)}

        # 核心修复：v8DetectionLoss 通过 model.model[-1] 获取 Detect 层
        # 我们需要确保这个层不是 None 并且包含 stride
        class ModelListEmulator:
            def __init__(self, detect):
                self.detect = detect
            def __getitem__(self, index):
                return self.detect
            def __len__(self):
                return 1
        
        # 避开 nn.Module.__setattr__ 的注册逻辑
        object.__setattr__(self, 'model', ModelListEmulator(detect_module))


class TwoStageTrainer:
    """两阶段训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        print(f"\n创建 Feature-level Fusion (MSAFM) 模型, 类别数: {args.num_classes}...")
        self.model = FeatureFusionYOLO(
            model_size=args.model_size,
            num_classes=args.num_classes,
            fusion_layers=[2, 3, 4],  # P2/P3/P4
            pretrained=args.pretrained
        )
        self.model = self.model.to(self.device)

        # 初始化检测损失函数，用于 stage1/ stage2 的真实检测训练
        if not HAS_YOLO_LOSS or v8DetectionLoss is None:
            print("⚠ 警告: 未检测到 v8DetectionLoss，阶段训练将无法使用检测损失")
            self.loss_criterion = None
        else:
            self.loss_criterion = v8DetectionLoss(ModelWrapper(self.model))
            print("✓ v8DetectionLoss 已创建，用于检测损失计算")

        # 加载检查点（如果指定，或者自动寻找最优检查点）
        ckpt_path = args.resume_checkpoint if (args.resume_checkpoint and os.path.exists(args.resume_checkpoint)) else None
        
        # 自动恢复机制：如果未指定检查点，则寻找历史最佳
        if not ckpt_path:
            possible_ckpts = ['best.pth', 'stage2_best.pth', 'final_stage1.pth', 'stage1_best.pth']
            for name in possible_ckpts:
                candidate_path = self.output_dir / name
                if candidate_path.exists():
                    ckpt_path = str(candidate_path)
                    break

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n加载自动/指定的最优检查点: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            except Exception as e:
                if 'weights_only' in str(e) or 'Unsupported global' in str(e):
                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                else:
                    raise
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✓ 模型历史权重加载成功，避免重新训练")
            elif 'model' in checkpoint:
                # 兼容Ultralytics原生权重 (如 yolo11s.pt)
                yolo_model = checkpoint['model']
                official_state_dict = yolo_model.state_dict()
                
                # 手动映射官方模型的权重名称以匹配到我们的 FeatureFusionYOLO 网络结构
                mapped_state_dict = {}
                for k, v in official_state_dict.items():
                    # k 的格式形如: "model.0.conv.weight"
                    # 我们需要将其映射到:
                    # 对于 0~9 层(Backbone): dual_backbone.model.model.0.conv.weight
                    # 对于 10~ 结尾分(Neck+Head): neck_head.0.conv.weight
                    parts = k.split('.') # ['model', '0', 'conv', 'weight']
                    if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        if layer_idx < 10: # Backbone: 赋予双流
                            new_k = k.replace(f'model.{layer_idx}.', f'dual_backbone.model.model.{layer_idx}.')
                            mapped_state_dict[new_k] = v
                        else: # Neck & Head
                            new_layer_idx = layer_idx - 10
                            new_k = k.replace(f'model.{layer_idx}.', f'neck_head.{new_layer_idx}.')
                            mapped_state_dict[new_k] = v
                    else:
                        mapped_state_dict[k] = v
                        
                # 加入到模型中
                missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state_dict, strict=False)
                print(f"✓ 官方 YOLO 模型预训练权重【带双流映射】加载完毕")
                print(f"  --> 匹配到的键数量: {len(mapped_state_dict) - len(unexpected_keys)} / 本地总层数: {len(self.model.state_dict())}")
                if missing_keys:
                    print(f"  --> 保留未加载层 (主要是新增融合层或通道改动层): {len(missing_keys)} 个")
            else:
                self.model.load_state_dict(checkpoint, strict=False) # 尝试直接加载
                print("✓ 模型权重加载成功 (全模型加载模式)")

        # 加载数据
        print("\n加载数据集...")
        self.train_loader = self.create_dataloader('train')
        self.val_loader = self.create_dataloader('val')

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'learning_rate': []
        }

        # 确保这些属性从 args 中正确获取
        self.stage1_epochs = args.stage1_epochs
        self.stage2_epochs = args.stage2_epochs
        self.batch_size = args.batch_size

    def create_dataloader(self, split: str):
        dataset = FusionDataset(
            data_dir=self.args.data_dir,
            split=split,
            img_size=self.args.img_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.args.workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        return dataloader

    def stage1_train(self):
        """
        阶段1: 冻结 Backbone，只训练融合模块
        """
        print("\n" + "=" * 60)
        print("阶段1: 训练融合模块 (Backbone 冻结)")
        print("=" * 60)

        if self.loss_criterion is None:
            print("错误: 无法运行 stage1，因为检测损失函数未初始化。")
            return 0.0

        # 冻结双路径 Backbone 中的共享 YOLO 模型参数
        for param in self.model.dual_backbone.model.parameters():
            param.requires_grad = False
        # 冻结 Neck 和 Head
        for param in self.model.neck_head.parameters():
            param.requires_grad = False

        # 只优化融合模块参数
        optimizer = torch.optim.Adam(
            self.model.dual_backbone.fusion_modules.parameters(),
            lr=self.args.stage1_lr
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.stage1_epochs
        )

        best_loss = float('inf')

        for epoch in range(self.args.stage1_epochs):
            # 训练
            train_loss = self.train_epoch_stage1(epoch, optimizer)

            # 学习率调度
            scheduler.step()

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, 'stage1_best.pth')

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'stage1_epoch_{epoch}.pth')

        print(f"\n阶段1训练完成！最佳损失: {best_loss:.4f}")

        return best_loss

    def train_epoch_stage1(self, epoch, optimizer):
        """阶段1的训练epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch}/{self.stage1_epochs}")
        
        for i, batch in enumerate(pbar):
            # 获取数据，使用与 FusionDataset 键名匹配的 'images_hazy' 和 'images_clear'
            images_orig = batch['images_hazy'].to(self.device)
            images_clear = batch['images_clear'].to(self.device)
            
            # 准备 targets：直接使用归一化坐标（0-1），不要再乘 img_size
            target_list = []
            for b_idx, labels in enumerate(batch['labels']):
                if labels.shape[0] > 0:
                    t = torch.zeros((labels.shape[0], 6), device=self.device)
                    t[:, 0] = b_idx
                    t[:, 1] = labels[:, 0]
                    t[:, 2:] = labels[:, 1:].to(self.device)
                    target_list.append(t)

            if len(target_list) > 0:
                targets = torch.cat(target_list, 0)
            else:
                targets = torch.zeros((0, 6), device=self.device)

            if targets.shape[0] > 0:
                loss_input = {
                    'img': images_orig,
                    'cls': targets[:, 1].view(-1, 1).to(dtype=torch.float32),
                    'batch_idx': targets[:, 0].to(dtype=torch.int64),
                    'bboxes': targets[:, 2:].to(dtype=torch.float32)
                }
            else:
                loss_input = {
                    'img': images_orig,
                    'cls': torch.zeros((0, 1), device=self.device, dtype=torch.float32),
                    'batch_idx': torch.zeros((0,), device=self.device, dtype=torch.int64),
                    'bboxes': torch.zeros((0, 4), device=self.device, dtype=torch.float32)
                }

            # 前向传播并计算检测损失
            preds = self.model(images_orig, images_clear)
            loss_result = self.loss_criterion(preds, loss_input)
            if isinstance(loss_result, tuple):
                loss_vec = loss_result[0]
            else:
                loss_vec = loss_result
            loss = loss_vec.sum() / images_orig.size(0)

            # 添加融合权重正则化
            if hasattr(self.model.dual_backbone.fusion_modules, 'weight_original'):
                w_orig = torch.sigmoid(self.model.dual_backbone.fusion_modules.weight_original)
                w_dehz = torch.sigmoid(self.model.dual_backbone.fusion_modules.weight_dehazed)
                reg_loss = torch.abs(w_orig - 0.5) + torch.abs(w_dehz - 0.5)
                loss = loss + 0.01 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (i + 1):.4f}'
            })

        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'args': vars(self.args)
        }

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  保存检查点: {save_path}")

        # 将最优检查点同步单独保存为 best.pth，防止后续覆盖破坏
        if filename in ['stage2_best.pth', 'stage1_best.pth']:
            best_path = self.output_dir / 'best.pth'
            import shutil
            shutil.copy2(save_path, best_path)
            print(f"  已同步备份最优检查点至: {best_path}")

    def train(self):
        """完整训练流程"""
        start_time = time.time()

        # 阶段1
        self.stage1_train()

        # 保存最终模型
        self.save_checkpoint(self.args.stage1_epochs - 1, 'final_stage1.pth')

        elapsed = time.time() - start_time
        print(f"\n总训练时间: {elapsed / 3600:.2f} 小时")

    def stage2_train(self):
        """
        阶段2: 端到端微调所有参数
        """
        print("\n" + "=" * 60)
        print("阶段2: 端到端微调 (全部解冻)")
        print("=" * 60)

        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        if not HAS_YOLO_LOSS or v8DetectionLoss is None:
            print("错误: 无法加载 v8DetectionLoss，请检查 ultralytics 安装")
            return

        if self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.stage2_lr,
                weight_decay=1e-2
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.stage2_lr
            )

        total_epochs = self.args.stage2_epochs
        warmup_epochs = self.args.stage2_warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / max(1, warmup_epochs)
            t = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        model_wrapper = ModelWrapper(self.model)
        self.loss_criterion = v8DetectionLoss(model_wrapper)
        
        # 打印一下确认权重是否被加载（可选调试）
        print(f"Loss 权重配置已应用: Box={model_wrapper.args.box}, Cls={model_wrapper.args.cls}, DFL={model_wrapper.args.dfl}")

        best_val_map = 0.0
        patience_counter = 0

        for epoch in range(total_epochs):
            # 训练
            train_loss = self.train_epoch_stage2(epoch, optimizer)

            # 验证 mAP
            map50 = self.validate_stage2()

            # 学习率调度
            scheduler.step()

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_map'].append(map50)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证 mAP@0.5: {map50:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if map50 > best_val_map:
                best_val_map = map50
                patience_counter = 0
                self.save_checkpoint(epoch, 'stage2_best.pth')
            else:
                patience_counter += 1

            # 早停
            if self.args.early_stopping_patience > 0 and patience_counter >= self.args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement.")
                break

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'stage2_epoch_{epoch}.pth')

        print(f"\n阶段2训练完成！最佳验证 mAP@0.5: {best_val_map:.4f}")

    def validate_stage2(self):
        """阶段2验证：计算验证集上的 mAP@0.5"""
        self.model.eval()
        all_preds = []
        all_gts = {}
        img_id = 0

        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            return inter / (area1 + area2 - inter + 1e-16)

        def compute_ap(recall, precision):
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        with torch.no_grad():
            for batch_data in self.val_loader:
                images_orig = batch_data['images_hazy'].to(self.device)
                images_dehz = batch_data['images_clear'].to(self.device)
                labels_batch = batch_data['labels']

                preds = self.model(images_orig, images_dehz)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.5, max_det=300)

                batch_base_id = img_id
                for i, det in enumerate(dets):
                    current_id = batch_base_id + i
                    if det is None:
                        img_id += 1
                        continue
                    det = det.cpu().numpy()
                    for row in det:
                        x1, y1, x2, y2, conf, cls = row[:6]
                        all_preds.append({
                            'image_id': current_id,
                            'class': int(cls),
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                    img_id += 1

                for i, labels in enumerate(labels_batch):
                    current_id = batch_base_id + i
                    gts = []
                    if labels.shape[0] > 0:
                        xywh = labels[:, 1:5].cpu().numpy()
                        classes = labels[:, 0].cpu().numpy().astype(int)
                        for cls, xywh_item in zip(classes, xywh):
                            x_c, y_c, w, h = xywh_item
                            x1 = (x_c - w / 2) * self.args.img_size
                            y1 = (y_c - h / 2) * self.args.img_size
                            x2 = (x_c + w / 2) * self.args.img_size
                            y2 = (y_c + h / 2) * self.args.img_size
                            gts.append({'class': int(cls), 'bbox': [float(x1), float(y1), float(x2), float(y2)]})
                    all_gts[current_id] = gts

        eval_classes = self.args.num_classes
        if eval_classes == 0:
            return 0.0

        aps = []
        iou_thresholds = [0.5]
        for c in range(eval_classes):
            c_preds = [p for p in all_preds if p['class'] == c]
            c_preds.sort(key=lambda x: x['confidence'], reverse=True)
            num_gt = sum(1 for gts in all_gts.values() for gt in gts if gt['class'] == c)
            if num_gt == 0:
                continue
            nd = len(c_preds)
            if nd == 0:
                aps.append(0.0)
                continue

            tp = np.zeros(nd)
            fp = np.zeros(nd)
            gt_matched = {img_id: [False] * len(gts) for img_id, gts in all_gts.items()}

            for i, pred in enumerate(c_preds):
                img_gts = all_gts.get(pred['image_id'], [])
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(img_gts):
                    if gt['class'] != c:
                        continue
                    iou = compute_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= 0.5 and best_j >= 0 and not gt_matched[pred['image_id']][best_j]:
                    tp[i] = 1
                    gt_matched[pred['image_id']][best_j] = True
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (num_gt + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap = compute_ap(recall, precision)
            aps.append(ap)

        mean_ap = np.mean(aps) if aps else 0.0
        return mean_ap

    def train_epoch_stage2(self, epoch, optimizer):
        """阶段2的训练epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Stage2 Epoch {epoch}/{self.args.stage2_epochs}")
        
        for i, batch_data in enumerate(pbar):
            images_orig = batch_data['images_hazy'].to(self.device)
            images_clear = batch_data['images_clear'].to(self.device)
            
            # 使用与 stage1 相同的目标预处理逻辑构建 targets 和 loss_input
            target_list = []
            for b_idx, labels in enumerate(batch_data['labels']):
                if labels.shape[0] > 0:
                    t = torch.zeros((labels.shape[0], 6), device=self.device)
                    t[:, 0] = b_idx
                    t[:, 1] = labels[:, 0]
                    t[:, 2:] = labels[:, 1:].to(self.device)
                    target_list.append(t)

            if len(target_list) > 0:
                targets = torch.cat(target_list, 0)
            else:
                targets = torch.zeros((0, 6), device=self.device)

            if targets.shape[0] > 0:
                loss_input = {
                    'img': images_orig,
                    'cls': targets[:, 1].view(-1, 1).to(dtype=torch.float32),
                    'batch_idx': targets[:, 0].to(dtype=torch.int64),
                    'bboxes': targets[:, 2:].to(dtype=torch.float32)
                }
            else:
                loss_input = {
                    'img': images_orig,
                    'cls': torch.zeros((0, 1), device=self.device, dtype=torch.float32),
                    'batch_idx': torch.zeros((0,), device=self.device, dtype=torch.int64),
                    'bboxes': torch.zeros((0, 4), device=self.device, dtype=torch.float32)
                }

            # 前向传播与损失计算
            preds = self.model(images_orig, images_clear)
            loss_result = self.loss_criterion(preds, loss_input)
            if isinstance(loss_result, tuple):
                loss_vec = loss_result[0]
            else:
                loss_vec = loss_result
            
            # 使用 sum() 对每张图的损失求平均
            loss = loss_vec.sum() / images_orig.size(0)

            optimizer.zero_grad()
            loss.backward()
            
            # (可选): 梯度裁剪，防止偶尔损失爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (i + 1):.4f}'
            })

        return total_loss / len(self.train_loader)

    def resume_training(self, checkpoint_path: str):
        """从检查点恢复训练"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            if 'weights_only' in str(e) or 'Unsupported global' in str(e):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            else:
                raise
                
        # 兼容单独恢复情况
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            yolo_model = checkpoint['model']
            self.model.load_state_dict(yolo_model.state_dict(), strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        # 恢复历史记录
        self.history = checkpoint.get('history', self.history)

        # 恢复优化器状态
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state is not None and hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception:
                pass


        epoch_record = checkpoint.get('epoch', 0)
        print(f"从 {checkpoint_path} 恢复或微调，基准轮次: {epoch_record}")


def parse_args():
    parser = argparse.ArgumentParser(description='Feature-level Fusion YOLO 两阶段训练')

    # 数据参数
    parser.add_argument('--data-dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--num-classes', type=int, default=10, help='类别数量')

    # 模型参数
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO模型大小')
    parser.add_argument('--fusion-type', type=str, default='learned',
                        choices=['average', 'learned', 'attention'], help='融合类型')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练权重')

    # 阶段1训练参数
    parser.add_argument('--stage1-epochs', type=int, default=30, help='阶段1训练轮数')
    parser.add_argument('--stage1-lr', type=float, default=0.001, help='阶段1学习率')

    # 阶段2训练参数
    parser.add_argument('--do-stage2', action='store_true', help='是否执行阶段2训练')
    parser.add_argument('--stage2-epochs', type=int, default=50, help='阶段2训练轮数')
    parser.add_argument('--stage2-lr', type=float, default=0.0001, help='阶段2学习率')
    parser.add_argument('--stage2-warmup-epochs', type=int, default=5, help='阶段2 warmup 轮数')
    parser.add_argument('--early-stopping-patience', type=int, default=1000, help='阶段2验证早停 patience')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='阶段2优化器')

    # 通用训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/two_stage_training',
                        help='输出目录')

    # 恢复训练参数
    parser.add_argument('--resume-checkpoint', type=str, default='', help='恢复训练的检查点路径')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature-level Fusion YOLO 两阶段训练")
    print("=" * 60)

    # 创建训练器
    trainer = TwoStageTrainer(args)

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"阶段1 epochs: {args.stage1_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"阶段1学习率: {args.stage1_lr}")
    print(f"输出目录: {args.output_dir}")

    # 如果指定了恢复检查点，则加载并恢复训练
    if args.resume_checkpoint:
        trainer.resume_training(args.resume_checkpoint)
    else:
        trainer.train()

    # 如果选择执行阶段2，则开始阶段2训练
    if args.do_stage2:
        trainer.stage2_train()

    print("\n训练完成！")


if __name__ == '__main__':
    main()
