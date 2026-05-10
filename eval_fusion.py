import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from yolosystem.feature_fusion_yolo import FeatureFusionYOLO, create_feature_fusion_yolo
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
import yaml


def make_visdrone_val_yaml(val_dir: str, subset: str = 'images') -> str:
    """在 val_dir 下创建缺失的 VisDrone 单路 YAML 配置。"""
    # 允许使用潜在的子目录名（images, images_dehazed, clear）
    assert subset in ['images', 'images_dehazed', 'clear'], "subset 只能是 'images', 'images_dehazed' 或 'clear'"
    subset_path = os.path.join(val_dir, subset)
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"找不到用作 val 的图像子目录: {subset_path}，请检查是否存在。")

    data_yaml = os.path.join(val_dir, f"visdrone_val_{subset}.yaml")
    yaml_content = {
        'path': val_dir,
        'train': 'images',
        'val': subset,
        'nc': 10,
        'names': {
            0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
            5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
        }
    }
    with open(data_yaml, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"已创建/更新评估配置文件: {data_yaml}")
    return data_yaml


def evaluate_baseline(model_path, data_yaml, img_size=640, conf=0.001, iou=0.6):
    """评估基线 YOLOv11 模型。"""
    print(f"\n开始评估基线模型: {model_path} on {data_yaml}")
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, conf=conf, iou=iou, plots=True)
    return results


def evaluate_fusion(model_path, val_dir, num_classes=10, baseline_model=None, img_size=640, conf_thres=0.001, iou_thres=0.6):
    """评估融合模型。"""
    print(f"\n开始评估融合模型: {model_path}")
    if baseline_model is None:
        raise ValueError("请通过参数 baseline_model 指定一个 YOLO baseline 模型路径")

    import copy
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import IterableSimpleNamespace

    class DualInputValidator(DetectionValidator):
        def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
            super().__init__(dataloader, save_dir, pbar, args, _callbacks)
            self.fusion_model = None
            self.batch_dehz_cached = None

        def preprocess(self, batch):
            batch = super().preprocess(batch)
            batch_dehz = batch['img'].clone()
            target_h, target_w = batch['img'].shape[2:]

            for i, file_path in enumerate(batch['im_file']):
                if 'images_dehazed' in file_path or 'clear' in file_path:
                    dehz_path = file_path
                elif 'hazy' in file_path:
                    cand1 = file_path.replace('hazy', 'clear')
                    cand2 = file_path.replace('hazy', 'images_dehazed')
                    dehz_path = cand1 if os.path.exists(cand1) else cand2 if os.path.exists(cand2) else file_path
                elif 'images' in file_path:
                    cand1 = file_path.replace('images', 'clear')
                    cand2 = file_path.replace('images', 'images_dehazed')
                    if os.path.exists(cand1):
                        dehz_path = cand1
                    elif os.path.exists(cand2):
                        dehz_path = cand2
                    else:
                        dehz_path = file_path
                else:
                    dehz_path = file_path

                if os.path.exists(dehz_path):
                    im = cv2.imread(dehz_path)
                    if im is None:
                        print(f"[Warning] 无法读取 dehazed 图像: {dehz_path}, 跳过该样本")
                        continue

                    h0, w0 = im.shape[:2]
                    r = min(target_h / h0, target_w / w0)
                    new_unpad = int(round(w0 * r)), int(round(h0 * r))
                    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]
                    dw /= 2
                    dh /= 2

                    if (w0, h0) != new_unpad:
                        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

                    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

                    im = im.transpose((2, 0, 1))[::-1]
                    im_tensor = torch.from_numpy(np.ascontiguousarray(im)).to(device=batch['img'].device, dtype=batch['img'].dtype) / 255.0
                    batch_dehz[i] = im_tensor
                else:
                    print(f"[Warning] 无法找到去雾图像: {dehz_path}.")

            self.batch_dehz_cached = batch_dehz
            if self.fusion_model is not None:
                self.fusion_model._current_dehz = batch_dehz
            return batch

        def predict(self, batch):
            # Fusion model 需要两路输入
            dehz = getattr(self.fusion_model, '_current_dehz', batch['img'].clone())
            return self.fusion_model(batch['img'], dehz)

    # 加载混合模型
    model_yolo = YOLO(baseline_model)
    fusion_model_instance = create_feature_fusion_yolo(model_size='n', num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint if not hasattr(checkpoint, 'state_dict') else checkpoint.state_dict()))
    fusion_model_instance.load_state_dict(state_dict, strict=False)
    fusion_model_instance.eval()
    fusion_model_instance.to(model_yolo.device)

    class_names = {
        0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
        5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
    }

    # 兼容 Ultralytics AutoBackend 属性
    if not hasattr(fusion_model_instance, 'yaml'):
        fusion_model_instance.yaml = {'ch': 3, 'nc': num_classes}
    if not hasattr(fusion_model_instance, 'pt'):
        fusion_model_instance.pt = True
    if not hasattr(fusion_model_instance, 'stride'):
        fusion_model_instance.stride = torch.tensor([8, 16, 32]).to(model_yolo.device)
    fusion_model_instance.names = class_names

    # 如果 AutoBackend 触发 fuse()，则直接返回自身
    if not hasattr(fusion_model_instance, 'fuse'):
        fusion_model_instance.fuse = lambda *args, **kwargs: fusion_model_instance

    # Polymorphic forward
    original_forward = fusion_model_instance.forward

    def custom_forward(x, *args, **kwargs):
        dehz_x = getattr(fusion_model_instance, '_current_dehz', x.clone())
        return original_forward(x, dehz_x)

    fusion_model_instance.forward = custom_forward
    model_yolo.model = fusion_model_instance
    model_yolo.model.names = {i: str(i) for i in range(num_classes)}
    model_yolo.model.stride = torch.tensor([8, 16, 32]).to(model_yolo.device)

    args = IterableSimpleNamespace(**{
        'model': model_path,
        'data': make_visdrone_val_yaml(val_dir, 'images'),
        'imgsz': img_size,
        'conf': conf_thres,
        'iou': iou_thres,
        'plots': True
    })

    # 通过自定义 validator 注入双输入逻辑
    def custom_validator(**kwargs):
        if isinstance(kwargs.get('args'), dict):
            val_args = IterableSimpleNamespace(**kwargs['args'])
        else:
            val_args = kwargs.get('args', args)

        validator = DualInputValidator(dataloader=kwargs.get('dataloader'), save_dir=kwargs.get('save_dir'), pbar=kwargs.get('pbar'), args=val_args, _callbacks=kwargs.get('_callbacks'))
        validator.fusion_model = fusion_model_instance
        return validator

    model_yolo.ValidatorClass = custom_validator

    print("\n========== 开始使用 YOLO 官方验证器评估融合模型 ==========")
    results = model_yolo.val(**vars(args))
    return results


if __name__ == '__main__':
    baseline_model = '/data/home/sczd119/run/YOLOsystem/yolov11_visdrone.pt'
    fusion_model = '/data/home/sczd119/run/YOLOsystem/runs/best.pth'
    val_dir = '/data/home/sczd119/run/YOLOsystem/datasets/fusion_training/VisDrone2019-DET-val'

    yaml_hazy = make_visdrone_val_yaml(val_dir, subset='images')

    if os.path.isdir(os.path.join(val_dir, 'images_dehazed')):
        yaml_dehazed = make_visdrone_val_yaml(val_dir, subset='images_dehazed')
        dehazed_sub = 'images_dehazed'
    elif os.path.isdir(os.path.join(val_dir, 'clear')):
        yaml_dehazed = make_visdrone_val_yaml(val_dir, subset='clear')
        dehazed_sub = 'clear'
    else:
        raise FileNotFoundError('找不到 images_dehazed 或 clear 子目录，请先生成去雾图像')

    print('='*60)
    baseline_hazy_results = evaluate_baseline(baseline_model, yaml_hazy)
    baseline_dehazed_results = evaluate_baseline(baseline_model, yaml_dehazed)

    print('='*60)
    fusion_results = evaluate_fusion(fusion_model, val_dir, num_classes=10, baseline_model=baseline_model)

    print('\n' + '='*60)
    b_hazy_map50 = baseline_hazy_results.results_dict['metrics/mAP50(B)']
    b_dehazed_map50 = baseline_dehazed_results.results_dict['metrics/mAP50(B)']
    f_map50 = fusion_results.results_dict['metrics/mAP50(B)']

    print(f"基线模型(hazy) mAP50: {b_hazy_map50:.4f}")
    print(f"基线模型({dehazed_sub}) mAP50: {b_dehazed_map50:.4f}")
    print(f"融合模型 mAP50: {f_map50:.4f}")
    print(f"融合提升 vs hazy: {f_map50 - b_hazy_map50:+.4f} ({(f_map50 - b_hazy_map50)/b_hazy_map50*100:+.2f}%)")
    print(f"融合提升 vs {dehazed_sub}: {f_map50 - b_dehazed_map50:+.4f} ({(f_map50 - b_dehazed_map50)/max(1e-6,b_dehazed_map50)*100:+.2f}%)")
    print('='*60)
