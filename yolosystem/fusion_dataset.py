import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class FusionDataset(Dataset):
    """
    用于融合模型(Feature-level/Image-level)训练的数据集加载器。
    它会同时读取成对的 hazy (原图/有雾图) 和 clear (去雾图) 图像，并返回 YOLO 格式的标签。
    """
    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # 根据拆分设定不同的读取路径
        if split == 'train':
            self.img_dir_hazy = self.data_dir / 'images_original'
            self.img_dir_clear = self.data_dir / 'images_dehazed'
            self.label_dir = self.data_dir / 'labels'
        else:
            # 默认 val 时的路径(需要根据你的数据集组织情况调整)
            self.img_dir_hazy = self.data_dir / 'VisDrone2019-DET-val/images'
            self.img_dir_clear = self.data_dir / 'VisDrone2019-DET-val/images_dehazed'
            self.label_dir = self.data_dir / 'VisDrone2019-DET-val/labels'
            
            # 如果没有单独的 clear 文件夹，就退退化为原始验证集
            if not self.img_dir_clear.exists():
                self.img_dir_clear = self.img_dir_hazy
        
        if not self.img_dir_hazy.exists():
            print(f"警告：找不到数据目录 {self.img_dir_hazy}")
            self.image_files = []
        else:
            self.image_files = [f for f in os.listdir(self.img_dir_hazy) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
        print(f"FusionDataset ({split}) 加载了 {len(self.image_files)} 张图像")
        
    def __len__(self):
        return len(self.image_files)

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """将图像按等比例缩放并填充黑边 (Letterbox)"""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
        
    def _load_image(self, path):
        """读取原图并返回，不在这里进行预处理以方便在 __getitem__ 中统一处理边界框"""
        img = cv2.imread(str(path))
        if img is None:
            # 找不到图像时返回空，交由外层处理
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        hazy_path = self.img_dir_hazy / img_name
        clear_path = self.img_dir_clear / img_name
        
        if not clear_path.exists():
            clear_path = hazy_path  # 当部分去雾图没有时防崩溃 fallback
        
        img_hazy_raw = self._load_image(hazy_path)
        img_clear_raw = self._load_image(clear_path)

        # -----------------------------
        #   数据增强 1: 空间同步增强 (水平翻转)
        # -----------------------------
        flip_lr = False
        if self.split == 'train' and np.random.rand() > 0.5:
            flip_lr = True
            
        if img_hazy_raw is None or img_clear_raw is None:
             # 返回黑图防崩溃
             img_hazy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
             img_clear = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
             ratio, pad = (1.0, 1.0), (0.0, 0.0)
        else:
             if flip_lr:
                 img_hazy_raw = np.fliplr(img_hazy_raw).copy()
                 img_clear_raw = np.fliplr(img_clear_raw).copy()
             
             # -----------------------------
             # 数据增强 2: HSV 颜色空间增强 (仅对 hazy 有雾图做光照的轻微扰动，不改结构)
             # 以增强模型对不同雾天天光颜色的鲁棒性
             # -----------------------------
             if self.split == 'train' and np.random.rand() > 0.5:
                 hsv = cv2.cvtColor(img_hazy_raw, cv2.COLOR_RGB2HSV).astype(np.float32)
                 # 随机轻微调整 S(饱和度) 和 V(明度)
                 s_gain = np.random.uniform(0.8, 1.2)
                 v_gain = np.random.uniform(0.8, 1.2)
                 hsv[:, :, 1] *= s_gain
                 hsv[:, :, 2] *= v_gain
                 hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                 hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                 img_hazy_raw = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

             # 进行 Letterbox 缩放，此处 scaleFill=False 保证等比例，auto=False 强制填充为指定大小
             img_hazy, ratio, pad = self._letterbox(img_hazy_raw, new_shape=(self.img_size, self.img_size), auto=False)
             img_clear, _, _ = self._letterbox(img_clear_raw, new_shape=(self.img_size, self.img_size), auto=False)
        
        img_hazy = torch.from_numpy(img_hazy).permute(2, 0, 1).float() / 255.0
        img_clear = torch.from_numpy(img_clear).permute(2, 0, 1).float() / 255.0
        
        # 2. 加载对应的 TXT 标签
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = self.label_dir / label_name
        
        labels_list = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # 读取原始 YOLO 格式 (相对坐标和宽高)
                        x_c, y_c, w, h = map(float, parts[1:5])
                        
                        # 如果是黑图情况不做标签转换
                        if img_hazy_raw is None:
                             continue
                             
                        # ---------------
                        # 处理水平翻转增强的标签更新
                        # ---------------
                        if flip_lr:
                            x_c = 1.0 - x_c

                        # 还原为原图的绝对像素坐标
                        orig_h, orig_w = img_hazy_raw.shape[:2]
                        abs_xc = x_c * orig_w
                        abs_yc = y_c * orig_h
                        abs_w = w * orig_w
                        abs_h = h * orig_h
                        
                        # 映射到 Letterbox 后的图像坐标
                        # x_new = x_old * ratio_w + pad_w
                        new_xc = abs_xc * ratio[0] + pad[0]
                        new_yc = abs_yc * ratio[1] + pad[1]
                        new_w = abs_w * ratio[0]
                        new_h = abs_h * ratio[1]
                        
                        # 转换回相对 Letterbox 图像的归一化坐标
                        norm_xc = new_xc / self.img_size
                        norm_yc = new_yc / self.img_size
                        norm_w = new_w / self.img_size
                        norm_h = new_h / self.img_size
                        
                        labels_list.append([cls_id, norm_xc, norm_yc, norm_w, norm_h])
        
        labels = torch.tensor(labels_list, dtype=torch.float32) if len(labels_list) > 0 else torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'images_hazy': img_hazy,
            'images_clear': img_clear,
            'labels': labels,
            'img_names': img_name
        }

def collate_fn(batch):
    """
    DataLoader 的拼批函数
    由于每个图片包含的边界框(labels)数量不同，不能直接用 torch.stack 去 stack labels
    """
    images_hazy = torch.stack([item['images_hazy'] for item in batch], 0)
    images_clear = torch.stack([item['images_clear'] for item in batch], 0)
    labels = [item['labels'] for item in batch]
    img_names = [item['img_names'] for item in batch]
    
    return {
        'images_hazy': images_hazy,
        'images_clear': images_clear,
        'labels': labels,
        'img_names': img_names
    }
