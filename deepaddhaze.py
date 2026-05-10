#对收集到的数据集手动加雾
import torch
import cv2
import os
# 设置模型下载路径到您想要的目录
os.environ['TORCH_HOME'] = '/data/home/sczd119/run/YOLOsystem/addfog_model'
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
 
# 修改这里：使用适合服务器的后端
matplotlib.use('Agg')  # 使用Agg后端，适合无图形界面的服务器
from tqdm import tqdm
 
# -----------------输入图像、深度图、雾图路径--------------#
img_path = Path(r'/data/home/sczd119/run/YOLOsystem/datasets/VisDrone/VisDrone2019-DET-val/images')
depth_path = Path(r'/data/home/sczd119/run/YOLOsystem/datasets/VisDrone2019-DET-val/depth_maps')
hazy_base_path = Path(r'/data/home/sczd119/run/YOLOsystem/datasets/VisDrone2019-DET-val/images')

# 确保输出目录存在
depth_path.mkdir(parents=True, exist_ok=True)
hazy_base_path.mkdir(parents=True, exist_ok=True)

# -----------------大气散射模型参数----------------------#
# 大气光值 A (通常为0.7-1.0之间，值越大雾越白)。我们默认设置为0.85，以防止完全曝光。
A = 0.85  
# 设置你要测试的不同雾浓度 beta 值: 轻雾、中雾、浓雾
# 由于我们深度 d 映射到 0~1 的归一化浮点数了，beta数值需要跟着量级调整。
beta_list = [1.0, 2.0, 4.0] 
# ------------------------------------------------------#
#   可选择的model: 'MiDas'、'MiDaS_small'、'DPT_Hybrid'
#   'MiDas'生成的深度图比MiDaS_small精度更高，适合一般的深度估计任务
#   'DPT_Hybrid'适合复杂场景下的深度估计
# ------------------------------------------------------#
model = 'DPT_Hybrid'
 
# ------------------------生成深度图---------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载 MiDaS 模型
repo_dir = '/data/home/sczd119/run/YOLOsystem/addfog_model/hub/intel-isl_MiDaS_master'
midas = torch.hub.load(repo_dir, model, source='local')
midas_transforms = torch.hub.load(repo_dir, "transforms", source='local')
midas.to(device)
midas.eval()
 
imglist = os.listdir(img_path)
with tqdm(total=len(imglist), desc=('深度图转换')) as pbar:
    for img in imglist:
        full_path = img_path / img
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        if model == 'MiDas':
            transform = midas_transforms.default_transform
        elif model == 'DPT_Hybrid':
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        input = transform(image).to(device)
 
        with torch.no_grad():
            predict = midas(input)
        depth_map = predict.squeeze().cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # 在服务器环境下，可视化代码无法工作，请保持注释
        # # 可视化深度图
        # plt.imshow(depth_map_normalized, cmap='plasma')
        # plt.colorbar()
        # plt.title('Estimated Depth Map')
        # plt.show()
 
        depth_map = (depth_map_normalized * 255).astype(np.uint8)
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
 
        # 保存深度图
        new_filename = depth_path / img
        new_filename = new_filename.with_suffix('.jpg')  # 可以自己更改深度图的格式，默认为png
        cv2.imwrite(str(new_filename), depth_map)
 
        pbar.update(1)
 
# -----------------------生成多浓度雾图---------------------#
for beta in beta_list:
    # 为每个 beta 创建独立的文件夹
    hazy_path = hazy_base_path / f"beta_{beta}"
    hazy_path.mkdir(parents=True, exist_ok=True)
    
    with tqdm(total=len(imglist), desc=f'雾图生成 (β={beta})') as pbar:
        for filename in imglist:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(img_path, filename)
                depthmap_file = os.path.join(depth_path, Path(filename).with_suffix('.jpg').name)
     
                original_image = cv2.imread(image_path)
                depth_map = cv2.imread(depthmap_file, cv2.IMREAD_GRAYSCALE)
     
                if depth_map is None or original_image is None:
                    print(f"跳过文件: {filename}，因为无法读取对应的深度图或原始图像。")
                    continue
                
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # 将图像转换到 0~1 的浮点数以便严格遵循光学公式
                J = original_image.astype(np.float32) / 255.0
                
                # MiDaS的输出原本是视差(黑色为远，白色为近)
                # 我们将其反转，使得 0 代表近处，1 代表最远处，将深度控制在 [0, 1] 之间。
                d = (255.0 - depth_map.astype(np.float32)) / 255.0
                
                # 计算透过率 t(x) = e^(-beta * d(x))
                # 当 d 在 [0,1] 之间，如果 beta 为 1，则远处 t 约为 0.36，也就是中等雾霾
                t = np.exp(-beta * d)
                # 将下限放宽，更贴近真实物理的自然消光感 (避免直接拍平到 0.05 导致的硬边界)
                t = np.clip(t, 0.01, 1.0) 
                t_3d = t[:, :, np.newaxis]
                
                # 大气散射公式：I(x) = J(x)t(x) + A(1 - t(x))
                foggy_norm = J * t_3d + A * (1 - t_3d)
                
                # 转回 uint8 并保存
                foggy_image = np.clip(foggy_norm * 255, 0, 255).astype(np.uint8)
                foggy_image_bgr = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(hazy_path, filename)
                cv2.imwrite(output_path, foggy_image_bgr)
     
            pbar.update(1)