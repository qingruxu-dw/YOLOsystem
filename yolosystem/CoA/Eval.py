import os
import re
import glob
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from model import Teacher, Student, Student_x
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
import numpy as np
import cv2
import importlib
import sys
import traceback
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Try to load local CLIP once (best-effort). If missing, set clip_mod=None and skip later.
clip_mod = None
clip_model_global = None
try:
    # repo_root should be the parent directory that contains the CoA package
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    # ensure parent is on sys.path so both 'CoA.CLIP.clip' and 'CLIP.clip' can be found
    parent = os.path.abspath(os.path.join(repo_root, '..'))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    tried = []
    for modname in ('CoA.CLIP.clip', 'CLIP.clip', 'CoA.CLIP.clip as clip'):
        try:
            clip_mod = importlib.import_module(modname.split(' as ')[0])
            break
        except Exception as _e:
            tried.append((modname, str(_e)))
            clip_mod = None

    # load model to CPU if possible; tolerate any failure and continue without CLIP
    if clip_mod is not None:
        try:
            device_try = torch.device('cpu')
            download_root = os.path.join(repo_root, 'clip_model') if os.path.isdir(os.path.join(repo_root, 'clip_model')) else os.path.join(parent, 'CoA', 'clip_model')
            clip_model_global, _ = clip_mod.load('ViT-B/32', device=device_try, download_root=download_root)
            if clip_model_global is not None:
                clip_model_global.to('cpu')
                clip_model_global.eval()
        except Exception:
            clip_model_global = None
        # write a small CLIP init status snapshot for debugging
        try:
            out_dir = os.path.join(repo_root, 'outputs', 'clip_dehaze')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'clip_init_status.txt'), 'a') as stf:
                stf.write(f"clip_mod_exists={clip_mod is not None}, clip_model_loaded={clip_model_global is not None}\n")
        except Exception:
            pass
except Exception:
    clip_mod = None
    clip_model_global = None
    try:
        out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'clip_dehaze')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'clip_init_status.txt'), 'a') as stf:
            stf.write("clip_import_failed\n")
    except Exception:
        pass

# common image transform used by dehaze(); placed at module top so dehaze() can use it
transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# MODEL_PATH = './model/Teacher_model/Teacher.pth'
# OUTPUT_FOLDER = './outputs/Teacher'

# MODEL_PATH = './model/Student_model/Student.pth'
# OUTPUT_FOLDER = './outputs/Student'

MODEL_PATH = './model/EMA_model/EMA_r.pth'
OUTPUT_FOLDER = './outputs/clip_dehaze'

print(f"模型文件是否存在: {os.path.exists(MODEL_PATH)}")

# Ensure `model` exists in module globals regardless of how the file is executed.
# Some execution paths (importing as module or running via stdin) may not run the
# original __main__ block that initializes `model`, which causes NameError below.
try:
    model
except NameError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model must be initialized in main block to be visible in global scope for fallback in module level catch
    # But here we are in 'if __name__ == "__main__":'
    # The 'try...except NameError' block at module level (lines 87-97) tries to access 'model'.
    # If this file is imported, that block runs. If run as script, this main block runs.
    # To fix the lint error "undefined model", we can define it None at top level or accept it.
    
    # model = Teacher().to(device)
    # model = Student().to(device)
    model = Student_x().to(device)

    #源代码
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model.eval()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# 在 __main__ 中添加：
EVAL_MODE =False  # 设为True启用评估模式

if EVAL_MODE:
    # 路径设置
    hazy_dir = 'dataset/Haze4K/test/hazy_school'
    clear_dir = 'dataset/Haze4K/test/clear_school'  # 真正的清晰图像目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 获取配对图像（确保同名）
    hazy_images = sorted(glob.glob(os.path.join(hazy_dir, '*.jpg')))
    clear_images = []
    for h_path in hazy_images:
        # 提取数字部分（如从 "1000_0.73_1.8.png" 提取 "1000"）
        base = os.path.basename(h_path)
        num = re.match(r'^(\d+)', base).group(1)  # 匹配开头的数字
        clear_path = os.path.join(clear_dir, f"{num}.jpg")  # 拼接为 1000.jpg
        clear_images.append(clear_path)

    total_ssim, total_psnr = 0, 0
    for hazy_path, true_clear_path in zip(hazy_images, clear_images):
        # 去雾处理
        dehazed_path = os.path.join(OUTPUT_FOLDER, os.path.basename(hazy_path))
        
        try:
            # 检查去雾结果文件
            if not os.path.exists(dehazed_path):
                print(f"警告: 去雾文件未生成，跳过 {dehazed_path}")
                continue
            
        # 读取图像
            dehazed_img = Image.open(dehazed_path)
            true_clear_img = Image.open(true_clear_path)
            
            # 尺寸对齐
            if true_clear_img.size != dehazed_img.size:
                true_clear_img = true_clear_img.resize(dehazed_img.size, Image.BICUBIC)
            # 4. 计算指标（去雾结果 vs 真实清晰图）
            ssim_val, psnr_val = calculate_metrics(dehazed_path, true_clear_path)
            total_ssim += ssim_val
            total_psnr += psnr_val
            
            print(f"图像: {os.path.basename(hazy_path)} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB")
        except Exception as e:
            print(f"处理失败 {os.path.basename(hazy_path)}: {str(e)}")
            continue
    # 打印平均指标
    avg_ssim = total_ssim / len(hazy_images)
    avg_psnr = total_psnr / len(hazy_images)
    print(f"\n平均指标 | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f} dB")



    

else:
    # 仅去雾模式
    # 修改为当前路径下的路径
    INPUT_FOLDER = 'dataset/Haze4K/test/hazy_school'

    images = glob.glob(os.path.join(INPUT_FOLDER, '*jpg')) + glob.glob(os.path.join(INPUT_FOLDER, '*png')) + glob.glob(os.path.join(INPUT_FOLDER, '*jpeg'))

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"
    with torch.no_grad():
        print(f"Total images found: {len(images)}. Processing...")
        print(f"First few images: {images[:5]}")
        for image in tqdm(images, bar_format=bar_format, desc="Processing images:"):
            dehaze(model, image, OUTPUT_FOLDER)
        print(f"去雾结果已保存到: {OUTPUT_FOLDER}")
    print("处理完成！")
