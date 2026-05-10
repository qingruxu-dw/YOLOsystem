import os
import glob

def simple_rename(folder_path):
    """简单的图片重命名函数"""
    
    # 获取所有图片文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']
    files = []
    
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
        files.extend(glob.glob(os.path.join(folder_path, f'*.{ext.upper()}')))
    
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    
    # 重命名
    for i, old_path in enumerate(files, 1):
        ext = os.path.splitext(old_path)[1]
        new_path = os.path.join(folder_path, f"{i:04d}{ext}")
        os.rename(old_path, new_path)
        print(f"重命名: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
    
    print(f"完成! 重命名了 {len(files)} 个文件")

# 使用
folder_path = "/data/home/sczd119/run/CoA/dataset/Haze4K/test/hazy_school"  # 修改为你的路径
simple_rename(folder_path)