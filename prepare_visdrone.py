import os
from pathlib import Path
import shutil
import zipfile

from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM
from PIL import Image

def visdrone2yolo(dir_path, split, source_name=None):
    """Convert VisDrone annotations to YOLO format with images/{split} and labels/{split} structure."""
    source_dir = dir_path / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = dir_path / "images" / split
    labels_dir = dir_path / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images to new structure
    source_images_dir = source_dir / "images"
    if source_images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in source_images_dir.glob("*.jpg"):
            # Move image if not already there
            dest_img = images_dir / img.name
            if not dest_img.exists():
                img.rename(dest_img)

    annotations_dir = source_dir / "annotations"
    if annotations_dir.exists():
        for f in TQDM(list(annotations_dir.glob("*.txt")), desc=f"Converting {split}"):
            img_path = images_dir / f.with_suffix(".jpg").name
            if not img_path.exists():
                continue
                
            img_size = Image.open(img_path).size
            dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
            lines = []

            with open(f, encoding="utf-8") as file:
                for row in [x.split(",") for x in file.read().strip().splitlines()]:
                    if len(row) >= 6 and row[4] != "0":  # Skip ignored regions
                        x, y, w, h = map(int, row[:4])
                        cls = int(row[5]) - 1
                        if cls < 0 or cls > 9: # Skip classes out of 0-9 range
                            continue
                            
                        # Convert to YOLO format [x_center, y_center, width, height] normalized
                        x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                        w_norm, h_norm = w * dw, h * dh
                        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            with open(labels_dir / f.name, "w", encoding="utf-8") as out_file:
                out_file.write("".join(lines))


def main():
    # 设置下载的目标路径
    dataset_root = Path("/data/home/sczd119/run/YOLOsystem/datasets/VisDrone")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading VisDrone to {dataset_root}...")

    # urls
    urls = [
        f"{ASSETS_URL}/VisDrone2019-DET-train.zip",
        f"{ASSETS_URL}/VisDrone2019-DET-val.zip",
        f"{ASSETS_URL}/VisDrone2019-DET-test-dev.zip"
    ]
    
    # Download
    download(urls, dir=dataset_root, threads=4)

    # Convert
    splits = {
        "VisDrone2019-DET-train": "train", 
        "VisDrone2019-DET-val": "val", 
        "VisDrone2019-DET-test-dev": "test"
    }
    
    # 自动解压并使用
    for folder, split in splits.items():
        zip_path = dataset_root / f"{folder}.zip"
        if zip_path.exists() and not (dataset_root / folder).exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_root)

    for folder, split in splits.items():
        print(f"Processing {folder} -> {split}...")
        visdrone2yolo(dataset_root, split, folder)  
        
        # cleanup original directory after conversion
        folder_path = dataset_root / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)  

    print("VisDrone dataset preparation completed successfully!")

if __name__ == "__main__":
    main()