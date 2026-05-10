#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=log_%j.txt
#SBATCH --partition=gpu       # 使用 gpu 分区
#SBATCH --gres=gpu:1          # 申请 1 块 GPU

# 加载必要的模块（根据集群配置调整）
module load cuda/11.8           # 加载 CUDA 11.8
module load miniforge3/24.11          # 加载 Conda

# 激活你的 Conda 环境（假设环境名为 `my_env`）
# source activate CoA
export PYTHONUNBUFFERED=1

python test_fusion_detection.py \
    --data-dir datasets/fusion_training/VisDrone2019-DET-val/images \
    --checkpoint best.pth \
    --yolo-weights runs/train_visdrone/yolo11n_baseline/weights/best.pt \
    --model-size n \
    --conf-threshold 0.001 \
    --do-eval \
    --dehz-dir datasets/fusion_training/VisDrone2019-DET-val/images_dehazed \
    --label-dir datasets/fusion_training/VisDrone2019-DET-val/labels_dehazed \
    --output-dir test_5_6 \
    --auto-dehaze \
    --dehaze-weights yolosystem/CoA/model/EMA_model/EMA_r.pth



# python test_fusion_detection.py \
#   --data-dir datasets/RTTS/images \
#   --checkpoint runs/visdrone_fusion_v2/stage2_best.pth \
#   --yolo-weights yolov11_visdrone.pt \
#   --model-size n \
#   --conf-threshold 0.001 \
#   --do-eval \
#   --dehz-dir datasets/RTTS/images_dehazed \
#   --label-dir datasets/RTTS/labels_dehazed \
#   --output-dir detection_RTTS_test \
#   --auto-dehaze \
#   --map-to-rtts \
#   --dehaze-weights yolosystem/CoA/model/EMA_model/EMA_r.pth

# 轻雾测评
# python test_fusion_detection.py \
#   --data-dir datasets/VisDrone2019-DET-val/images/beta_1.0 \
#   --label-dir datasets/fusion_training/VisDrone2019-DET-val/labels_dehazed \
#   --checkpoint runs/visdrone_fusion_v2/best.pth \
#   --yolo-weights /data/home/sczd119/run/YOLOsystem/runs/yolov10_train_visdrone/yolov10_n_visdrone/weights/best.pt \
#   --model-size n \
#   --output-dir evaluationplus_beta_rtdetr_2.0 \
#   --auto-dehaze \
#   --do-eval


# 中雾测评 (β=2.0)
# python test_fusion_detection.py \
#   --data-dir datasets/VisDrone/VisDrone2019-DET-val/images/beta_2.0 \
#   --label-dir datasets/fusion_training/VisDrone2019-DET-val/labels_dehazed \
#   --checkpoint runs/visdrone_fusion/best.pth \
#   --yolo-weights /data/home/sczd119/run/YOLOsystem/runs/train_visdrone/yolo11s_baseline_dehazed/weights/best.pt \
#   --model-size s \
#   --output-dir evaluationplus_yolov11s_dehazed_2.0 \
#   --auto-dehaze \
#   --do-eval


# 浓雾测评 (β=4.0)
# python test_fusion_detection.py \
#   --data-dir datasets/VisDrone2019-DET-val/images/beta_2.0 \
#   --label-dir datasets/fusion_training/VisDrone2019-DET-val/labels_dehazed \
#   --checkpoint runs/visdrone_fusion_v2/best.pth \
#   --yolo-weights yolov11_visdrone.pt \
#   --model-size n \
#   --output-dir test_5_6 \
#   --auto-dehaze \
#   --do-eval
  # --output-dir evaluation_beta_4.0 \



# python inference_fusion.py --checkpoint runs/best.pth --input test_images/hazy.mp4 --output outputs/result_video.mp4 --video