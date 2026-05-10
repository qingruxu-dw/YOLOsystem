import kagglehub

# Download latest version
path = kagglehub.dataset_download("datasets/fusion_training")

print("Path to dataset files:", path)
