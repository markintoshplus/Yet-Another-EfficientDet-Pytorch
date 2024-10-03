import numpy as np
from PIL import Image
import os

def calculate_mean_std(dataset_path):
    means = []
    stds = []
    
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(dataset_path, split)
        for img_name in os.listdir(split_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(split_path, img_name)
                img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))
    
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    return mean, std

# Use the correct path to your TACO dataset
dataset_path = 'datasets/taco'
mean, std = calculate_mean_std(dataset_path)
print(f"Mean: {mean}")
print(f"Std: {std}")
