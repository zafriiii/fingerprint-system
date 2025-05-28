
import os
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import random

# Input/output folders
input_dir = "data/train/spoof"
output_dir = "data_augmented/train/spoof"
os.makedirs(output_dir, exist_ok=True)

# Augmentation transforms
transform_ops = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
])

def augment_image(file_path, out_path, count=5):
    img = Image.open(file_path).convert("RGB")
    for i in range(count):
        aug = transform_ops(img)
        aug.save(os.path.join(out_path, f"{Path(file_path).stem}_aug{i}.jpg"))

# Apply to each image
for file in os.listdir(input_dir):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        full_path = os.path.join(input_dir, file)
        augment_image(full_path, output_dir, count=5)
