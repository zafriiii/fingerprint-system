# Script to create more real fingerprint images using augmentation
import os
from pathlib import Path

from PIL import Image
from torchvision import transforms

# Input and output folders
input_dir = "realval"
output_dir = "data_augmented/realval"
os.makedirs(output_dir, exist_ok=True)

# Image augmentations to make new images
transform_ops = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    ]
)

# How many images we want in total
TARGET_COUNT = 5224

# Get all input image file names
input_files = [
    f
    for f in os.listdir(input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
]

# Count how many images are already in the output folder
existing_augmented = len(
    [
        f
        for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ]
)

# Make new images until we reach the target number
idx = 0
aug_idx = 0
while existing_augmented < TARGET_COUNT:
    file = input_files[idx % len(input_files)]
    full_path = os.path.join(input_dir, file)
    img = Image.open(full_path).convert("RGB")
    aug = transform_ops(img)
    aug.save(os.path.join(output_dir, f"{Path(file).stem}_autoaug{aug_idx}.jpg"))
    existing_augmented += 1
    idx += 1
    if idx % len(input_files) == 0:
        aug_idx += 1
