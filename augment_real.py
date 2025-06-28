import os
from pathlib import Path

from PIL import Image
from torchvision import transforms

input_dir = "realval"
output_dir = "data_augmented/realval"
os.makedirs(output_dir, exist_ok=True)

transform_ops = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    ]
)

TARGET_COUNT = 5224

input_files = [
    f
    for f in os.listdir(input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
]

existing_augmented = len(
    [
        f
        for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ]
)

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
