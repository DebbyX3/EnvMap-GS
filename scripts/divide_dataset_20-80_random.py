import os
import random
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Divide a dataset of images into two subsets (20% and 80%) randomly.")
parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing images.")
args = parser.parse_args()

# Set the path to the folder containing the images
source_dir = Path(args.input_folder)

# Destination folders
subset_test_dir = source_dir.parent / "test"
subset_train_dir = source_dir.parent / "train" / "input"

# Create destination folders if they do not exist
subset_test_dir.mkdir(exist_ok=True, parents=True)
subset_train_dir.mkdir(exist_ok=True, parents=True)

# Valid image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# Get all images in the source folder
all_images = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions]

# Shuffle and split the images
random.shuffle(all_images)
split_index = int(0.2 * len(all_images))
subset_20 = all_images[:split_index]
subset_80 = all_images[split_index:]

# Copy images to respective folders
for img in subset_20:
    shutil.copy(img, subset_test_dir / img.name)

for img in subset_80:
    shutil.copy(img, subset_train_dir / img.name)

print(f"Total images: {len(all_images)}")
print(f"Copied {len(subset_20)} images to: {subset_test_dir}")
print(f"Copied {len(subset_80)} images to: {subset_train_dir}")
