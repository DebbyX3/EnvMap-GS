import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Scale images to a maximum dimension of 1600 pixels while preserving aspect ratio.")
parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing images.")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder for scaled images.")
args = parser.parse_args()

# Folder with the images (change as needed)
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
MAX_SIZE = 1600  # maximum dimension in pixels

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image extensions
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

def resize_image(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size

        # If image is already within limits, save it unchanged
        if max(width, height) <= MAX_SIZE:
            img.save(output_path)
            print(f"OK: No resize needed: {os.path.basename(image_path)}")
            return

        # Calculate new size while preserving aspect ratio
        if width > height:
            new_width = MAX_SIZE
            new_height = int((MAX_SIZE / width) * height)
        else:
            new_height = MAX_SIZE
            new_width = int((MAX_SIZE / height) * width)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(output_path)
        print(f"Resized: {os.path.basename(image_path)} -> {new_width}x{new_height}")

def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(EXTENSIONS):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path)

if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
