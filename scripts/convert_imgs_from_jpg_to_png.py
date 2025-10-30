import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Convert all JPG images in a folder to PNG format.")
parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing JPG images.")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder for PNG images.")
args = parser.parse_args()

def convert_jpg_to_png(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder  # save the PNGs in the same folder

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            jpg_path = os.path.join(input_folder, filename)
            img = Image.open(jpg_path).convert("RGB")
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)
            img.save(png_path)
            print(f"Converted {filename} → {png_filename}")

convert_jpg_to_png(args.input_folder, output_folder=args.output_folder)