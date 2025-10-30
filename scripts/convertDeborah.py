import os
import logging
from argparse import ArgumentParser
import shutil
import sys
import tempfile
import numpy as np
import collections
import struct
from utils.read_write_model import read_model, write_model, read_images_binary

# This Python script is based on the shell converter script provided in the 3D Gaussian Splatting repository.
parser = ArgumentParser("Colmap converter")

parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--max_reproj_error", type=float, default=2.0, help="Maximum reprojection error in pixels for image filtering")
parser.add_argument("--min_num_points", type=int, default=50, help="Minimum number of valid points for image filtering")

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:

    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + "\"" + args.source_path + "/distorted/database.db\" \
        --image_path " + "\"" + args.source_path + "/input\" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + "\"" + args.source_path + "/distorted/database.db\" \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    # ORIGINAL
    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + "\"" + args.source_path + "/distorted/database.db\" \
        --image_path "  + "\"" + args.source_path + "/input\" \
        --output_path "  + "\"" + args.source_path + "/distorted/sparse\" " \
        "--Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # Merge all the sparse model in a single one
    # If one of the merge works, then great!

    sparse_dir = os.path.join(args.source_path, "distorted", "sparse")
    models = [d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
    models.sort(key=lambda x: int(x))  # sort numerically by model number
    model_paths = [os.path.join(sparse_dir, m) for m in models]

    if len(model_paths) == 0:
        logging.error(f"No reconstructions available. Exiting.")
        exit(1)

    elif len(model_paths) == 1:
        final_path = os.path.join(sparse_dir, "final")
        shutil.copytree(model_paths[0], final_path)
        print("Only one model found. Renamed to final.")

    else: # if there are multiple models
        current_model = model_paths[0]

        for next_model in model_paths[1:]:
            temp_output = os.path.join(sparse_dir, "temp_merge")

            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)

            os.makedirs(temp_output, exist_ok=True)

            merge_cmd = colmap_command + " model_merger \
                --input_path1 " + "\"" + current_model + "\" \
                --input_path2 "  + "\"" + next_model + "\" \
                --output_path " + "\"" + temp_output + "\""
            exit_code = os.system(merge_cmd)

            if exit_code != 0:
                logging.error(f"Model Merger failed with code {exit_code}. Exiting.")
                exit(exit_code)

        # Count images in the merged model
        merged_imgs = read_images_binary(os.path.join(current_model, "images.bin"))
        merged_imgs_count = len(merged_imgs)

        # Count images in the original models
        best_model = current_model
        best_img_count = merged_imgs_count

        for m in model_paths:
            img = read_images_binary(os.path.join(m, "images.bin"))
            img_count = len(img)
            
            if img_count > best_img_count:
                best_model = m
                best_img_count = img_count

        final_path = os.path.join(sparse_dir, "final")

        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.copytree(best_model, final_path)
        print(f"\nFinal model ({best_img_count} images): {final_path}")

    temp_output = os.path.join(sparse_dir, "temp_merge")
    
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)

# CLEAN MODEL
# --- Read model ---
cameras, images, points3D = read_model(args.source_path + "/distorted/sparse/final", ext=".bin")

# --- Filter images ---
images_to_keep = {}
for img_id, img in images.items():
    # valid points observed by the image
    valid_points = [pid for pid in img.point3D_ids if pid != -1]
    num_valid = len(valid_points)
    # average reprojection error
    total_error = 0
    for pid in valid_points:
        total_error += points3D[pid].error
    avg_error = total_error / num_valid if num_valid > 0 else float("inf")

    # apply thresholds
    if avg_error <= args.max_reproj_error and num_valid >= args.min_num_points:
        images_to_keep[img_id] = img
    else:
        print(f"Discarded image {img.name}: err={avg_error:.2f}px, points={num_valid}")

# --- Filter 3D points: only those observed by valid images ---
valid_img_ids = set(images_to_keep.keys())
points3D_filtered = {}
for pid, pt in points3D.items():
    if any(img_id in valid_img_ids for img_id in pt.image_ids):
        points3D_filtered[pid] = pt

 # --- Create output folder ---
output_clean_model = os.path.join(args.source_path + "/distorted/sparse", "final_cleaned")

if os.path.exists(output_clean_model):
    shutil.rmtree(output_clean_model)
os.makedirs(output_clean_model, exist_ok=True)

# --- Save filtered model ---
write_model(cameras, images_to_keep, points3D_filtered, output_clean_model, ext=".bin")

print(f"Cleaned model saved in {output_clean_model} \n ({len(images_to_keep)} images, {len(points3D_filtered)} 3D points)")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + "\"" + args.source_path + "/input\" \
    --input_path " + "\"" + args.source_path + "/distorted/sparse/final_cleaned\" \
    --output_path " + "\"" + args.source_path + "\" \
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Img undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

# Move all files in final_cleaned except '0' into the '0' directory
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% \"" + destination_file + "\"")
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% \"" + destination_file + "\"")
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% \"" + destination_file + "\"")
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")