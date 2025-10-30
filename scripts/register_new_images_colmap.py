import os
import logging
from argparse import ArgumentParser
import shutil
import sys
import tempfile
import numpy as np
import collections
import struct

parser = ArgumentParser("Colmap converter")

parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--exhisting_source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0

os.makedirs(args.exhisting_source_path + "/distorted/sparse", exist_ok=True)

print(args.exhisting_source_path)

## Feature extraction
feat_extracton_cmd = colmap_command + " feature_extractor "\
    "--database_path " + "\"" + args.exhisting_source_path + "/distorted/database.db\" \
    --image_path " + "\"" + args.exhisting_source_path + "/input_test\" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model " + args.camera + " \
    --SiftExtraction.use_gpu " + str(use_gpu)

print("\n\n\n\n"+feat_extracton_cmd)
exit_code = os.system(feat_extracton_cmd)
if exit_code != 0:
    logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
    exit(exit_code)

## Feature matching
feat_matching_cmd = colmap_command + " exhaustive_matcher \
    --database_path " + "\"" + args.exhisting_source_path + "/distorted/database.db\" \
    --SiftMatching.use_gpu " + str(use_gpu)
exit_code = os.system(feat_matching_cmd)
if exit_code != 0:
    logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
    exit(exit_code)

os.makedirs(args.exhisting_source_path + "/distorted/sparse_reg_with_test", exist_ok=True)

## Image registration
registration = colmap_command + " image_registrator \
    --database_path " + "\"" + args.exhisting_source_path + "/distorted/database.db\" \
    --input_path " + "\"" + args.exhisting_source_path + "/distorted/sparse/final_cleaned\" \
    --output_path " + "\"" + args.exhisting_source_path + "/distorted/sparse_reg_with_test\""
exit_code = os.system(registration)
if exit_code != 0:
    logging.error(f"Image registration failed with code {exit_code}. Exiting.")
    exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + "\"" + args.exhisting_source_path + "/input_test\" \
    --input_path " + "\"" + args.exhisting_source_path + "/distorted/sparse_reg_with_test\" \
    --output_path " + "\"" + args.exhisting_source_path + "\" \
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Img undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.exhisting_source_path + "/sparse")
os.makedirs(args.exhisting_source_path + "/sparse/0", exist_ok=True)

# Move all files in final_cleaned except '0' into the '0' directory
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.exhisting_source_path, "sparse", file)
    destination_file = os.path.join(args.exhisting_source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)