from argparse import ArgumentParser
import os
import logging
import subprocess
import shutil
import sys

parser = ArgumentParser("Run automatic pipeline scripts in sequence.")

parser.add_argument("--convert_in_png", default=False, action='store_true', help="Convert images to PNG format. False if already in PNG.")
parser.add_argument("--scale_imgs", default=False, action='store_true', help="Scale images to 1600px . False if already scaled.")
parser.add_argument("--dataset_base_folder", type=str, required=True, help="Path to base dataset folder")
parser.add_argument("--scripts_folder", required=True, type=str, help="Path to folder containing pipeline scripts")
parser.add_argument("--conda_video_depth_anything", type=str, required=True, help="Name of the conda environment with video-depth-anything installed")
parser.add_argument("--video_depth_anything_metric_folder", type=str, required=True, help="Path to the folder containing video-depth-anything metric script")
parser.add_argument("--conda_3DGS_VR", type=str, required=True, help="Name of the conda environment with 3DGS-VR installed")
parser.add_argument("--gs_first_pass_folder", type=str, required=True, help="Path to the folder containing the first pass of the Gaussian Splatting")
parser.add_argument("--gs_second_pass_folder", type=str, required=True, help="Path to the folder containing the second pass of the Gaussian Splatting")
parser.add_argument("--conda_gs", type=str, required=True, help="Name of the conda environment with Gaussian Splatting installed")
parser.add_argument("--iterations", type=int, default=30000, help="Number of iterations for Gaussian Splatting.")
parser.add_argument("--skip_gs", default=False, action='store_true', help="Skip Gaussian Splatting. Just prepare dataset.")
parser.add_argument("--skip_prepare_dataset", default=False, action='store_true', help="Skip dataset preparation. Just run Gaussian Splatting.")
parser.add_argument("--distance_threshold", type=float, help="Distance threshold for Gaussian Splatting. Used only if skip_prepare_dataset is True.")
parser.add_argument("--dataset_base_folder_train", type=str, help="Path to the training dataset base folder. Used only if skip_prepare_dataset is True.")
parser.add_argument("--geodetic_points_filename", type=str, help="Path to the geodetic point cloud filename for 1st pass of GS. Used only if skip_prepare_dataset is True.")
parser.add_argument("--center_of_scene", type=str, help="Center of the scene for 1st pass of GS. Used only if skip_prepare_dataset is True.")
parser.add_argument("--sphere_radius", type=float, help="Radius of the sphere for 1st pass of GS. Used only if skip_prepare_dataset is True.")
parser.add_argument("--sphere_radii", type=float, help="Radii of the sphere for 1st pass of GS. Used only if skip_prepare_dataset is True.")
parser.add_argument("--scaled_inner_radius", type=float, help="Scaled inner radius for 1st pass of GS. Used only if skip_prepare_dataset is True.")

args = parser.parse_args()

scripts_folder = args.scripts_folder
convert_in_png = args.convert_in_png
dataset_base_folder = args.dataset_base_folder
scale_imgs = args.scale_imgs
conda_video_depth_anything = args.conda_video_depth_anything
video_depth_anything_metric_folder = args.video_depth_anything_metric_folder
conda_3DGS_VR = args.conda_3DGS_VR
gs_first_pass_folder = args.gs_first_pass_folder
gs_second_pass_folder = args.gs_second_pass_folder
conda_gs = args.conda_gs
iterations = args.iterations
skip_gs = args.skip_gs
skip_prepare_dataset = args.skip_prepare_dataset
threshold_value_args = args.distance_threshold
dataset_base_folder_train_args = args.dataset_base_folder_train
geodetic_points_filename_args = args.geodetic_points_filename
center_of_scene_args = args.center_of_scene
sphere_radius_args = args.sphere_radius
sphere_radii_args = args.sphere_radii
scaled_inner_radius_args = args.scaled_inner_radius

# ------------

if not skip_prepare_dataset:
    try:
        os.chdir(scripts_folder)
        print(f"Changed directory to {scripts_folder}")
    except OSError as e:
        logging.error(f"Failed to change directory to {scripts_folder}: {e}. Exiting.")
        exit(1)

    # ------------

    input_imgs_base_folder = os.path.join(dataset_base_folder, "all_imgs")
    png_images_folder = os.path.join(dataset_base_folder, "all_imgs_png") #keep here to be compatible with different modalities (eg if not converting to png)

    if convert_in_png:
        jpg_images_folder = input_imgs_base_folder
        
        os.makedirs(png_images_folder, exist_ok=True)

        convert_in_png_command = f'python .\\convert_imgs_from_jpg_to_png.py \
                                --input_folder "{jpg_images_folder}" \
                                --output_folder "{png_images_folder}"'

        exit_code = os.system(convert_in_png_command)
        if exit_code != 0:
            logging.error(f"Image conversion failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
    input_imgs_base_folder = png_images_folder

    #-----------

    scaled_images_folder = os.path.join(dataset_base_folder, "all_imgs_scaled_6k") #keep here to be compatible with different modalities (eg if not scaling)

    if scale_imgs:    
        os.makedirs(scaled_images_folder, exist_ok=True)

        scale_imgs_command = f'python .\\scale_imgs_to_1600.py \
                                --input_folder "{input_imgs_base_folder}" \
                                --output_folder "{scaled_images_folder}"'

        exit_code = os.system(scale_imgs_command)
        if exit_code != 0:
            logging.error(f"Image scaling failed with code {exit_code}. Exiting.")
            exit(exit_code)

    input_imgs_base_folder = scaled_images_folder

    #-----------

    divide_dataset_command = f'python .\\divide_dataset_20-80_random.py \
                                --input_folder "{input_imgs_base_folder}"'

    exit_code = os.system(divide_dataset_command)
    if exit_code != 0:
        logging.error(f"Dataset division failed with code {exit_code}. Exiting.")
        exit(exit_code)

    dataset_base_folder_train = os.path.join(dataset_base_folder, "train")

    #-----------

    convert_colmap_command = f'python .\\convertDeborah.py \
                                -s "{dataset_base_folder_train}" \
                                --resize'

    exit_code = os.system(convert_colmap_command)
    if exit_code != 0:
        logging.error(f"Colmap failed with code {exit_code}. Exiting.")
        exit(exit_code)

    input_imgs_base_folder = os.path.join(dataset_base_folder_train, "images")

    #----------

    try:
        os.chdir(video_depth_anything_metric_folder)
        print(f"Changed directory to {video_depth_anything_metric_folder}")
    except OSError as e:
        logging.error(f"Failed to change directory to {video_depth_anything_metric_folder}: {e}. Exiting.")
        exit(1)

    #----------

    video_DA_output_folder = os.path.join(dataset_base_folder_train, "video-depth-anything-metric", "original_from_img_seq")

    video_DA_metric_command = f'conda run -n {conda_video_depth_anything} \
                                python .\\run_on_img_sequence.py \
                                --input_images_dir "{input_imgs_base_folder}" \
                                --output_dir "{video_DA_output_folder}" \
                                --save_npz --max_res -1 --save_npz_fps -1'

    exit_code = os.system(video_DA_metric_command)
    if exit_code != 0:
        logging.error(f"Video Depth Anything metric failed with code {exit_code}. Exiting.")
        exit(exit_code)

    #----------

    try:
        os.chdir(scripts_folder)
        print(f"Changed directory to {scripts_folder}")
    except OSError as e:
        logging.error(f"Failed to change directory to {scripts_folder}: {e}. Exiting.")
        exit(1)

    #----------


    video_DA_folder = os.path.join(video_DA_output_folder, "images_depths_npz") 

    sparse_folder = os.path.join(dataset_base_folder_train, "sparse", "0")
    camera_bin_path = os.path.join(sparse_folder, "cameras.bin")
    images_bin_path = os.path.join(sparse_folder, "images.bin")
    distance_folder = os.path.join(video_DA_output_folder, "fromSceneCenter", "distances")

    distance_command = f'conda run -n {conda_3DGS_VR} \
                        python .\\camera_depth_to_scene_center_distance.py \
                        --cameras_bin_path "{camera_bin_path}" \
                        --images_bin_path "{images_bin_path}" \
                        --images_folder "{input_imgs_base_folder}" \
                        --depth_maps_folder "{video_DA_folder}" \
                        --save_folder "{distance_folder}" \
                        --silent'

    exit_code = os.system(distance_command)
    if exit_code != 0:
        logging.error(f"Convert to distance from center failed with code {exit_code}. Exiting.")
        exit(exit_code)

    #----------

    distances_threshold_folder = os.path.join(video_DA_output_folder, "fromSceneCenter", "distances_threshold")
    threshold_script_path = os.path.abspath(".\\limit_depth_maps_to_threshold.py")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_3DGS_VR,
        "python", threshold_script_path,
        "--distance_maps_folder", distance_folder,
        "--images_folder", input_imgs_base_folder,
        "--save_folder", distances_threshold_folder
    ]

    # catch stdout to get the threshold value
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=sys.stdin, 
        text=True,
        shell=True
    )

    threshold_value = None

    # read line by line
    for line in proc.stdout:
        print(line, end="")  # continue to print output
        if line.startswith("__THRESHOLD__:"):
            threshold_value = float(line.split(":")[1])

    exit_code = proc.wait()

    if exit_code != 0:
        logging.error(f"Thresholding background-foreground failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)

    if threshold_value is None:
        logging.error(f"Threshold not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using threshold value: {threshold_value}")

    #----------

    images_bg_folder = distances_threshold_folder + "_" + str(threshold_value)

    # EXECUTE BOTH DEPTH AND RANDOM INITIALIZATION
    # ---------------- DEPTH
    geodesic_script_path = os.path.abspath(".\\generate_geodesic_points_faster_depth.py")

    cmd = [
        "conda", "run", "-n", conda_3DGS_VR,
        "python", geodesic_script_path,
        "--sparse_folder", sparse_folder,
        "--images_folder", input_imgs_base_folder,
        "--images_bg_folder", images_bg_folder,
        "--dataset_base_folder", dataset_base_folder_train,
        "--dataset_distances_folder", distance_folder,
        "--object_threshold", threshold_value
    ]

    # catch stdout to get the threshold value
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=sys.stdin, 
        text=True,
        shell=True
    )

    geodetic_filename = None
    center_of_scene = None
    sphere_radius = None

    # read line by line
    for line in proc.stdout:
        print(line, end="")  # continue to print output
        if line.startswith("__GEOD_FILENAME__:"):
            geodetic_filename = line.split(":")[1].strip()
        if line.startswith("__SCENE_CENTER__:"):
            center_of_scene = line.split(":")[1].strip()
        if line.startswith("__SPHERE_RADIUS__:"):
            sphere_radius = line.split(":")[1].strip()
        if line.startswith("__SPHERE_RADII__:"):
            sphere_radii = line.split(":")[1].strip()
        if line.startswith("__SCALED_INNER_RADIUS__:"):
            scaled_inner_radius = line.split(":")[1].strip()

    exit_code = proc.wait()

    if exit_code != 0:
        logging.error(f"Generate geodesic points failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)

    if geodetic_filename is None:
        logging.error(f"Geodetic filename not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using geodetic filename: {geodetic_filename}")

    if center_of_scene is None:
        logging.error(f"Scene center not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code) 
    else:
        print(f"Using scene center: {center_of_scene}")  

    if sphere_radius is None:
        logging.error(f"Sphere radius not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using sphere radius: {sphere_radius}")
    
    '''
    if sphere_radii is None:
        logging.error(f"Sphere radii not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using sphere radii: {sphere_radii}")
    '''

    if scaled_inner_radius is None:
        logging.error(f"Scaled inner radius not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code) 
    else:
        print(f"Using scaled inner radius: {scaled_inner_radius}")




    # ---------------- RANDOM
    geodesic_script_path = os.path.abspath(".\\generate_geodesic_points_faster_on_shell.py")

    cmd = [
        "conda", "run", "-n", conda_3DGS_VR,
        "python", geodesic_script_path,
        "--sparse_folder", sparse_folder,
        "--images_folder", input_imgs_base_folder,
        "--images_bg_folder", images_bg_folder,
        "--dataset_base_folder", dataset_base_folder_train,
        "--dataset_distances_folder", distance_folder,
        "--object_threshold", threshold_value
    ]

    # catch stdout to get the threshold value
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=sys.stdin, 
        text=True,
        shell=True
        
    )

    geodetic_filename = None
    center_of_scene = None
    sphere_radius = None

    # read line by line
    for line in proc.stdout:
        print(line, end="")  # continue to print output
        if line.startswith("__GEOD_FILENAME__:"):
            geodetic_filename = line.split(":")[1].strip()
        if line.startswith("__SCENE_CENTER__:"):
            center_of_scene = line.split(":")[1].strip()
        if line.startswith("__SPHERE_RADIUS__:"):
            sphere_radius = line.split(":")[1].strip()
        if line.startswith("__SPHERE_RADII__:"):
            sphere_radii = line.split(":")[1].strip()
        if line.startswith("__SCALED_INNER_RADIUS__:"):
            scaled_inner_radius = line.split(":")[1].strip()

    exit_code = proc.wait()

    if exit_code != 0:
        logging.error(f"Generate geodesic points failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)

    if geodetic_filename is None:
        logging.error(f"Geodetic filename not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using geodetic filename: {geodetic_filename}")

    if center_of_scene is None:
        logging.error(f"Scene center not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code) 
    else:
        print(f"Using scene center: {center_of_scene}")  

    if sphere_radius is None:
        logging.error(f"Sphere radius not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using sphere radius: {sphere_radius}")
    
    '''
    if sphere_radii is None:
        logging.error(f"Sphere radii not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code)
    else:
        print(f"Using sphere radii: {sphere_radii}")
    '''

    if scaled_inner_radius is None:
        logging.error(f"Scaled inner radius not captured. Failed with code {exit_code}. Exiting.")
        sys.exit(exit_code) 
    else:
        print(f"Using scaled inner radius: {scaled_inner_radius}")

    #----------

    points3d_bin_path = os.path.join(sparse_folder, "points3D.bin")

    keep_3Dpts_command = f'conda run -n {conda_3DGS_VR} \
                            python .\\keep_only_3Dpoints_using_depth_threshold.py \
                            --points3D_bin_path "{points3d_bin_path}" \
                            --images_bin_path "{images_bin_path}" \
                            --distances_dir "{distance_folder}" \
                            --depth_threshold {threshold_value} \
                            --dataset_root_dir "{dataset_base_folder_train}"'

    exit_code = os.system(keep_3Dpts_command)
    if exit_code != 0:
        logging.error(f"Keep 3D points failed with code {exit_code}. Exiting.")
        exit(exit_code)

    #----------

#--------------------------------------- BEGIN GAUSSIAN SPLATTING FIRST AND SECOND PASS ------------------------------------------------

#--------------------------------------------------------
# DI DEFALT GS ESEGUE CON NUVOLA DI PUNTI RANDOM!!!! 
# QUELLA DEPTH SI DEVE FARE A PARTE!!!
#--------------------------------------------------------

if not skip_gs:

    if skip_prepare_dataset and threshold_value_args is None:
        logging.error("If skip_prepare_dataset is True, distance_threshold must be provided. Exiting.")
        exit(1)
    if skip_prepare_dataset and dataset_base_folder_train_args is None:
        logging.error("If skip_prepare_dataset is True, dataset_base_folder_train must be provided. Exiting.")
        exit(1) 
    if skip_prepare_dataset and geodetic_points_filename_args is None:
        logging.error("If skip_prepare_dataset is True, geodetic_points_filename must be provided. Exiting.")
        exit(1)        
    if skip_prepare_dataset and center_of_scene_args is None:
        logging.error("If skip_prepare_dataset is True, center_of_scene must be provided. Exiting.")
        exit(1)  
    if skip_prepare_dataset and sphere_radius_args is None:
        logging.error("If skip_prepare_dataset is True, sphere_radius_args must be provided. Exiting.")
        exit(1)  
    '''
    if skip_prepare_dataset and sphere_radii_args is None:
        logging.error("If skip_prepare_dataset is True, sphere_radii_args must be provided. Exiting.")
        exit(1)  
    '''
    if skip_prepare_dataset and scaled_inner_radius_args is None:
        logging.error("If skip_prepare_dataset is True, scaled_inner_radius must be provided. Exiting.")
        exit(1)

    if skip_prepare_dataset:        
        dataset_base_folder_train = dataset_base_folder_train_args
        dataset_base_folder = os.path.dirname(dataset_base_folder_train)
        threshold_value = threshold_value_args
        images_bg_folder = os.path.join(dataset_base_folder_train, "video-depth-anything-metric", "original_from_img_seq", "fromSceneCenter", "distances_threshold") + "_" + str(threshold_value)
        sparse_folder = os.path.join(dataset_base_folder_train, "sparse", "0")
        images_bin_path = os.path.join(sparse_folder, "images.bin")
        camera_bin_path = os.path.join(sparse_folder, "cameras.bin")
        input_imgs_base_folder = os.path.join(dataset_base_folder_train, "images")
        geodetic_filename = geodetic_points_filename_args        
        center_of_scene = center_of_scene_args
        sphere_radius = sphere_radius_args
        #sphere_radii = sphere_radii_args
        scaled_inner_radius = scaled_inner_radius_args
    
        print(f"Using provided dataset base folder for dataset preparation: {dataset_base_folder_train}")
        print(f"Using provided distance threshold value: {threshold_value}")
        print(f"Using provided geodetic points filename: {geodetic_filename}")
        print(f"Using provided center of the scene: {center_of_scene}")
        print(f"Using provided sphere radius: {sphere_radius}")
        #print(f"Using provided sphere radii: {sphere_radii}")
        print(f"Using provided scaled inner radius: {scaled_inner_radius}")

    #----------

    try:
        os.chdir(gs_first_pass_folder)
        print(f"Changed directory to {gs_first_pass_folder}")
    except OSError as e:
        logging.error(f"Failed to change directory to {gs_first_pass_folder}: {e}. Exiting.")
        exit(1)
    
    #----------
    
    #GS First Pass

    # Create dataset in data folder
    dataset_name = os.path.basename(os.path.normpath(dataset_base_folder))
    dataset_name_first_pass = dataset_name + f"_eval_th{threshold_value}_1stPass"

    gs_first_pass_dataset_folder = os.path.join(gs_first_pass_folder, "data", "paper", dataset_name_first_pass)
    os.makedirs(gs_first_pass_dataset_folder, exist_ok=True)

    # Copy thresholded images inside 'images' folder
    gs_first_pass_imgs_folder = os.path.join(gs_first_pass_dataset_folder, "images")
    os.makedirs(gs_first_pass_imgs_folder, exist_ok=True)

    for item in os.listdir(images_bg_folder):
        s = os.path.join(images_bg_folder, item)
        d = os.path.join(gs_first_pass_imgs_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Copy cameras.bin and images.bin inside 'sparse/0' folder
    gs_first_pass_sparse_folder = os.path.join(gs_first_pass_dataset_folder, "sparse", "0")
    os.makedirs(gs_first_pass_sparse_folder, exist_ok=True)

    shutil.copy2(images_bin_path, os.path.join(gs_first_pass_sparse_folder, "images.bin"))
    shutil.copy2(camera_bin_path, os.path.join(gs_first_pass_sparse_folder, "cameras.bin"))

    # Copy geodetic points file inside 'sparse/0' folder and rename it to points3D.ply
    geodetic_folder = os.path.join(dataset_base_folder_train, geodetic_filename)
    shutil.copy2(geodetic_folder, os.path.join(gs_first_pass_sparse_folder, "points3D.ply"))

    #----------

    gs_first_pass_output_folder = os.path.join(gs_first_pass_folder, "output", "paper", dataset_name_first_pass)

    gs_first_pass_cmd = f'conda run --no-capture-output -n {conda_gs} \
                        python .\\train.py \
                        -s "{gs_first_pass_dataset_folder}" \
                        --iterations {iterations} \
                        -m "{gs_first_pass_output_folder}" \
                        --scene_center "{center_of_scene}" \
                        --background_radius {sphere_radius} \
                        --scaled_inner_radius {scaled_inner_radius} \
                        --exposure_lr_init 0.001 \
                        --exposure_lr_final 0.0001 \
                        --exposure_lr_delay_steps 5000 \
                        --exposure_lr_delay_mult 0.001 \
                        --train_test_exp -r 1'
                        #--background_radii "{sphere_radii}" \

    exit_code = os.system(gs_first_pass_cmd)
    if exit_code != 0:
        logging.error(f"Gaussian Splatting First Pass failed with code {exit_code}. Exiting.")
        exit(exit_code)

    #----------

    try:
        os.chdir(gs_second_pass_folder)
        print(f"Changed directory to {gs_second_pass_folder}")
    except OSError as e:
        logging.error(f"Failed to change directory to {gs_second_pass_folder}: {e}. Exiting.")
        exit(1)

    #----------

    #GS Second Pass

    # Create dataset in data folder
    dataset_name_second_pass = dataset_name + f"_eval_th{threshold_value}_2ndPass"

    gs_second_pass_dataset_folder = os.path.join(gs_second_pass_folder, "data", "paper", dataset_name_second_pass)
    os.makedirs(gs_second_pass_dataset_folder, exist_ok=True)

    # Copy whole images inside 'images' folder
    gs_second_pass_imgs_folder = os.path.join(gs_second_pass_dataset_folder, "images")
    os.makedirs(gs_second_pass_imgs_folder, exist_ok=True)

    for item in os.listdir(input_imgs_base_folder):
        s = os.path.join(input_imgs_base_folder, item)
        d = os.path.join(gs_second_pass_imgs_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Copy cameras.bin and images.bin inside 'sparse/0' folder
    gs_second_pass_sparse_folder = os.path.join(gs_second_pass_dataset_folder, "sparse", "0")
    os.makedirs(gs_second_pass_sparse_folder, exist_ok=True)

    shutil.copy2(images_bin_path, os.path.join(gs_second_pass_sparse_folder, "images.bin"))
    shutil.copy2(camera_bin_path, os.path.join(gs_second_pass_sparse_folder, "cameras.bin"))

    # Copy thresholded points file inside 'sparse/0' folder and rename it to points3D.ply
    thresholded_points_filename = f"points3d_in_threshold_{threshold_value}.ply"
    thresholded_points_folder = os.path.join(dataset_base_folder_train, thresholded_points_filename)
    shutil.copy2(thresholded_points_folder, os.path.join(gs_second_pass_sparse_folder, "points3D.ply"))

    # Copy checkpoint file in root directory of second pass
    checkpoint_first_pass = os.path.join(gs_first_pass_output_folder, "gaussians_params_first_pass.pt")
    shutil.copy2(checkpoint_first_pass, os.path.join(gs_second_pass_folder, "gaussians_params_first_pass.pt"))

    #----------

    gs_second_pass_output_folder = os.path.join(gs_second_pass_folder, "output", "paper", dataset_name_second_pass)

    gs_second_pass_cmd = f'conda run --no-capture-output -n {conda_gs} \
                        python .\\train.py \
                        -s "{gs_second_pass_dataset_folder}" \
                        --iterations {iterations} \
                        -m "{gs_second_pass_output_folder}" \
                        --scene_center "{center_of_scene}" \
                        --background_radius {sphere_radius} \
                        --exposure_lr_init 0.001 \
                        --exposure_lr_final 0.0001 \
                        --exposure_lr_delay_steps 5000 \
                        --exposure_lr_delay_mult 0.001 \
                        --train_test_exp -r 1'

    exit_code = os.system(gs_second_pass_cmd)
    if exit_code != 0:
        logging.error(f"Gaussian Splatting Second Pass failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # ---------
   
# BEGIN EVALUATION AND METRICS

#----------

try:
    os.chdir(scripts_folder)
    print(f"Changed directory to {scripts_folder}")
except OSError as e:
    logging.error(f"Failed to change directory to {scripts_folder}: {e}. Exiting.")
    exit(1)

#----------

# Create folder
register_test_folder = os.path.join(dataset_base_folder, "register_test_in_train")
os.makedirs(register_test_folder, exist_ok=True)

# copy inside the test imgs
register_test_imgs_folder = os.path.join(register_test_folder, "input_test")
os.makedirs(register_test_imgs_folder, exist_ok=True)

test_imgs_folder = os.path.join(dataset_base_folder, "test")
for filename in os.listdir(test_imgs_folder):
    src_file = os.path.join(test_imgs_folder, filename)
    dst_file = os.path.join(register_test_imgs_folder, filename)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)

# copy inside the distorted folder
distorted_folder = os.path.join(dataset_base_folder_train, "distorted")
register_distorted_folder = os.path.join(register_test_folder, "distorted")
os.makedirs(register_distorted_folder, exist_ok=True)   

for root, dirs, files in os.walk(distorted_folder):
    rel_path = os.path.relpath(root, distorted_folder)
    target_root = os.path.join(register_distorted_folder, rel_path) if rel_path != '.' else register_distorted_folder
    os.makedirs(target_root, exist_ok=True)
    for file in files:
        src_file = os.path.join(root, file)
        dst_file = os.path.join(target_root, file)
        shutil.copy2(src_file, dst_file)


# Register the test images into the existing colmap model
register_train_cmd = f'conda run --no-capture-output -n {conda_3DGS_VR} \
                    python .\\register_new_images_colmap.py \
                    --exhisting_source_path "{register_test_folder}"'
exit_code = os.system(register_train_cmd)
if exit_code != 0:
    logging.error(f"Registering test images failed with code {exit_code}. Exiting.")
    exit(exit_code)


# Render test images

#----------

try:
    os.chdir(gs_first_pass_folder)
    print(f"Changed directory to {gs_first_pass_folder}")
except OSError as e:
    logging.error(f"Failed to change directory to {gs_first_pass_folder}: {e}. Exiting.")
    exit(1)

#----------

render_test_cmd = f'conda run --no-capture-output -n {conda_gs} \
                    python .\\render.py \
                    -m "{gs_second_pass_output_folder}" \
                    -s "{register_test_folder}" --render_for_metrics_debh'
exit_code = os.system(render_test_cmd)
if exit_code != 0:
    logging.error(f"Rendering test images failed with code {exit_code}. Exiting.")
    exit(exit_code)

# Run metrics

metrics_cmd = f'conda run --no-capture-output -n {conda_gs} \
                python .\\metricsDeborah.py \
                -m "{gs_second_pass_output_folder}"'

exit_code = os.system(metrics_cmd)
if exit_code != 0:
    logging.error(f"Running metrics script failed with code {exit_code}. Exiting.")
    exit(exit_code)



'''
dataset_base_folder_train = os.path.join(dataset_base_folder, "train") #da togliere
input_imgs_base_folder = os.path.join(dataset_base_folder_train, "images") #da togliere
video_DA_output_folder = os.path.join(dataset_base_folder_train, "video-depth-anything-metric", "original_from_img_seq") #da togliere
distance_folder = os.path.join(video_DA_output_folder, "fromSceneCenter", "distances") #da togliere
distances_threshold_folder = os.path.join(video_DA_output_folder, "fromSceneCenter", "distances_threshold") #da togliere
images_bin_path = os.path.join(dataset_base_folder_train, "sparse", "0", "images.bin") #da togliere
threshold_value = 40.0 #da togliere
sphere_radius = 63.6660 #da togliere
center_of_scene = "[0.42776714 0.02187429 0.16586567]" #da togliere
images_bg_folder = distances_threshold_folder + "_" + str(threshold_value) #da togliere
camera_bin_path = os.path.join(dataset_base_folder_train, "sparse", "0", "cameras.bin") #da togliere
geodetic_filename = "points3D_6subd_10radius_color.ply" #da togliere
dataset_name = os.path.basename(os.path.normpath(dataset_base_folder)) #da togliere
dataset_name_first_pass = dataset_name + f"_eval_th{threshold_value}_1stPass" #da togliere
gs_first_pass_output_folder = os.path.join(gs_first_pass_folder, "output", dataset_name_first_pass) #da togliere


dataset_base_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\caterpillar\\caterpillar"
dataset_base_folder_train = os.path.join(dataset_base_folder, "train")
conda_3DGS_VR = "3DGS-VR"
conda_gs = "gaussian-splatting-5090"
gs_second_pass_output_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\francis_eval_random_80-20-train_2ndPass_1600px_debug_postFabioLossKL5_ExpReg"
gs_first_pass_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gass-splat-first-pass-multiply"
scripts_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\10.02.25-gaussian_dome"
'''