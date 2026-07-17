# EnvMap-GS: Two-Stage Outdoor Gaussian Reconstruction with Background-to-Environment Map Baking

Official scripts for the paper:

> **EnvMap-GS: two-stage outdoor Gaussian reconstruction with background-to-environment map baking**
> Deborah Pintani, Ariel Caputo, Noah Lewis, Marc Stamminger, Fabio Pellacini, Andrea Giachetti
> *The Visual Computer*, 2026 — [DOI: 10.1007/s00371-026-04593-w](https://doi.org/10.1007/s00371-026-04593-w)

> [!NOTE]  
> Presented at Computer Graphics International 2026 (CGI2026) \
> Best Presentation Award

EnvMap-GS is a two-pass Gaussian Splatting pipeline for "inside-out" outdoor captures, i.e. scenes where the camera moves in a small, bounded area but looks out towards a much larger environment (sky, mountains, distant buildings, etc.). The scene is split into a **foreground** area (the navigation area, reconstructed with standard 3DGS) and a **background** area, initialized on a geodesic sphere and constrained inside a spherical shell during optimization. This removes the "floaters" that normally appear when standard Gaussian Splatting tries to model very distant, low-parallax content. As a bonus, the background Gaussians can be baked into a classic equirectangular environment map, which is much cheaper to render and works fine for VR navigation.

If you just want to run everything end to end, jump to [Running the full pipeline automatically](#running-the-full-pipeline-automatically). If you want to understand (or run) the pipeline step by step, see [Running the pipeline manually](#running-the-pipeline-manually), which follows exactly what `automatic_pipeline.py` does internally.

### What's in this repo

> [!WARNING]  
> This readme is NOT complete! \
> Some of the commands and instructions are outdated and a general rewriting is needed. \
> Take the instructions below with a grain of salt.

Besides the pre/post-processing scripts described below, the repository also includes the two Gaussian Splatting codebases used for training: **`gaussian-splatting-first-pass/`** (background) and **`gaussian-splatting-second-pass/`** (foreground). Both are forks of the [original 3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting), extended with a few extra command-line arguments (`--scene_center`, `--background_radius`, `--scaled_inner_radius`) and with the shell/planarity losses and custom pruning described in the paper. Since they derive directly from the official codebase, they are installed exactly like it — see [Installing the two Gaussian Splatting environments](#installing-the-two-gaussian-splatting-environments) — the only difference is that you need **two separate conda environments**, one per pass, since the two forks may drift apart over time.

## Pipeline overview

The pipeline has three main phases:

1. **Pre-processing**: run COLMAP to get camera poses and a sparse point cloud, estimate metric depth maps, and use them to separate the input images into a foreground region and a background region.
2. **Two-pass Gaussian Splatting**:
   - **Pass 1** optimizes only the background Gaussians, initialized on a geodesic sphere and constrained inside a spherical shell (`Ri` to `Ro`).
   - **Pass 2** adds the foreground Gaussians (initialized from the filtered COLMAP point cloud) on top of the fixed background, and optimizes them with the standard rendering loss.
3. **Environment map baking (optional)**: the background Gaussians from Pass 1 can be rendered into an equirectangular environment map, which can replace the background Gaussians entirely for faster rendering (e.g. on VR headsets).

<details>
<summary><strong>Repository structure</strong> (click to expand)</summary>

```
.
├── automatic_pipeline.py                      # runs the whole pipeline end to end
├── convertDeborah.py                          # COLMAP reconstruction (based on the official 3DGS converter)
├── convert_imgs_from_jpg_to_png.py             # optional: JPG -> PNG conversion
├── scale_imgs_to_1600.py                       # optional: rescale images (max side 1600 px)
├── divide_dataset_20-80_random.py              # random 80/20 train/test split
├── camera_depth_to_scene_center_distance.py    # builds per-pixel "distance from scene center" maps
├── limit_depth_maps_to_threshold.py            # foreground/background thresholding on the distance maps
├── keep_only_3Dpoints_using_depth_threshold.py # filters the COLMAP point cloud to the foreground only
├── generate_geodesic_points_faster_on_shell.py # background point cloud init. (random radius in the shell)
├── generate_geodesic_points_faster_depth.py    # background point cloud init. (radius from depth, unused by default)
├── register_new_images_colmap.py               # registers test images into the existing COLMAP model
├── align_gs_and_cameras_with_world_axis.py      # utility: aligns Gaussians/cameras to the world axes (e.g. for Unity)
├── remove_env_gs_from_2nd_pass.py              # utility: removes background Gaussians duplicated in the 2nd pass PLY
├── 360-gs-scripts/
│   └── render_spherical_ply.py                 # renders a 360°/equirectangular image from a trained PLY
├── utils/
│   └── read_write_model.py                     # COLMAP model I/O (from the official COLMAP repo)
├── gaussian-splatting-first-pass/               # 3DGS fork used to train the background (Pass 1)
└── gaussian-splatting-second-pass/              # 3DGS fork used to train the foreground (Pass 2)
```

</details>

## Requirements

- [COLMAP](https://colmap.github.io/) installed and available in your `PATH` (or pass its path with `--colmap_executable`).
- Python 3.9+ with the usual scientific stack (`numpy`, `opencv-python`, `open3d`, `plyfile`, `scipy`, `Pillow`, ...) for the scripts in this repo.
- A working copy of [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) (metric variant), in its own conda environment, used to estimate metric depth maps.
- The two Gaussian Splatting codebases included in this repo, each installed in its **own** conda environment (see below).
- A recent GPU (the paper's experiments were run on an RTX 5090).

`automatic_pipeline.py` expects each of these components to live in its own conda environment (video-depth-anything, the utility environment used for the scripts in this repo, and the Gaussian Splatting environment — see `--conda_video_depth_anything`, `--conda_3DGS_VR`, `--conda_gs`). You pass the environment names on the command line, and the script handles switching between them with `conda run`. Note that the same `--conda_gs` environment is used for both passes in `automatic_pipeline.py` by default; if you keep the two codebases in separate environments instead, just run the corresponding steps manually (see [Running the pipeline manually](#running-the-pipeline-manually)) with the right environment for each.

### Installing the two Gaussian Splatting environments

`gaussian-splatting-first-pass/` and `gaussian-splatting-second-pass/` are both forks of the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo, so they are installed the same way as the original codebase — just once per folder, ideally in two separate conda environments:

```bash
# first pass
cd gaussian-splatting-first-pass
conda env create --file environment.yml   # creates the "gaussian_splatting" env, rename/adjust as needed
conda activate gaussian_splatting
# (if the submodules were not already built by environment.yml)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# second pass (repeat in a separate environment)
cd ../gaussian-splatting-second-pass
conda env create --file environment.yml -n gaussian_splatting_2nd_pass
conda activate gaussian_splatting_2nd_pass
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Both forks add a few extra command-line arguments to `train.py`/`render.py` on top of the original ones, used throughout this README:

- `--scene_center` — world-space coordinates of the scene center `O`
- `--background_radius` — outer shell radius `Ro` (1st pass) / inner radius used to bound the foreground (2nd pass)
- `--scaled_inner_radius` — inner shell radius `Ri`, scaled to the COLMAP scale

and implement the shell/planarity losses and the custom pruning strategy described in Section 3.2 of the paper.

## Running the full pipeline automatically

`automatic_pipeline.py` chains all the steps described below, from raw images to trained model, rendered test views and metrics. A minimal run looks like this:

```bash
python automatic_pipeline.py \
    --dataset_base_folder "/path/to/dataset/my_scene" \
    --scripts_folder "/path/to/this/repo" \
    --conda_video_depth_anything video-depth-anything \
    --video_depth_anything_metric_folder "/path/to/Video-Depth-Anything" \
    --conda_3DGS_VR 3DGS-VR \
    --gs_first_pass_folder "/path/to/gaussian-splatting-first-pass" \
    --gs_second_pass_folder "/path/to/gaussian-splatting-second-pass" \
    --conda_gs gaussian-splatting \
    --infinity_threshold 200 \
    --convert_in_png --scale_imgs
```

`dataset_base_folder` must contain an `all_imgs` subfolder with the raw input images (or a video already split into frames). Everything else (train/test split, COLMAP model, depth maps, background/foreground images, first-pass/second-pass dataset folders, trained models...) is created automatically inside `dataset_base_folder` and the two Gaussian Splatting folders.

Useful flags:

- `--convert_in_png` / `--scale_imgs`: skip these if your images are already PNG / already at the target resolution.
- `--infinity_threshold`: the "at infinity" distance `Ro` used to place points on the sky sphere (see paper, Section 3.2.1). Distances beyond this value are considered infinite (e.g. sky/mountains).
- `--distance_threshold`: manually fix the foreground/background threshold `Ri` instead of using the automatic heuristic (see paper, Section 3.1.2).
- `--iterations`: number of Gaussian Splatting iterations per pass (default 30000).
- `--skip_prepare_dataset`, `--skip_change_thresholds`, `--skip_gs`, `--skip_registration_in_train`: skip individual phases if you already computed them and just want to re-run a later part (in that case you'll need to also pass the values that phase would otherwise have produced, e.g. `--distance_threshold`, `--geodetic_points_filename`, `--center_of_scene`, `--sphere_radius`, `--scaled_inner_radius`, `--registration_folder` — check the argument list at the top of the script for details).

The script prints its progress step by step and stops immediately if any sub-step fails, so it's easy to see which stage to resume from.

## Running the pipeline manually

If you'd rather run (or debug) each step yourself, here is exactly what `automatic_pipeline.py` does under the hood, in order. All commands assume you are inside the folder containing these scripts, and that `conda_3DGS_VR` is a conda environment with the dependencies of this repo installed (open3d, plyfile, opencv, ...). Paths and thresholds below are placeholders — swap them with the actual paths/values of your run.

### 1. (Optional) Convert images to PNG

```bash
python convert_imgs_from_jpg_to_png.py \
    --input_folder "dataset/all_imgs" \
    --output_folder "dataset/all_imgs_png"
```

### 2. (Optional) Rescale images to 1600 px on the longest side

```bash
python scale_imgs_to_1600.py \
    --input_folder "dataset/all_imgs_png" \
    --output_folder "dataset/all_imgs_scaled"
```

### 3. Split the dataset into train (80%) and test (20%)

```bash
python divide_dataset_20-80_random.py \
    --input_folder "dataset/all_imgs_scaled"
```

This creates `dataset/train` and `dataset/test` folders next to the input folder.

### 4. Run COLMAP on the training images

```bash
python convertDeborah.py \
    -s "dataset/train" \
    --resize
```

This is a slightly modified version of the converter script shipped with the official 3DGS repo: it runs feature extraction, matching, sparse reconstruction and undistortion, and also filters out cameras/points with a high reprojection error (`--max_reproj_error`, `--min_num_points`). It writes the usual `sparse/0` COLMAP model and an `images` folder inside `dataset/train`.

### 5. Estimate metric depth maps

This step uses Video-Depth-Anything (metric), run from its own environment/folder:

```bash
conda run -n video-depth-anything python run_on_img_sequence.py \
    --input_images_dir "dataset/train/images" \
    --output_dir "dataset/train/video-depth-anything-metric" \
    --save_npz --max_res -1 --save_npz_fps -1
```

### 6. Compute per-pixel distance from the scene center

```bash
python camera_depth_to_scene_center_distance.py \
    --cameras_bin_path "dataset/train/sparse/0/cameras.bin" \
    --images_bin_path "dataset/train/sparse/0/images.bin" \
    --images_folder "dataset/train/images" \
    --depth_maps_folder "dataset/train/video-depth-anything-metric/images_depths_npz" \
    --save_folder "dataset/train/distances"
```

For every image, each pixel is unprojected using the metric depth map and the camera pose, and its distance to the world-space scene center `O` (placed at the barycenter of the camera positions) is stored in a "Distance-from-Center" (DFC) map.

### 7. Pick the foreground/background threshold `Ri` and mask the background

```bash
python limit_depth_maps_to_threshold.py \
    --distance_maps_folder "dataset/train/distances" \
    --images_folder "dataset/train/images" \
    --save_folder "dataset/train/distances_threshold"
```

If `--threshold` is not passed, the script applies the automatic heuristic described in the paper (histogram of the DFC values, binned every 10 m, threshold set right after the first bin with less than 2.5% of the points) and prints the chosen value, `<Ri>`. Pass `--threshold <Ri>` to override it manually. The output folder (suffixed with the threshold value, e.g. `distances_threshold_<Ri>`) contains the training images with all foreground pixels masked out — these are the background-only images used in Pass 1.

### 8. Initialize the background point cloud (geodesic sphere)

```bash
python generate_geodesic_points_faster_on_shell.py \
    --sparse_folder "dataset/train/sparse/0" \
    --images_folder "dataset/train/images" \
    --images_bg_folder "dataset/train/distances_threshold_<Ri>" \
    --dataset_base_folder "dataset/train" \
    --dataset_distances_folder "dataset/train/distances" \
    --object_threshold <Ri> \
    --infinity_threshold <Ro>
```

This builds a finely-subdivided icosphere (see `--subdivisions`, `--radius_mult`), places its vertices radially between `Ri` (object threshold) and `Ro` (infinity threshold) based on the depth maps — clamping to `Ro` for anything farther, which is how sky/very distant points get pinned to the outer sphere — and colors every point from the corresponding pixel in the input images. It writes a `points3D_*.ply` file inside `dataset_base_folder` and prints (on stdout, as `__KEY__: value` lines) the values needed later: geodesic point cloud filename, scene center, sphere radius, scaled inner radius. Keep these values around — they show up as placeholders (`<SCENE_CENTER>`, `<Ro>`, `<Ri_scaled>`) in the steps below.

There is also a depth-based variant, `generate_geodesic_points_faster_depth.py`, which places points using the depth value directly instead of a random radius inside the shell. In our tests it gave worse convergence in the outer shell, so it's not used by default (it's kept mainly for reference/experiments) — it takes the same arguments minus `--infinity_threshold`.

### 9. Filter the COLMAP point cloud to keep only the foreground

```bash
python keep_only_3Dpoints_using_depth_threshold.py \
    --points3D_bin_path "dataset/train/sparse/0/points3D.bin" \
    --images_bin_path "dataset/train/sparse/0/images.bin" \
    --distances_dir "dataset/train/distances" \
    --depth_threshold <Ri> \
    --dataset_root_dir "dataset/train"
```

This keeps only the COLMAP sparse points whose distance from the center is below `Ri`, and writes a `points3d_in_threshold_<Ri>.ply` file used to initialize the foreground in Pass 2.

### 10. Gaussian Splatting — Pass 1 (background)

Before training, arrange a dataset folder for the first-pass codebase like this:

```
gaussian-splatting-first-pass/
└── data/
    └── my_scene-1stPass/
        ├── images/              # masked background images, from step 7
        └── sparse/
            └── 0/
                ├── cameras.bin  # from step 4 (COLMAP)
                ├── images.bin   # from step 4 (COLMAP)
                └── points3D.ply # geodesic point cloud from step 8, renamed to points3D.ply
```

then train:

```bash
conda run -n gaussian-splatting python train.py \
    -s "gaussian-splatting-first-pass/data/my_scene-1stPass" \
    -m "gaussian-splatting-first-pass/output/my_scene-1stPass" \
    --iterations <ITERATIONS> \
    --scene_center "<SCENE_CENTER>" \
    --background_radius <Ro> \
    --scaled_inner_radius <Ri_scaled> \
    -r 1
```

Pass 1 uses the shell loss and planarity loss described in the paper (Section 3.2.2) and a modified pruning strategy that keeps large background Gaussians instead of pruning them (Section 3.2.3); both are implemented in `gaussian-splatting-first-pass/`.

### 11. Gaussian Splatting — Pass 2 (foreground)

Similarly, arrange a dataset folder for the second-pass codebase:

```
gaussian-splatting-second-pass/
└── data/
    └── my_scene-2ndPass/
        ├── gaussians_params_first_pass.pt  # Pass 1 checkpoint, copied here
        ├── images/                          # full, unmasked training images
        └── sparse/
            └── 0/
                ├── cameras.bin  # same COLMAP model as Pass 1
                ├── images.bin   # same COLMAP model as Pass 1
                └── points3D.ply # filtered foreground point cloud from step 9, renamed to points3D.ply
```

then train:

```bash
conda run -n gaussian-splatting python train.py \
    -s "gaussian-splatting-second-pass/data/my_scene-2ndPass" \
    -m "gaussian-splatting-second-pass/output/my_scene-2ndPass" \
    --iterations <ITERATIONS> \
    --scene_center "<SCENE_CENTER>" \
    --background_radius <Ri_scaled> \
    -r 1
```

Here `--background_radius` is set to the scaled inner radius `Ri`, used as the maximum radius allowed for foreground Gaussians (any foreground Gaussian moving past `Ri` is pruned, see paper Section 3.3.2). The background Gaussians from Pass 1 are rendered into the final image but stay frozen during this pass.

### 12. Evaluate on the held-out test set

First, register the test images (step 3's 20% split) into the existing COLMAP model, so we get camera poses for them too:

```bash
python register_new_images_colmap.py \
    --exhisting_source_path "dataset/register_test_in_train"
```

(`dataset/register_test_in_train` should contain an `input_test` folder with the test images and a copy of the training `distorted` COLMAP folder — `automatic_pipeline.py` sets this up automatically before calling the script.)

Then render the test views with the trained Pass 2 model and compute metrics:

```bash
conda run -n gaussian-splatting python render.py \
    -m "gaussian-splatting-second-pass/output/my_scene-2ndPass" \
    -s "dataset/register_test_in_train"

conda run -n gaussian-splatting python metricsDeborah.py \
    -m "gaussian-splatting-second-pass/output/my_scene-2ndPass"
```

`render.py` and `metricsDeborah.py` are part of `gaussian-splatting-first-pass/` (the codebase used to render, since it holds the renderer used for evaluation) and report SSIM/PSNR/LPIPS as in Table 1 of the paper.

## Extra utilities

These scripts are not called by `automatic_pipeline.py`, but are useful for downstream tasks:

- **`align_gs_and_cameras_with_world_axis.py`** — aligns a trained PLY (and the corresponding COLMAP cameras) to the world axes, e.g. to get a "flat ground" orientation for viewers like the Unity Gaussian Splatting plugin. Takes the COLMAP sparse folder, the PLY to align, and optionally a `cameras.json`, and writes the aligned PLY/`images.bin`/`cameras.bin`/`cameras.json` (with `--save_unity_ply_path` / `--save_unity_json_path` variants for Unity).
- **`remove_env_gs_from_2nd_pass.py`** — compares the Pass 1 and Pass 2 PLYs and removes from the Pass 2 point cloud any Gaussian that is identical to one already present in the Pass 1 (background) point cloud. Useful when you want a "foreground only" PLY to combine manually with an environment map. Paths are currently hard-coded at the top of the script (`PLY_FILE_1`, `PLY_FILE_2`, `PLY_FILE_OUT`) — edit them before running.
- **`360-gs-scripts/render_spherical_ply.py`** — renders an equirectangular 360° image directly from a trained PLY file (used to produce the environment maps of Section 3.4/4.1 in the paper):

```bash
python 360-gs-scripts/render_spherical_ply.py \
    --ply_path "gaussian-splatting-first-pass/output/my_scene-1stPass/point_cloud/iteration_<ITERATIONS>/point_cloud.ply" \
    --output_path "envmap.png" \
    --sh_degree 0 \
    --image_width 8192 \
    --image_height 4096 \
    --center "<SCENE_CENTER>"
```

Use `--sh_degree 0` (or `1`) as recommended in the paper's supplementary material, to avoid baking view-dependent artifacts into the static texture.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{pintani2026envmapgs,
  title   = {EnvMap-GS: two-stage outdoor Gaussian reconstruction with background-to-environment map baking},
  author  = {Pintani, Deborah and Caputo, Ariel and Lewis, Noah and Stamminger, Marc and Pellacini, Fabio and Giachetti, Andrea},
  journal = {The Visual Computer},
  year    = {2026},
  doi     = {10.1007/s00371-026-04593-w}
}
```
