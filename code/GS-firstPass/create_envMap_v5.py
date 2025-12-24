import os
import torch
import math
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image as PILImage
import torch.nn.functional as F

# --- IMPORT STANDARD GS ---
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel

# --- 1. MATEMATICA DI SUPPORTO ---

def create_look_at_view_matrix(camera_center, target, up):
    forward = target - camera_center
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0, 0, 1]) 
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    
    R = np.stack([right, new_up, -forward], axis=0)
    C2W = np.identity(4)
    C2W[:3, :3] = R.T
    C2W[:3, 3] = camera_center
    W2C = np.linalg.inv(C2W)
    return torch.tensor(W2C, dtype=torch.float32).cuda()

# --- 2. STITCHING CON BLENDING "CENTRALE" ---

def stitch_multiview_blended(views_data, out_h=1024, out_w=2048, fov_rad=None):
    print(f"Blending {len(views_data)} views (High-Focus Blending)...")
    
    tan_half_fov = math.tan(fov_rad / 2.0)
    
    # old - only to fix img flipped
    #y_range = torch.linspace(-math.pi/2, math.pi/2, out_h, device="cuda")
    #new
    y_range = torch.linspace(math.pi/2, -math.pi/2, out_h, device="cuda")

    x_range = torch.linspace(-math.pi, math.pi, out_w, device="cuda")
    phi, theta = torch.meshgrid(y_range, x_range, indexing='ij')
    
    theta = theta + math.pi 
    
    dir_x = torch.cos(phi) * torch.sin(theta)
    dir_y = torch.sin(phi)
    dir_z = torch.cos(phi) * torch.cos(theta)
    
    world_dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1).reshape(-1, 3) 
    
    final_color = torch.zeros((3, out_h * out_w), device="cuda")
    total_weight = torch.zeros((1, out_h * out_w), device="cuda") + 1e-6

    for view in tqdm(views_data, desc="Blending"):
        image = view['image'].unsqueeze(0) 
        w2c = view['w2c']
        
        R = w2c[:3, :3] 
        cam_dirs = R @ world_dirs.T 
        
        x_cam = cam_dirs[0, :]
        y_cam = cam_dirs[1, :]
        z_cam = cam_dirs[2, :]
        
        # Check visibilità
        valid_mask = (z_cam < -0.001)
        
        depth = -z_cam + 1e-8
        u_raw = x_cam / depth
        v_raw = -y_cam / depth 
        
        u = u_raw / tan_half_fov
        v = v_raw / tan_half_fov
        
        valid_mask = valid_mask & (u > -1.0) & (u < 1.0) & (v > -1.0) & (v < 1.0)
        
        if not valid_mask.any():
            continue

        # --- MODIFICA CRUCIALE: PESO ---
        dist_from_center = torch.sqrt(u**2 + v**2)
        
        # Formula: (1 - dist)^N. 
        # Se N è alto (es. 10), il peso crolla velocemente appena ti allontani dal centro.
        # Questo elimina il ghosting perché i bordi (dove avviene il ghosting) pesano 0.0001
        # mentre il centro della vista sovrapposta peserà 1.0.
        weight = torch.clamp(1.0 - (dist_from_center / 1.41), 0.0, 1.0)
        weight = torch.pow(weight, 10.0) 
        
        current_weight = weight * valid_mask.float()
        
        grid = torch.stack([u, v], dim=-1).unsqueeze(0).unsqueeze(1) 
        # align_corners=False è matematicamente più preciso per proiezioni geometriche
        sampled_color = F.grid_sample(image, grid, align_corners=False, padding_mode='border') 
        sampled_color = sampled_color.squeeze() 

        final_color += sampled_color * current_weight
        total_weight += current_weight

    final_color = final_color / total_weight
    return final_color.reshape(3, out_h, out_w)

# --- 3. MAIN ---

def create_environment_map(model_path, source_path, iteration, resolution=2048, scene_center_coords=None):
    parser = ArgumentParser(description="Config setup.")
    model_params_container = ModelParams(parser)
    
    class Args:
        def __init__(self, m_path, s_path):
            self.model_path = m_path
            self.source_path = s_path
            self.depths = "" 
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
            self.train_test_exp = False
            self.sh_degree = 3
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False

    args = Args(model_path, source_path)
    dataset = model_params_container.extract(args)
    gaussians = GaussianModel(dataset.sh_degree) 
    
    print(f"Loading scene from iteration {iteration}...")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if scene_center_coords:
        scene_center = torch.tensor(scene_center_coords, dtype=torch.float32, device="cuda")
    else:
        scene_center = torch.zeros(3, device="cuda")
        train_cams = scene.getTrainCameras()
        if train_cams:
            for cam in train_cams:
                scene_center += cam.camera_center
            scene_center /= len(train_cams)
        print(f"Scene center: {scene_center.cpu().numpy()}")

    render_res = resolution 
    
    # 10 Viste: 8 Orizzontali (45 deg step) + Top + Bottom
    views_config = [
        # Orizzontali (Orizzonte sempre dritto)
        {'target': [1, 0, 0],   'up': [0, 1, 0]}, # 0
        {'target': [1, 0, 1],   'up': [0, 1, 0]}, # 45
        {'target': [0, 0, 1],   'up': [0, 1, 0]}, # 90
        {'target': [-1, 0, 1],  'up': [0, 1, 0]}, # 135
        {'target': [-1, 0, 0],  'up': [0, 1, 0]}, # 180
        {'target': [-1, 0, -1], 'up': [0, 1, 0]}, # 225
        {'target': [0, 0, -1],  'up': [0, 1, 0]}, # 270
        {'target': [1, 0, -1],  'up': [0, 1, 0]}, # 315
        # Verticali
        {'target': [0, 1, 0],   'up': [0, 0, -1]}, # Top
        {'target': [0, -1, 0],  'up': [0, 0, 1]},  # Bottom
    ]

    # FoV = 100 gradi.
    # Abbastanza per coprire i buchi, ma con bassa distorsione.
    fov_deg = 100
    FoV = math.radians(fov_deg) 

    pipeline_args = PipelineParams(parser).extract(args)
    views_data = []

    debug_path = os.path.join(dataset.model_path, "output_equirectangular/v5", "debug_views")
    os.makedirs(debug_path, exist_ok=True)

    with torch.no_grad():
        for i, config in enumerate(tqdm(views_config, desc="Rendering")):
            target_vec = np.array(config['target'], dtype=np.float32)
            up_vec = np.array(config['up'], dtype=np.float32)
            
            center_np = scene_center.cpu().numpy()
            w2c_gpu = create_look_at_view_matrix(center_np, center_np + target_vec, up_vec)
            
            R_cpu = w2c_gpu[:3, :3].T.cpu().numpy()
            T_cpu = w2c_gpu[:3, 3].cpu().numpy()
            
            dummy_pil = PILImage.new('RGB', (render_res, render_res), (0, 0, 0))

            cam = Camera(
                colmap_id=i, R=R_cpu, T=T_cpu, FoVx=FoV, FoVy=FoV, 
                image=dummy_pil, image_name=f"view_{i}", uid=i, 
                data_device="cuda", resolution=(render_res, render_res),
                depth_params=None, invdepthmap=None
            )
            
            cam.image_width = render_res
            cam.image_height = render_res

            rendered_image = render(cam, gaussians, pipeline_args, background)["render"]
            
            torchvision.utils.save_image(rendered_image, os.path.join(debug_path, f"view_{i:02d}.png"))
            
            views_data.append({
                'image': rendered_image, 
                'w2c': w2c_gpu 
            })

    # Blending
    out_height = resolution
    out_width = resolution * 2
    equi_image = stitch_multiview_blended(views_data, out_h=out_height, out_w=out_width, fov_rad=FoV)
    
    output_path = os.path.join(dataset.model_path, "output_equirectangular/v5")
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, "env_map_final_blended_clean.png")
    torchvision.utils.save_image(equi_image, output_file)
    print(f"\nEnvironment map salvata in: {output_file}")

if __name__ == "__main__":
    MODEL_PATH = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper\\fields_deb_eval_th70.0_2ndPass" 
    SOURCE_PATH = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\data\\paper\\fields_deb_eval_th70.0_2ndPass"
    ITERATION = 30000 
    
    try:
        create_environment_map(MODEL_PATH, SOURCE_PATH, ITERATION, resolution=4096)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERRORE: {e}")