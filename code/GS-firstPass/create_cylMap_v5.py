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
import gc 

# --- IMPORT STANDARD GS ---
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel

# ==========================================
# 1. MATEMATICA DI VISTA
# ==========================================
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

# ==========================================
# 2. TRUE SLIT-SCAN RENDERING
# ==========================================
def create_true_slitscan_safe(model_path, source_path, iteration, resolution=2048, scene_center_coords=None):
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
        print(f"Center detected: {scene_center.cpu().numpy()}")

    # --- PARAMETRI OUTPUT ---
    out_width = resolution * 2  # 4096 per res 2048
    out_height = resolution     # 2048
    
    # Renderizziamo una striscia di 16 pixel per ogni colonna finale
    # Questo garantisce che il rasterizzatore (che lavora a tiles di 16x16) sia stabile.
    RENDER_WIDTH = 16 
    CENTER_COL = RENDER_WIDTH // 2
    
    # FoV Verticale (limite per cilindrica ~140-150 gradi)
    fov_y_rad = math.radians(145) 
    
    # Calcolo FoV Orizzontale coerente per la striscia da 16px
    strip_aspect = RENDER_WIDTH / out_height
    fov_x_rad = 2 * math.atan(strip_aspect * math.tan(fov_y_rad / 2.0))
    
    pipeline_args = PipelineParams(parser).extract(args)
    
    # UP VECTOR: Usiamo [0, 1, 0] visto che le debug image erano dritte
    GLOBAL_UP = np.array([0, 1, 0], dtype=np.float32)
    center_np = scene_center.cpu().numpy()

    # Mappa finale su CPU (Cruciale!)
    final_map = torch.zeros((3, out_height, out_width), dtype=torch.float32)

    # Cartella output
    output_path = os.path.join(dataset.model_path, "output_cylindrical/v5.1")
    os.makedirs(output_path, exist_ok=True)
    temp_file = os.path.join(output_path, "true_slitscan_progress.png")

    print(f"Starting PIXEL-PERFECT True Slit-Scan...")
    print(f"Rendering {out_width} columns individually.")

    # Ciclo colonna per colonna
    for col_idx in tqdm(range(out_width), desc="Scanning Columns"):
        
        # Angolo esatto per questa singola colonna
        theta = (col_idx / out_width) * 2 * math.pi
        theta += math.pi 
        
        # Target
        x = math.sin(theta)
        z = math.cos(theta)
        target_vec = np.array([x, 0, z], dtype=np.float32)
        
        # W2C
        w2c_gpu = create_look_at_view_matrix(center_np, center_np + target_vec, GLOBAL_UP)
        R_cpu = w2c_gpu[:3, :3].T.cpu().numpy()
        T_cpu = w2c_gpu[:3, 3].cpu().numpy()
        
        dummy_pil = PILImage.new('RGB', (RENDER_WIDTH, out_height), (0, 0, 0))

        # Camera
        cam = Camera(
            colmap_id=col_idx, R=R_cpu, T=T_cpu, FoVx=fov_x_rad, FoVy=fov_y_rad, 
            image=dummy_pil, image_name=f"p_{col_idx}", uid=col_idx, 
            data_device="cuda", resolution=(RENDER_WIDTH, out_height),
            depth_params=None, invdepthmap=None
        )
        cam.image_width = RENDER_WIDTH
        cam.image_height = out_height

        # Rendering
        with torch.no_grad():
            render_result = render(cam, gaussians, pipeline_args, background)["render"]
            
            # Estraiamo SOLO il singolo pixel centrale (colonna di 1px)
            # Spostiamo subito su CPU
            single_column = render_result[:, :, CENTER_COL].cpu()
            
            # Inseriamo nella mappa finale
            final_map[:, :, col_idx] = single_column

        # --- PULIZIA MEMORIA ---
        del render_result
        del cam
        del w2c_gpu
        
        # Pulizia ogni 50 colonne (visto che sono render piccoli, possiamo farlo meno spesso)
        if col_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
            # Salvataggio progressivo
            if col_idx % 200 == 0:
                torchvision.utils.save_image(final_map, temp_file)

    # Salvataggio Finale
    output_file = os.path.join(output_path, "env_map_TRUE_slitscan.png")
    torchvision.utils.save_image(final_map, output_file)
    print(f"\nFinal Slit-Scan Map saved to: {output_file}")

if __name__ == "__main__":
    MODEL_PATH = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gass-splat-first-pass-multiply\\output\\paper_I3D\\fields_80-20-eval-NO_EXP-shell_from_70_to_200-1stPass" 
    SOURCE_PATH = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gass-splat-first-pass-multiply\\data\\paper_eurographics\\fields_deb_eval_th70.0_1stPass"
    ITERATION = 30000 
    
    try:
        # Se 2048 è troppo lungo per i test, prova resolution=512 per un check rapido
        create_true_slitscan_safe(MODEL_PATH, SOURCE_PATH, ITERATION, resolution=2048)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERRORE: {e}")