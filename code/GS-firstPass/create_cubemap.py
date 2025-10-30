import os
import torch
import math
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image as PILImage

# Importa le classi e le funzioni necessarie dal codebase di GS
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel

def create_look_at_view_matrix(camera_center, target, up):
    """Crea una matrice di trasformazione world-to-view (stile lookAt)."""
    forward = target - camera_center
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        if abs(forward[1]) < 0.99:
            right = np.cross(forward, [0, 1, 0])
        else:
            right = np.cross(forward, [1, 0, 0])
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    R = np.stack([right, new_up, -forward], axis=0)
    t = -R @ camera_center.reshape(3, 1)
    C2W = np.identity(4)
    C2W[:3, :3] = R.T
    C2W[:3, 3] = camera_center
    W2C = np.linalg.inv(C2W)
    return torch.tensor(W2C, dtype=torch.float32).cuda()

# La funzione ora accetta tutti i parametri necessari
def create_cubemap(model_path, source_path, iteration, output_path, resolution=2048, scene_center_coords=None):
    # Creiamo un oggetto ModelParams nel modo corretto
    parser = ArgumentParser(description="Configurazione fittizia per il caricamento della scena.")
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

            # Parametri per PipelineParams
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False

    dataset = model_params_container.extract(Args(model_path, source_path))

    # 1. Crea un'istanza "vuota" di GaussianModel. 
    gaussians = GaussianModel(dataset.sh_degree) 
    
    # 2. Inizializza la Scena, che caricherà il modello dall'iterazione specificata.
    #    Durante questa chiamata, il metodo `gaussians.restore()` verrà eseguito
    #    e popolerà tutti gli attributi, incluso `active_sh_degree`.
    print(f"Caricamento della scena e del modello dall'iterazione {iteration} in '{dataset.model_path}'...")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    scene.white_background = dataset.white_background
    
    print(f"Modello caricato con successo. Grado SH attivo: {gaussians.active_sh_degree}")
    
    # --- FINE MODIFICA CHIAVE V4 ---
    
    # Prepara la pipeline di rendering
    pipeline_args_container = PipelineParams(parser)
    pipeline = pipeline_args_container.extract(Args(model_path, source_path))
    
    bg_color = [1, 1, 1] if scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # --- LOGICA CORRETTA PER IL CENTRO DELLA SCENA ---
    if scene_center_coords:
        scene_center = torch.tensor(scene_center_coords, dtype=torch.float32, device="cuda")
        print(f"Usando il centro della scena fornito dall'utente: {scene_center.cpu().numpy()}")
    else:
        print("Centro della scena non fornito. Calcolo della media dalle telecamere di training...")
        scene_center = torch.zeros(3, device="cuda")
        train_cams = scene.getTrainCameras()
        if not train_cams:
            print("ERRORE: Nessuna camera di training trovata per calcolare il centro. Fornire un centro con --scene_center.")
            return
        for cam in train_cams:
            scene_center += cam.camera_center
        scene_center /= len(train_cams)
        print(f"Centro della scena calcolato in: {scene_center.cpu().numpy()}")

    # Definisci le 6 facce della cubemap (target, up)
    cubemap_faces = {
        'px': {'target': [1, 0, 0], 'up': [0, 1, 0]},   # Right
        'nx': {'target': [-1, 0, 0], 'up': [0, 1, 0]},  # Left
        'py': {'target': [0, 1, 0], 'up': [0, 0, -1]},  # Up (guarda su, "avanti" è -Z)
        'ny': {'target': [0, -1, 0], 'up': [0, 0, 1]},  # Down (guarda giù, "avanti" è +Z)
        'pz': {'target': [0, 0, 1], 'up': [0, 1, 0]},   # Back
        'nz': {'target': [0, 0, -1], 'up': [0, 1, 0]}   # Front
    }

    output_path = os.path.join(dataset.model_path, "output_cubemap")
    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        for name, face_params in tqdm(cubemap_faces.items(), desc="Rendering facce della cubemap"):
            target_vec = np.array(face_params['target'])
            up_vec = np.array(face_params['up'])
            center_np = scene_center.cpu().numpy()
            
            world_view_transform_gpu = create_look_at_view_matrix(center_np, center_np + target_vec, up_vec)

            # Prepariamo gli argomenti per il costruttore della Camera, assicurandoci
            # che siano sulla CPU se le funzioni interne usano NumPy.
            R_cpu = world_view_transform_gpu[:3, :3].T.cpu()
            T_cpu = world_view_transform_gpu[:3, 3].cpu()
            
            FoV = math.pi / 2.0
            
            # Crea un'immagine PIL fittizia (nera) della dimensione corretta.
            # Questo è il tipo di oggetto che la funzione PILtoTorch si aspetta.
            dummy_pil_image = PILImage.new('RGB', (resolution, resolution), (0, 0, 0))

            # Crea un oggetto Camera sintetico, passando l'immagine PIL.
            synthetic_camera = Camera(
                colmap_id=0, 
                R=R_cpu, # Passiamo il tensore sulla CPU
                T=T_cpu, # Passiamo il tensore sulla CPU
                FoVx=FoV, 
                FoVy=FoV, 
                image=dummy_pil_image,
                image_name=name, 
                uid=0, 
                data_device="cuda",
                resolution=resolution,
                depth_params=None,
                invdepthmap=None
            )
            
            # Queste righe non sono più strettamente necessarie, ma sono innocue.
            synthetic_camera.image_width = resolution
            synthetic_camera.image_height = resolution

            rendering = render(synthetic_camera, gaussians, pipeline, background)["render"]
            
            output_file = os.path.join(output_path, f"{name}.png")
            torchvision.utils.save_image(rendering, output_file)
            print(f"Salvata faccia '{name}' in '{output_file}'")

    print("\nRendering della cubemap completato.")
    stitch_cubemap(output_path, resolution)

def stitch_cubemap(path, res):
    try:
        faces = {name: PILImage.open(os.path.join(path, f"{name}.png")) for name in ['px', 'nx', 'py', 'ny', 'pz', 'nz']}
        cross_layout = PILImage.new('RGB', (res * 4, res * 3))
        cross_layout.paste(faces['nx'], (0, res))
        cross_layout.paste(faces['pz'], (res, res))
        cross_layout.paste(faces['px'], (res * 2, res))
        cross_layout.paste(faces['nz'], (res * 3, res))
        cross_layout.paste(faces['py'], (res, 0))
        cross_layout.paste(faces['ny'], (res, res * 2))
        output_file = os.path.join(path, "cubemap_cross.png")
        cross_layout.save(output_file)
        print(f"Cubemap unita salvata in '{output_file}'")
    except Exception as e:
        print(f"Impossibile unire le facce della cubemap: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a cubemap from a trained Gaussian Splatting model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model output folder (containing point_cloud, etc.).")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the input scene folder (containing the COLMAP sparse reconstruction).")
    parser.add_argument("-i", "--iteration", type=int, default=30000, help="Iteration number of the model to load.")
    parser.add_argument("-o", "--output_path", type=str, default="cubemap_output", help="Path to save the output cubemap faces.")
    parser.add_argument("-r", "--resolution", type=int, default=2048, help="Resolution of each square cubemap face.")
    # Manteniamo l'opzione per specificare il centro
    parser.add_argument("-c", "--scene_center", type=float, nargs=3, default=None, help="Specify the XYZ coordinates for the cubemap center (e.g., -c 1.0 2.5 0.5).")
    args = parser.parse_args()

    # Passiamo tutti gli argomenti necessari alla funzione
    create_cubemap(args.model_path, args.source_path, args.iteration, args.output_path, args.resolution, args.scene_center)