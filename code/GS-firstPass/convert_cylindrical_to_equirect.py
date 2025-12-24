import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import math
import os
import argparse

def convert_cylindrical_to_equirectangular(image_path, output_path, src_fov_deg=145.0, out_res=None):
    """
    Converte una mappa Cilindrica Prospettica (Slit-Scan) in Equirettangolare.
    """
    print(f"Loading map: {image_path}")
    
    # 1. Carica Immagine
    # Usiamo PIL e convertiamo in Tensore PyTorch
    pil_img = Image.open(image_path).convert('RGB')
    W_src, H_src = pil_img.size
    
    # Converti in tensore (C, H, W) normalizzato 0-1
    img_tensor = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).cuda()
    
    # Risoluzione Output (se non specificata, usa la larghezza della sorgente)
    if out_res is None:
        out_w = W_src
        out_h = W_src // 2 # Equirettangolare è sempre 2:1
    else:
        out_w = out_res
        out_h = out_res // 2
        
    print(f"Converting to Equirectangular ({out_w}x{out_h})...")
    print(f"Source Vertical FoV assumed: {src_fov_deg} degrees")

    # 2. Crea la Griglia Target (Equirettangolare)
    # Coordinate UV normalizzate da -1 a 1
    # Y va da -1 (Alto/Nord) a 1 (Basso/Sud) secondo la convenzione grid_sample
    y_rng = torch.linspace(-1, 1, out_h, device="cuda")
    x_rng = torch.linspace(-1, 1, out_w, device="cuda")
    
    y_grid, x_grid = torch.meshgrid(y_rng, x_rng, indexing='ij')
    
    # 3. Mappa UV Target -> Coordinate Sferiche (Lat/Lon)
    # x_grid (-1 to 1) -> Longitudine (-PI to PI)
    theta = x_grid * math.pi
    
    # y_grid (-1 to 1) -> Latitudine (PI/2 to -PI/2)
    # Nota: grid_sample y=-1 è Alto. Latitudine PI/2 è Alto.
    # Quindi phi = -y * (pi/2)
    phi = -y_grid * (math.pi / 2.0)
    
    # 4. Calcola dove pescare nella mappa Cilindrica Sorgente
    # Coordinata U (Orizzontale): È lineare in entrambi i casi.
    # Quindi u_source = u_target.
    u_source = x_grid
    
    # Coordinata V (Verticale): Qui sta la magia.
    # La sorgente è prospettica: y = tan(phi)
    # Dobbiamo normalizzare in base al FoV della sorgente.
    
    fov_rad = math.radians(src_fov_deg)
    tan_half_fov = math.tan(fov_rad / 2.0)
    
    # Proiezione: altezza sulla pellicola piana
    projected_h = torch.tan(phi)
    
    # Normalizzazione (-1 a 1) rispetto al FoV originale
    # v_source = projected_h / max_h
    # Invertiamo il segno perché Y pixel cresce verso il basso, ma tan(phi) cresce verso l'alto
    v_source = -projected_h / tan_half_fov
    
    # 5. Maschera i pixel fuori dal FoV originale
    # Se la latitudine richiesta è più alta di quanto la camera vedeva, sarà nero.
    # La camera vedeva fino a src_fov_deg / 2
    max_lat = fov_rad / 2.0
    valid_mask = (torch.abs(phi) <= max_lat)
    
    # Costruiamo la grid finale per il campionamento
    sampling_grid = torch.stack([u_source, v_source], dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # 6. Campionamento (Warping)
    # padding_mode="zeros" mette nero dove non abbiamo dati (i poli)
    sampled = F.grid_sample(img_tensor, sampling_grid, align_corners=True, mode='bilinear', padding_mode="zeros")
    
    # Applica la maschera (opzionale, grid_sample coi poli fuori range > 1 fa già nero o repeat)
    # Con padding_mode='zeros', v_source > 1 diventa nero automaticamente.
    
    # 7. Salvataggio
    output_tensor = sampled.squeeze(0).cpu()
    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.utils.save_image(output_tensor, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    
    # Percorso della mappa cilindrica che hai generato
    INPUT_MAP = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper\\fields_deb_eval_th70.0_2ndPass\\output_cylindrical\\v5.1\\env_map_TRUE_slitscan.png"
    
    # Dove salvare quella sferica
    OUTPUT_MAP = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\gaussian-splatting-code\\gaussian-splatting-second-pass\\output\\paper\\fields_deb_eval_th70.0_2ndPass\\output_converted\\env_map_equirectangular_final.png"
    
    # IMPORTANTE: Inserisci qui lo stesso numero che c'era nello script di generazione
    # Nel tuo ultimo script era math.radians(145), quindi 145 gradi.
    SOURCE_FOV_Y = 145.0
    
    # Risoluzione larghezza finale (es. 4096)
    # Se metti None, usa la stessa dell'input
    FINAL_RES = 4096 
    
    try:
        # Crea dummy files per test se non esistono o correggi i path
        if os.path.exists(INPUT_MAP):
            convert_cylindrical_to_equirectangular(INPUT_MAP, OUTPUT_MAP, SOURCE_FOV_Y, FINAL_RES)
        else:
            print(f"Errore: File non trovato {INPUT_MAP}")
            print("Modifica i percorsi nel blocco __main__.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()