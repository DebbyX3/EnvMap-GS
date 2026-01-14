#
# Script per rendering sferico di un modello Gaussian Splatting da file .ply
#
# Usage example:
#   python render_spherical_ply.py --ply_path path/to/model.ply --output_path path/to/output.png --sh_degree 3 --image_width 2048 --image_height 1024

import torch
import os
import argparse
from gaussian_renderer import render_spherical
from scene.gaussian_model import GaussianModel
import torchvision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render sferico di modello Gaussian Splatting da .ply")
    parser.add_argument('--ply_path', type=str, required=True, help='Percorso al file .ply del modello Gaussian Splatting')
    parser.add_argument('--output_path', type=str, required=True, help='Percorso dove salvare l\'immagine 360 risultante')
    parser.add_argument('--sh_degree', type=int, default=3, help='Spherical Harmonics degree (default: 3)')
    parser.add_argument('--image_width', type=int, default=8192, help='Larghezza immagine 360 (default: 2048)')
    parser.add_argument('--image_height', type=int, default=4096, help='Altezza immagine 360 (default: 1024)')
    parser.add_argument('--white_background', action='store_true', help='Usa sfondo bianco invece che nero')
    parser.add_argument('--center', nargs=3, type=float, default=None, metavar=('CX','CY','CZ'), help='Centro della sfera/scena (es: --center 1.0 2.0 3.0)')
    parser.add_argument('--scale_modifier', type=float, default=1.0, help='Fattore di scala per le gaussiane nel renderer (default: 1.0)')
    parser.add_argument('--internal_scale', type=float, default=1.0, help='Moltiplicatore diretto sui parametri di scala delle gaussiane dopo il load_ply (default: 1.0)')
    parser.add_argument('--fovx', type=float, default=1, help='Campo visivo orizzontale (radiani, default: 3.14159 ~ 180°)')
    parser.add_argument('--fovy', type=float, default=0.5, help='Campo visivo verticale (radiani, default: pi/2 ~ 90°)')
    args = parser.parse_args()


    # Carica modello Gaussian Splatting da .ply
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply_path)
    # Applica scaling diretto ai parametri di scala delle gaussiane se richiesto
    if args.internal_scale != 1.0:
        with torch.no_grad():
            gaussians._scaling *= args.internal_scale
        print(f"[INFO] Scaling diretto applicato alle gaussiane: x{args.internal_scale}")

    # Crea una camera virtuale sferica (mock)

    class SphericalCamera:
        def __init__(self, width, height, center, fovx, fovy):
            self.image_width = width
            self.image_height = height
            self.FoVx = fovx
            self.FoVy = fovy
            self.world_view_transform = torch.eye(4, dtype=torch.float32, device="cuda")
            self.full_proj_transform = torch.eye(4, dtype=torch.float32, device="cuda")
            self.camera_center = torch.tensor(center, dtype=torch.float32, device="cuda")

    center = args.center if args.center is not None else [0.0, 0.0, 0.0]
    camera = SphericalCamera(args.image_width, args.image_height, center, args.fovx, args.fovy)

    # Parametri pipeline mock
    class PipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    pipeline = PipelineParams()

    bg_color = [1,1,1] if args.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        result = render_spherical(camera, gaussians, pipeline, background, scaling_modifier=args.scale_modifier)
        image = result["render"]
        torchvision.utils.save_image(image, args.output_path)
    print(f"Immagine 360 salvata in {args.output_path}")

# conda activate gaussian-splatting-360

# cd C:\Users\User\Desktop\Gaussian Splatting\gaussian-splatting-code\360-gaussian-splatting

# python render_spherical_ply.py --ply_path "C:\Users\User\Desktop\Gaussian Splatting\gaussian-splatting-code\gass-splat-first-pass-multiply\output\paper_I3D\fields_80-20-eval-NO_EXP-shell_from_70_to_200-1stPass\point_cloud\iteration_30000\point_cloud.ply" --output_path "C:\Users\User\Desktop\Gaussian Splatting\gaussian-splatting-code\gass-splat-first-pass-multiply\output\paper_I3D\fields_80-20-eval-NO_EXP-shell_from_70_to_200-1stPass\output_spherical.png" --image_width 8192 --image_height 4096 --center 0.01860108 -0.03761125 -0.14536059