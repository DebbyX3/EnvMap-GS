#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not gt_path.exists():
            print(f"Warning: ground truth file not found for {fname}, skipping.")
            continue
        render = Image.open(render_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGBA')
        # Crop render to gt size if needed
        if render.size != gt_img.size:
            print(f"Cropping render {fname} from {render.size} to {gt_img.size}")
            render = render.crop((0, 0, gt_img.size[0], gt_img.size[1]))
        gt_tensor = tf.to_tensor(gt_img)  # shape: [4, H, W]
        mask = gt_tensor[3] > 0.99  # mask: True where alpha is full
        gt_rgb = gt_tensor[:3]
        render_tensor = tf.to_tensor(render)
        renders.append(render_tensor.unsqueeze(0).cuda())
        gts.append(gt_rgb.unsqueeze(0).cuda())
        masks.append(mask.unsqueeze(0).cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            ours_dir = Path(scene_dir) / "train" / "ours_30000"
            gt_dir = ours_dir / "gt"
            renders_dir = ours_dir / "renders"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir)
            ssims = []
            psnrs = []
            lpipss = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                render = renders[idx]
                gt = gts[idx]
                mask = masks[idx]
                # Espandi la maschera sui canali colore
                mask3 = mask.unsqueeze(1).repeat(1, 3, 1, 1)  # shape: [1, 3, H, W]
                ssims.append(ssim(render * mask3, gt * mask3))
                psnrs.append(psnr(render * mask3, gt * mask3))
                lpipss.append(lpips(render * mask3, gt * mask3, net_type='vgg'))
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")
            full_dict[scene_dir]["ours_30000"].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir]["ours_30000"].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            with open(str(scene_dir) + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(str(scene_dir) + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Error:", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
