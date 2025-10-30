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

import re
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import build_rotation
import torch.nn as nn
import torchvision.models as models

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# kernel Laplaciano
laplacian_kernel = torch.tensor([
[1,  1,  1],
[1, -8,  1],
[1,  1,  1]
], dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    Trains the Gaussian Splatting model.
    Args:
        dataset: The dataset object containing training data and configurations.
        opt: The options object containing training hyperparameters and configurations.
        pipe: The rendering pipeline object.
        testing_iterations: List of iterations at which to perform testing.
        saving_iterations: List of iterations at which to save the model.
        checkpoint_iterations: List of iterations at which to save checkpoints.
        checkpoint: Path to a checkpoint file to resume training from.
        debug_from: Iteration number from which to enable debugging.
    Returns:
        None
    """

    '''
    Converts a string representing a list of floats into a list of floats.
    Handles various formats:
      - "[1 2 3]"
      - "[1, 2, 3]"
      - "1 2 3"
      - "1, 2, 3"
    '''
    cleaned = dataset.scene_center.strip("[]() ")
    cleaned = cleaned.replace(",", " ")
    parts = re.split(r"\s+", cleaned.strip())
    scene_center_list = [float(x) for x in parts if x]

    cleaned = dataset.background_radii.strip("[]() ")
    cleaned = cleaned.replace(",", " ")
    parts = re.split(r"\s+", cleaned.strip())
    background_radii_list = [float(x) for x in parts if x]

    scene_center = torch.tensor(scene_center_list, device="cuda")
    background_radius = torch.tensor(dataset.background_radius, device="cuda")
    background_radii = torch.tensor(background_radii_list, device="cuda")
    scaled_inner_radius = torch.tensor(dataset.scaled_inner_radius, device="cuda")

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # initialize gaussian model. class handles aspects of a gauss based model
    # sh_degree is the degree of the spherical harmonics
    # optimizer_type is the type of optimizer to use
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # Scene class is designed to manage and organize various components of a 3D scene
    # dataset: instance of ModelParams, contains various configuration settings and paths necessary for setting up the scene
    # gaussians: is an instance of the GaussianModel class, which handles the Gaussian-based model used within the scene
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # riesume il modello da un checkpoint precedente
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # regolizzatore di pesi 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # prende par che gli servono tipo le camere
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # aggiorna il learning rate - moltiplicatore per l'ottimizzatore 
        # mi sposto di meno o di più rispetto l'opposto del gradiente
        # se basso aggiorno di poco i pesi, se alto aggiorno di molto
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # IMAGE: imagine in 'output' da gs, è quella renderizzata
        # GT_IMAGE: immagine originale, ground truth

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # maschera (binaria?) (probabilmente data nel dataset) che maschera/oscura/toglie parti dell'immagine
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Ora usiamo la chiave che ESISTE GIA' nel tuo codice
        visibility_indices = render_pkg["visibility_filter"]

        # Incrementa il contatore per le gaussiane che sono state viste in questo frame
        gaussians.times_seen[visibility_indices] += 1



        # Loss
        # prendo img originale e carico il tensore su gpu nvidia (cuda) per fare l'op su gpu
        gt_image = viewpoint_cam.original_image.cuda() #.cuda() è di pytorch - original_image deve essere già un tensore prima di chaimare .cuda()
        loss_l1 = l1_loss(image, gt_image)
        
        # calcola la loss l1 tra img e gt_img
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 2. Definisci dei lambda di partenza ragionevoli
        #    La loss di planarità può essere debole, quella sui gusci più forte.
        lambda_planarity = 0.1
        lambda_shell_constraint = 1.0

        # 3. Rimuovi tutte le loss per la nitidezza (gradient, laplacian, etc.).
        #    Non ci servono più, perché la geometria corretta produrrà nitidezza.
        loss_rendering = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 4. Calcola le loss geometriche
        loss_planarity = compute_planarity_loss(gaussians, scene_center)

        '''
        loss_shell = compute_inward_shell_constraint_loss(
            gaussians, 
            scene_center, 
            background_radius,
            scaled_inner_radius
        )
        '''

        # 5. Calcola la loss totale
        loss = (
            loss_rendering +
            lambda_planarity * loss_planarity# +
            #lambda_shell_constraint * loss_shell
        )





        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Backward pass sulla loss
        # ho tutto il grafo delle operazioni.
        #faccio backward indipendente su l1 e su ssim
        # sto solo costruendo il grafo delle deriv parziali -> non sto aggiornando ancora i pesi
        loss.backward()

        # ESEGUI IL GRADIENT CLIPPING QUI!
        params = [p for group in gaussians.optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(params, 1.0)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss_l1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, iteration, opt.opacity_reset_interval, scene_center, background_radius)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # performs a single optimization step, updating the model parameters based on the computed gradients
                # in questo step di opt MODIFICO effettivamente i valori dei pesi -> probab il par delle gaussiane?
                gaussians.exposure_optimizer.step()
                # sets the gradients of all optimized tensors to None
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    # A boolean mask visible is created, indicating which elements in radii are greater than zero.
                    visible = radii > 0
                    # The step() method of gaussians.optimizer is called with the visible mask and the total number of elements in radii. 
                    # This method updates only the visible parameters, which can be more efficient for sparse data
                    gaussians.optimizer.step(visible, radii.shape[0])
                    # The zero_grad() method of gaussians.optimizer is called with set_to_none=True, 
                    # resetting the gradients for the next iteration
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # If use_sparse_adam is False, the code performs standard optimization steps:
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # ----- CUSTOM CODE DEBORAH
                '''
                # Project the Gaussians onto the sphere after each optimization step
                if scene_center is not None and background_radius is not None:
                    #if iteration > opt.densify_from_iter and iteration % (opt.densification_interval) == 0: #new line
                    # meglio ad ogni passo
                    gaussians._xyz, gaussians._scaling = gaussians.project_xyz_on_sphere(scene_center, background_radius)
                    #gaussians._xyz, gaussians._scaling, gaussians._rotation = gaussians.project_gaussian_on_sphere(scene_center, background_radius)
                '''    

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    gaussiansData = gaussians.capture()

    state_params_dict = {}

    for idx, param in enumerate(gaussiansData):
        if isinstance(param, torch.Tensor):
            state_params_dict[f'param_{idx}'] = param.detach().cpu()
            print(f'param_{idx}' + "is a tensor")
        else:
            state_params_dict[f'param_{idx}'] = param
            print(f'param_{idx}' + "is NOT a tensor")

    torch.save(state_params_dict, scene.model_path + "/gaussians_params_first_pass.pt")
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

'''
def anti_spike_loss(xyz, scaling, center, rotation, max_allowed=0.05, lambda_spike=1.0):
    """
    Penalizza gaussiane troppo allungate verso il centro della sfera.
    xyz: (N, 3) posizione delle gaussiane
    scaling: (N, 3) scaling delle gaussiane
    center: (1, 3) centro della sfera
    rotation: (N, 4) quaternioni
    max_allowed: valore massimo consentito per la componente radiale
    lambda_spike: peso della loss
    """
    direction = xyz - center
    norm = torch.norm(direction, dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    radial = direction / norm  # (N, 3)

    # Costruisci la base locale
    from scene.gaussian_model import build_rotation
    local_basis = GaussianModel.get_local_basis(None, radial)  # static call

    # Porta lo scaling nella base locale
    rot_matrix = build_rotation(rotation)  # (N, 3, 3)
    scaling_global = torch.bmm(rot_matrix, scaling.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    scaling_local = torch.bmm(local_basis.transpose(1, 2), scaling_global.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # Penalizza la componente radiale (asse z)
    #spike_loss = torch.relu(scaling_local[:, 2] - max_allowed)
    spike_loss = torch.relu(torch.max(scaling_local, dim=1)[0] - max_allowed)
    return lambda_spike * spike_loss.mean()
'''

'''
def anti_spike_loss(xyz, scaling, center, rotation, max_allowed_diff=0.05, lambda_spike=1.0):
    """
    Penalizza gaussiane troppo allungate in qualsiasi direzione.
    """
    direction = xyz - center
    norm = torch.norm(direction, dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    radial = direction / norm  # (N, 3)

    # Costruisci la base locale
    from scene.gaussian_model import build_rotation
    local_basis = GaussianModel.get_local_basis(None, radial)  # static call

    # Porta lo scaling nella base locale
    rot_matrix = build_rotation(rotation)  # (N, 3, 3)
    scaling_global = torch.bmm(rot_matrix, scaling.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    scaling_local = torch.bmm(local_basis.transpose(1, 2), scaling_global.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # Penalizza spike in qualsiasi direzione
    max_scale = torch.max(scaling_local, dim=1)[0]
    min_scale = torch.min(scaling_local, dim=1)[0]
    spike_loss = torch.relu(max_scale - min_scale - max_allowed_diff)
    return lambda_spike * spike_loss.mean()
'''
'''
def anti_spike_loss(scaling, max_ratio=5.0, lambda_spike=1.0):
    max_scale = torch.max(scaling, dim=1)[0]
    min_scale = torch.min(scaling, dim=1)[0] + 1e-6

    ratio = (max_scale / min_scale)

    # Penalizza solo se il ratio supera la soglia
    spike_penalty = torch.relu(ratio - max_ratio)

    # La penalità cresce anche con la grandezza della gaussiana
    loss = spike_penalty * max_scale
    return lambda_spike * loss.mean()
'''
'''
def anti_spike_loss(scaling, max_ratio=5.0, lambda_spike=1.0, min_penalty=0.1):
    """
    Penalizza gaussiane troppo allungate (spike), con penalità maggiore se la gaussiana è grande.
    Penalizza comunque anche le piccole, ma meno.
    """
    max_scale = torch.max(scaling, dim=1)[0]
    min_scale = torch.min(scaling, dim=1)[0] + 1e-6
    ratio = max_scale / min_scale

    # Penalità base per lo spike (sempre >0 se ratio > max_ratio)
    spike_penalty = torch.relu(ratio - max_ratio)

    # Penalità pesata: cresce con la grandezza, ma ha un minimo anche per gaussiane piccole
    weighted_penalty = spike_penalty * (min_penalty + max_scale)

    return lambda_spike * weighted_penalty.mean()
'''
'''
def compute_anisotropy_loss(scaling, threshold=1.5):
    """
    Calcola una loss per penalizzare le gaussiane allungate ("spikes"),
    con un peso maggiore per le gaussiane più grandi.

    Args:
        scaling (torch.Tensor): Tensore di shape [N, 3] con i fattori di scaling delle N gaussiane.
        threshold (float): La soglia del rapporto max/min scale oltre la quale applicare la penalità.

    Returns:
        torch.Tensor: Il valore della loss (scalare).
    """
    # 1. Calcola l'anisotropia (spikiness)
    # Aggiungiamo epsilon per la stabilità numerica, come hai giustamente fatto tu.
    epsilon = 1e-8
    max_scale, _ = torch.max(scaling, dim=1)
    min_scale, _ = torch.min(scaling, dim=1)
    ratio = max_scale / (min_scale + epsilon)

    # 2. Applica una penalità solo quando il rapporto supera la soglia (approccio "ReLU")
    # In questo modo, le gaussiane quasi-sferiche non vengono penalizzate.
    # Il -1 assicura che un rapporto perfetto di 1 dia 0.
    anisotropy_penalty = F.relu(ratio - threshold)

    # 3. Pesa la penalità in base al volume della gaussiana
    # Usiamo il prodotto degli assi di scaling come proxy del volume.
    # Usiamo .detach() per evitare che il peso stesso influenzi il gradiente in modi indesiderati.
    # Vogliamo che il gradiente provenga solo dalla "forma" (anisotropy_penalty), non dalla dimensione.
    volume_proxy = torch.prod(scaling, dim=1).detach()
    
    # La loss finale è la media delle penalità pesate
    loss = torch.mean(anisotropy_penalty * volume_proxy)
    
    return loss
'''

'''
def compute_hybrid_anisotropy_loss(scaling, size_threshold=5.0):
    """
    Calcola una loss che penalizza SOLO le gaussiane GRANDI che sono allungate.
    - Le gaussiane piccole sono ignorate.
    - La penalità per le gaussiane grandi è proporzionale sia alla loro "spikiness"
      (misurata con la divergenza KL) sia alla loro dimensione.

    Args:
        scaling (torch.Tensor): Tensore [N, 3] con i fattori di scaling.
        size_threshold (float): La soglia di "area" sopra la quale una gaussiana
                                è considerata "grande" e soggetta a penalità.

    Returns:
        torch.Tensor: Il valore della loss (scalare).
    """
    epsilon = 1e-8

    # Passo 1: Calcola la "grandezza" di ogni gaussiana
    # Ordiniamo gli assi di scaling per trovare i due più grandi
    sorted_scales, _ = torch.sort(scaling, dim=1, descending=True)
    s_max = sorted_scales[:, 0]
    s_mid = sorted_scales[:, 1]
    
    # Usiamo il prodotto dei due assi più grandi come proxy dell'area proiettata
    area_proxy = s_max * s_mid

    # Passo 2: Crea una maschera per selezionare solo le gaussiane "grandi"
    large_gaussians_mask = area_proxy > size_threshold
    
    # Se non ci sono gaussiane grandi, la loss è 0
    if not torch.any(large_gaussians_mask):
        return torch.tensor(0.0, device=scaling.device)

    # Filtra per tenere solo le gaussiane che ci interessano
    scaling_large = scaling[large_gaussians_mask]
    area_proxy_large = area_proxy[large_gaussians_mask]

    # Passo 3: Calcola la penalità di forma (KL) SOLO per le gaussiane grandi
    variances_large = scaling_large.pow(2) + epsilon
    arithmetic_mean_variances = torch.mean(variances_large, dim=1)
    geometric_mean_variances = torch.prod(variances_large, dim=1).pow(1/3)
    
    # kl_divergence è la nostra misura di "spikiness"
    kl_divergence = 0.5 * (arithmetic_mean_variances / geometric_mean_variances - 1.0)

    # Passo 4: Pesa la penalità di spikiness in base alla grandezza
    # Usiamo .detach() sull'area per assicurarci che il gradiente venga solo
    # dalla forma (KL) e non dalla dimensione stessa.
    loss_per_gaussian = kl_divergence * area_proxy_large.detach()

    # Passo 5: Calcola la media della loss solo sulle gaussiane penalizzate
    loss = torch.mean(loss_per_gaussian)
    
    return loss
'''

'''
def compute_robust_anisotropy_loss(scaling, size_threshold: float):
    """
    Versione robusta della loss che penalizza le gaussiane grandi e allungate,
    identificando correttamente anche gli "spike" radiali.

    Args:
        scaling (torch.Tensor): Tensore [N, 3] con i fattori di scaling.
        size_threshold (float): La soglia di "grandezza media" sopra la quale
                                una gaussiana è considerata grande.

    Returns:
        torch.Tensor: Il valore della loss (scalare).
    """
    epsilon = 1e-8

    # Passo 1 (MODIFICATO): Calcola la "grandezza" usando la media delle scale.
    # Questa metrica è robusta sia a spike tangenziali che radiali.
    mean_scale = torch.mean(scaling, dim=1)

    # Passo 2: Crea una maschera per selezionare solo le gaussiane "grandi"
    large_gaussians_mask = mean_scale > size_threshold
    
    if not torch.any(large_gaussians_mask):
        return torch.tensor(0.0, device=scaling.device)

    # Filtra per tenere solo le gaussiane che ci interessano
    scaling_large = scaling[large_gaussians_mask]
    mean_scale_large = mean_scale[large_gaussians_mask]

    # Passo 3: Calcola la penalità di forma (KL) SOLO per le gaussiane grandi
    variances_large = scaling_large.pow(2) + epsilon
    arithmetic_mean_variances = torch.mean(variances_large, dim=1)
    geometric_mean_variances = torch.prod(variances_large, dim=1).pow(1/3)
    
    kl_divergence = 0.5 * (arithmetic_mean_variances / geometric_mean_variances - 1.0)

    # Passo 4 (MODIFICATO): Pesa la penalità di spikiness in base alla grandezza media
    loss_per_gaussian = kl_divergence * mean_scale_large.detach()

    # Passo 5: Calcola la media della loss solo sulle gaussiane penalizzate
    loss = torch.mean(loss_per_gaussian)
    
    return loss
'''

'''
def compute_spike_prevention_loss(scaling, spike_threshold: float = 2.0):
    """
    Una loss chirurgica che penalizza SOLO le gaussiane a forma di "spillo"
    (un asse molto più grande degli altri due), indipendentemente dalla loro dimensione.
    Incoraggia le gaussiane a essere sferiche o a forma di disco, ma non a forma di ago.

    Args:
        scaling (torch.Tensor): Tensore [N, 3] con i fattori di scaling.
        spike_threshold (float): La soglia del rapporto s_max/s_mid oltre
                                 la quale applicare la penalità.

    Returns:
        torch.Tensor: Il valore della loss (scalare).
    """
    epsilon = 1e-8
    
    # Passo 1: Ordina gli assi di scaling per trovare s_max, s_mid e s_min
    sorted_scales, _ = torch.sort(scaling, dim=1, descending=True)
    s_max = sorted_scales[:, 0]
    s_mid = sorted_scales[:, 1]

    # Passo 2: Calcola il "rapporto di spuntone" (spike ratio)
    # Questo rapporto è grande solo per gli "spilli"
    spike_ratio = s_max / (s_mid + epsilon)

    # Passo 3: Applica una penalità solo quando il rapporto supera la soglia
    # Questo permette anisotropie lievi ma punisce gli spike evidenti
    spike_penalty = F.relu(spike_ratio - spike_threshold)

    # Passo 4: Pesa la penalità in base alla lunghezza dello spike (s_max)
    # Uno spike più lungo è visivamente peggiore, quindi dovrebbe essere penalizzato di più
    loss_per_gaussian = spike_penalty * s_max.detach()

    # Passo 5: Calcola la media della loss su tutte le gaussiane
    # Nota: la media viene calcolata sul totale. Le gaussiane non-spike avranno
    # una penalità di 0, quindi non contribuiscono alla somma.
    loss = torch.mean(loss_per_gaussian)
    
    return loss
'''
'''
def compute_global_sh_dampening_loss(gaussians):
    """
    Una loss semplice e robusta che penalizza la magnitudine dei coefficienti
    SH di ordine superiore per TUTTE le gaussiane. Questo scoraggia gli effetti
    view-dependent globalmente, rendendo la scena più diffusa e stabile.

    Args:
        gaussians: L'oggetto che contiene le gaussiane.

    Returns:
        torch.Tensor: Il valore della loss di regolarizzazione SH.
    """
    # Estrai i coefficienti SH (di dimensioni [N, 16, 3] per SH di grado 3)
    sh_coefficients = gaussians.get_features

    # Isola i coefficienti di ordine superiore (tutti tranne il primo, che è il colore base DC)
    # features[:, 0, :] è il DC term. Noi vogliamo penalizzare tutto il resto.
    sh_ac_coefficients = sh_coefficients[:, 1:, :]

    # La loss è semplicemente la media dei quadrati di questi coefficienti.
    # Spingere questo valore a zero significa spingere i colori a essere diffusi.
    # Usiamo .abs() o .pow(2) - il quadrato è più comune e penalizza di più i valori grandi.
    loss = torch.mean(sh_ac_coefficients**2)
    
    return loss

'''

def compute_planarity_loss(gaussians, sphere_center):
    """
    Forza le gaussiane ad essere "piatte" e tangenti a una sfera, penalizzando
    quelle il cui asse più corto non è allineato con il raggio della sfera.
    """
    xyz = gaussians.get_xyz
    rotations = gaussians.get_rotation
    scaling = gaussians.get_scaling
    
    # 1. Calcola il vettore radiale per ogni gaussiana
    radial_vectors = xyz - sphere_center
    radial_vectors = F.normalize(radial_vectors, p=2, dim=1)

    # 2. Trova l'asse più corto per ogni gaussiana nel suo sistema di coordinate locale
    # `argsort` ci dà gli indici. L'indice 0 sarà quello del valore più piccolo.
    scaling_sorted_indices = torch.argsort(scaling, dim=1)
    shortest_axis_indices = scaling_sorted_indices[:, 0]

    # Crea i vettori base locali (1,0,0), (0,1,0), (0,0,1)
    local_axes = torch.eye(3, device=xyz.device)
    # Seleziona il vettore base corrispondente all'asse più corto per ogni gaussiana
    shortest_local_axes = local_axes[shortest_axis_indices]

    # 3. Trasforma l'asse più corto dal sistema locale a quello globale
    # Applica la rotazione della gaussiana al suo vettore dell'asse più corto
    rotation_matrices = build_rotation(rotations)
    # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
    shortest_world_axes = torch.bmm(rotation_matrices, shortest_local_axes.unsqueeze(-1)).squeeze(-1)

    # 4. Calcola la penalità di allineamento
    # Il prodotto scalare tra il raggio e l'asse corto ci dice quanto sono allineati.
    # Vogliamo che il valore assoluto sia 1 (perfettamente allineati, non importa il verso).
    # La nostra penalità è (1 - |dot_product|), quindi è 0 per un allineamento perfetto
    # e 1 per un allineamento perpendicolare (il caso peggiore).
    dot_product = torch.sum(radial_vectors * shortest_world_axes, dim=-1)
    alignment_penalty = 1.0 - torch.abs(dot_product)

    # 5. Pesa la penalità in base a quanto la gaussiana è anisotropica
    # Una gaussiana quasi sferica può avere qualsiasi orientamento, non ci interessa.
    # Una gaussiana molto allungata DEVE essere orientata correttamente.
    s_min, _, s_max = torch.sort(scaling, dim=1)[0].T
    # Aggiungiamo epsilon per evitare la divisione per zero
    anisotropy_weight = (s_max / (s_min + 1e-8)).detach()
    
    loss = torch.mean(alignment_penalty * anisotropy_weight)
    
    return loss

def compute_sphere_constraint_loss_L2(gaussians, sphere_center, sphere_radius):
    """
    Calcola una loss L2 (quadratica) che penalizza le gaussiane per non essere 
    sulla superficie di una sfera definita.
    """
    xyz = gaussians.get_xyz
    
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)
    
    # Calcola l'errore come la differenza tra la distanza e il raggio
    distance_error = distance_from_center - sphere_radius
    
    # La loss è la media degli errori AL QUADRATO
    loss = torch.mean(distance_error**2) # o torch.mean(torch.square(distance_error))
    
    return loss

def compute_sphere_constraint_loss_Elastic(gaussians, sphere_center, sphere_radius, lambda_L1=0.1, lambda_L2=1.0):
    """
    Calcola una loss "elastica" (L1 + L2) che combina una penalità quadratica
    con una spinta costante per una convergenza precisa e robusta.
    """
    xyz = gaussians.get_xyz
    
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)
    distance_error = distance_from_center - sphere_radius
    
    # Calcoliamo le due componenti della loss
    loss_L1 = torch.mean(torch.abs(distance_error))
    loss_L2 = torch.mean(distance_error**2)
    
    # Le combiniamo con dei pesi interni
    # Questi pesi controllano la "forma" della valle
    loss = (lambda_L1 * loss_L1) + (lambda_L2 * loss_L2)
    
    return loss

def compute_inward_shell_constraint_loss(gaussians, sphere_center, sphere_radius, inner_radius):
    """
    Penalizza le gaussiane se sono TROPPO DENTRO o TROPPO FUORI da una sfera,
    ma permette loro di esistere liberamente in un guscio definito tra inner_radius e sphere_radius.

    Args:
        gaussians: L'oggetto che contiene le gaussiane.
        sphere_center (torch.Tensor): Centro della sfera.
        sphere_radius (float): Raggio ESTERNO massimo della sfera.
        inner_radius (float): Raggio INTERNO minimo della sfera (limite interno della ciambella).
    """
    xyz = gaussians.get_xyz
    
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)
    
    # Definiamo i limiti del guscio
    outer_radius = sphere_radius
    # inner_radius è ora passato come argomento

    # Calcoliamo due tipi di errore:
    # 1. Errore "verso l'esterno": si attiva solo se la distanza supera il raggio esterno.
    error_outward = F.relu(distance_from_center - outer_radius)
    
    # 2. Errore "verso l'interno": si attiva solo se la distanza è inferiore al raggio interno.
    error_inward = F.relu(inner_radius - distance_from_center)

    # La penalità totale è la somma dei due errori.
    total_error = error_outward + error_inward
    
    # Penalità quadratica (L2) sull'errore totale.
    loss = torch.mean(total_error**2)
    
    return loss
'''
def compute_inward_shell_constraint_loss(gaussians, sphere_center, sphere_radius, shell_thickness_percent=0.1):
    """
    Penalizza le gaussiane se sono TROPPO DENTRO o TROPPO FUORI da una sfera,
    ma permette loro di esistere liberamente in un guscio definito verso l'interno.

    Args:
        gaussians: L'oggetto che contiene le gaussiane.
        sphere_center (torch.Tensor): Centro della sfera.
        sphere_radius (float): Raggio ESTERNO massimo della sfera.
        shell_thickness_percent (float): Spessore del guscio come percentuale del raggio (es. 0.1 per il 10%).
    """
    xyz = gaussians.get_xyz
    
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)
    
    # Definiamo i limiti del guscio
    outer_radius = sphere_radius
    inner_radius = sphere_radius * (1.0 - shell_thickness_percent)

    # Calcoliamo due tipi di errore:
    # 1. Errore "verso l'esterno": si attiva solo se la distanza supera il raggio esterno.
    #    F.relu(x) è x se x > 0, altrimenti 0.
    error_outward = F.relu(distance_from_center - outer_radius)
    
    # 2. Errore "verso l'interno": si attiva solo se la distanza è inferiore al raggio interno.
    error_inward = F.relu(inner_radius - distance_from_center)

    # La penalità totale è la somma dei due errori.
    # Una gaussiana all'interno del guscio avrà entrambi gli errori a zero.
    total_error = error_outward + error_inward
    
    # Applichiamo una penalità quadratica (L2) sull'errore totale.
    loss = torch.mean(total_error**2)
    
    return loss
'''

'''
def compute_image_gradient_loss(rendered_image, gt_image):
    Calcola la differenza L1 tra i gradienti delle immagini per
    incoraggiare la nitidezza in modo più morbido del Laplaciano.
    """
    # I gradienti vengono calcolati per canale.
    # Aggiungiamo una dimensione batch per la convoluzione.
    rendered_batch = rendered_image.unsqueeze(0)
    gt_batch = gt_image.unsqueeze(0)

    # Kernel Sobel per il gradiente orizzontale
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    # Kernel Sobel per il gradiente verticale
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # Calcola i gradienti per entrambe le immagini
    rendered_grad_x = F.conv2d(rendered_batch, sobel_x, padding='same', groups=3)
    rendered_grad_y = F.conv2d(rendered_batch, sobel_y, padding='same', groups=3)
    gt_grad_x = F.conv2d(gt_batch, sobel_x, padding='same', groups=3)
    gt_grad_y = F.conv2d(gt_batch, sobel_y, padding='same', groups=3)

    # La loss è la somma delle differenze L1 dei gradienti orizzontali e verticali
    loss_x = F.l1_loss(rendered_grad_x, gt_grad_x)
    loss_y = F.l1_loss(rendered_grad_y, gt_grad_y)
    
    return loss_x + loss_y
'''
'''
def compute_laplacian_loss(rendered_image, gt_image):
    """
    Calcola la differenza L1 tra le versioni filtrate con il Laplaciano
    delle immagini per incoraggiare la nitidezza.
    """
    # Il kernel si aspetta un'immagine in scala di grigi, quindi convertiamo
    rendered_gray = 0.299 * rendered_image[0] + 0.587 * rendered_image[1] + 0.114 * rendered_image[2]
    gt_gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]
    
    # Aggiungi le dimensioni del batch e del canale
    rendered_gray = rendered_gray.unsqueeze(0).unsqueeze(0)
    gt_gray = gt_gray.unsqueeze(0).unsqueeze(0)

    # Applica il filtro Laplaciano (convoluzione 2D)
    rendered_lap = F.conv2d(rendered_gray, laplacian_kernel, padding='same')
    gt_lap = F.conv2d(gt_gray, laplacian_kernel, padding='same')

    # Calcola la loss L1 sulla "mappa degli spigoli"
    return F.l1_loss(rendered_lap, gt_lap)

def compute_spherical_shell_constraint_loss(gaussians, sphere_center, sphere_radius, shell_thickness):
    """
    Penalizza le gaussiane SOLO se escono da un "guscio sferico"
    di spessore definito. All'interno del guscio, la loss è zero.

    Args:
        gaussians: L'oggetto che contiene le gaussiane.
        sphere_center (torch.Tensor): Centro della sfera.
        sphere_radius (float): Raggio della superficie centrale della sfera.
        shell_thickness (float): La "tolleranza" o metà dello spessore del guscio.
                                 Una gaussiana non viene penalizzata se la sua distanza
                                 dal centro è compresa tra 
                                 (radius - thickness) e (radius + thickness).
    """
    xyz = gaussians.get_xyz
    
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)
    
    # Calcola l'errore di distanza assoluto dalla superficie centrale
    abs_distance_error = torch.abs(distance_from_center - sphere_radius)
    
    # --- LA MODIFICA CHIAVE È QUI ---
    # Usiamo F.relu per creare una "zona morta".
    # La penalità si attiva solo se l'errore assoluto supera lo spessore del guscio.
    # Se abs_distance_error < shell_thickness, il risultato di ( ... ) è negativo,
    # e F.relu lo rende zero.
    penalty = F.relu(abs_distance_error - shell_thickness)
    
    # Applichiamo la penalità quadratica (L2) solo sull'errore che eccede la tolleranza.
    loss = torch.mean(penalty**2)
    
    return loss
'''
'''
def compute_multi_sphere_constraint_loss_L2(gaussians, sphere_center, sphere_radii):
    """
    Calcola una loss L2 che spinge ogni gaussiana verso il guscio sferico
    più vicino tra quelli forniti.

    Args:
        gaussians: L'oggetto che contiene le gaussiane.
        sphere_center (torch.Tensor): Centro comune delle sfere, shape (3,).
        sphere_radii (list of floats): Una lista di raggi per i diversi gusci (es. [50.0, 95.0, 500.0]).

    Returns:
        torch.Tensor: Il valore della loss (scalare).
    """
    # === PASSO 2.1: Calcola la distanza attuale di ogni gaussiana ===
    # Prende le posizioni [N, 3] di tutte le N gaussiane.
    xyz = gaussians.get_xyz
    
    # Calcola la norma euclidea (distanza) di ogni punto dal centro.
    # Il risultato è un tensore 1D con la distanza per ogni gaussiana.
    # -> Shape: [N]
    distance_from_center = torch.norm(xyz - sphere_center, dim=1)

    # === PASSO 2.2: Prepara i raggi dei gusci per il calcolo parallelo ===
    # Converte la nostra lista Python di raggi in un tensore PyTorch.
    # unsqueeze(0) aggiunge una dimensione, trasformando la sua shape da [M] a [1, M]
    # (dove M è il numero di gusci). Questo è FONDAMENTALE per il broadcasting.
    # -> Shape: [1, M]
    radii_tensor = torch.tensor(sphere_radii, device=xyz.device).unsqueeze(0)

    # === PASSO 2.3: Calcola l'errore rispetto a TUTTI i gusci in un colpo solo ===
    # Qui avviene la magia di PyTorch (broadcasting).
    # 1. distance_from_center viene "espanso" da [N] a [N, 1].
    # 2. PyTorch sottrae il tensore [1, M] dal tensore [N, 1].
    #    Il risultato è una matrice [N, M] dove l'elemento (i, j) è la
    #    differenza tra la distanza della i-esima gaussiana e il raggio del j-esimo guscio.
    # -> Shape: [N, M]
    errors_to_all_spheres = torch.abs(distance_from_center.unsqueeze(1) - radii_tensor)

    # === PASSO 2.4: Seleziona solo l'errore più piccolo per ogni gaussiana ===
    # torch.min(..., dim=1) trova il valore minimo lungo ogni riga della nostra matrice.
    # Questo, per ogni gaussiana, seleziona l'errore rispetto al guscio più vicino.
    # Il risultato è un tensore 1D con l'errore minimo per ogni gaussiana.
    # -> Shape: [N]
    min_error, _ = torch.min(errors_to_all_spheres, dim=1)

    # === PASSO 2.5: Calcola la loss finale ===
    # Applichiamo la media dell'errore al quadrato (L2) solo su questi errori minimi.
    # Questo spinge l'optimizer a ridurre a zero la distanza dal guscio più vicino.
    loss = torch.mean(min_error**2)
    
    return loss
'''

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
