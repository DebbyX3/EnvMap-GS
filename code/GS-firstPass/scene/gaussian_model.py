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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        # initialize an empty tensor of just ONE scalar
        # position of the gaussian
        self._xyz = torch.empty(0)
        ''' 
        features_dc represents diffuse color which is kind of the base color of the object 
        if you're not considering any lighting effects. This would correspond to the lowest degree of 
        the spherical harmonics where it is just constant (no view dependence). 
        With this interpretation features_rest would, well, be the rest of the spherical harmonic coefficients.
        
        When dealing with direction dependent lighting effects, spherical harmonics are used. 
        These are special functions that can be combined to represent any function on the surface of a sphere, 
        which in our case would be the direction dependent appearance of a given Gaussian. 
        In my understanding, the higher the degree the higher frequencies (finer details) can be represented.
        '''
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        
        # scaling (dimension) of the gaussian
        self._scaling = torch.empty(0)

        # gaussian rotation
        self._rotation = torch.empty(0)
        
        # opacity of the gaussian
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Aggiungiamo un buffer per contare quante volte una gaussiana
        # è stata vista (cioè è passata il culling del frustum).
        self.times_seen = torch.zeros(0, dtype=torch.int32, device="cuda")

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # converte lo scaling da scala log a scala normale!
    # cioè prende lo scaling log e ne fa l'esponenziale
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # call this function when we have a points3d.ply to load the point cloud from
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.times_seen = torch.zeros(self.get_xyz.shape[0], dtype=torch.int32, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        self.times_seen = self.times_seen[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, 
                              new_opacities, new_scaling, new_rotation, new_tmp_radii, 
                              scene_center = None, background_radius = None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling" : new_scaling,
             "rotation" : new_rotation}

        # lascia stare per ora
        '''
        # Project the new Gaussians onto the sphere using the dedicated function
        if scene_center is not None and background_radius is not None:
            d["xyz"], d["scaling"], d["rotation"] = self.project_gaussian_on_sphere(scene_center, background_radius, xyz=d["xyz"], 
                                                                                     scaling=d["scaling"], rotation=d["rotation"])
            #old deb #d["xyz"] = self.project_xyz_on_sphere(scene_center, background_radius, xyz=d["xyz"])
        '''

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # ==================== AGGIUNGI QUESTA LOGICA QUI ====================
        # Creiamo un tensore di zeri per i nuovi contatori. La sua dimensione
        # è data dal numero di nuove gaussiane, che possiamo ottenere da new_xyz.shape[0].
        new_times_seen = torch.zeros(new_xyz.shape[0], dtype=torch.int32, device="cuda")
        self.times_seen = torch.cat((self.times_seen, new_times_seen), dim=0)
        # =================================================================

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, 
                          scene_center = None, background_radius = None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacity, new_scaling, new_rotation, new_tmp_radii,
                                   scene_center, background_radius)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, 
                          scene_center = None, background_radius = None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacities, new_scaling, new_rotation, new_tmp_radii, 
                                   scene_center, background_radius)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, iteration, prune_zombies_at_iter,
                          scene_center=None, background_radius=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent, scene_center=scene_center, background_radius=background_radius)
        self.densify_and_split(grads, max_grad, extent, scene_center=scene_center, background_radius=background_radius)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # max screen size è uguale a 20 ED esiste SOLO se 
        # l'iterazione corrente è maggiore di opt.opacity_reset_interval, 
        # che è generalmente 3000
        '''
        This code snippet is part of a conditional block that executes 
        only if max_screen_size is set (i.e., not None or zero). 
        Its purpose is to update a mask (prune_mask) that likely determines 
        which points or elements should be excluded ("pruned") from further 
        processing, based on their size or scaling.
        '''
        #ORIGINAL CODE
        # lo tolgo per ora
        '''
        if max_screen_size:
            # Salva un vettore di booleani che indicano le gaussiane che hanno un raggio maggiore di max screen size, ovvero 20
            # This results in a boolean tensor where each element is True if the corresponding radius exceeds the maximum allowed screen size.
            # original # big_points_vs = self.max_radii2D > max_screen_size
            big_points_vs = self.max_radii2D > (max_screen_size * 2)
            # salva vettore di booleani che indicano le gaussiane che hanno uno scaling maggiore di 10% dell'extent della scena
            # This results in a boolean tensor where each element is True if the corresponding scaling exceeds
            #original # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big_points_ws = self.get_scaling.max(dim=1).values > 0.2 * extent
            # si fa poi l'or logico tra i due vettori di booleani
            # so any point that was already marked for pruning, or is too large in either screen size or scaling, will be included in the updated mask
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        '''

        # ==================== INIZIO NOSTRA MODIFICA ====================
        
        if prune_zombies_at_iter > 0 and iteration % prune_zombies_at_iter == 0:
            print(f"[GARBAGE COLLECTION] Esecuzione del pruning delle gaussiane 'zombie' a iterazione {iteration}.")

            # Una soglia di visibilità. Se una gaussiana è stata vista meno di, diciamo,
            # 1 volta ogni 3000 iterazioni, è quasi certamente inutile.
            # Iniziare con 1 è la scelta più sicura.
            SEEN_THRESHOLD = 1 
            
            prune_mask_unseen = self.times_seen < SEEN_THRESHOLD
            
            num_unseen = torch.sum(prune_mask_unseen).item()
            if num_unseen > 0:
                print(f"[GARBAGE COLLECTION] Trovate e contrassegnate per la rimozione {num_unseen} gaussiane mai viste.")

            # Combina la maschera di pruning originale con la nostra maschera anti-zombie
            # Una gaussiana viene rimossa se è quasi trasparente OPPURE se è una zombie
            final_prune_mask = torch.logical_or(prune_mask, prune_mask_unseen)
        else:
            # Per tutte le altre iterazioni, usa solo il pruning dell'opacità
            final_prune_mask = prune_mask
        
        # ===================== FINE NOSTRA MODIFICA =====================

        # --- INIZIO CODICE ESISTENTE (USA LA NUOVA MASCHERA) ---
        # NOTA: assicurati di usare 'final_prune_mask' da qui in poi, 
        # al posto della vecchia 'prune_mask'.

        #le gaussiane scelte sopra sono eliminate
        self.prune_points(final_prune_mask)
        

        #self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # first projection version - sposta solo la gaussiana nella pos più vicina e regola lo scaling
    def project_xyz_on_sphere(self, center, radius, xyz=None, scaling=None):
        '''
        Projects each Gaussian onto the surface of the sphere with center `center` and radius `radius`.
        Args:
            center (torch.Tensor): shape (3,) or (1,3), device same as self._xyz
            radius (float): radius of the sphere
            xyz (torch.Tensor, optional): shape (N,3), device same as self._xyz. If None, uses self._xyz
            scaling (torch.Tensor, optional): shape (N,3), device same as self._scaling. If None, uses self._scaling
        '''
        # Allows projecting an arbitrary xyz tensor, default: self._xyz
        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling

        # Sottrai i due vettori (ottieni il vettore spostamento) e calcola la lunghezza (norma euclidea) di quel vettore
        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)

        projected_xyz = center + radius * direction / norm

        # da fabio:
        # fattore di scala è il raggio della sfera diviso la distanza dal centro PRIMA della proiezione
        # moltiplicare lo scaling attuale per il rapporto delle due distanze
        #scaling_factor = radius / norm # dist dal centro prima della proiezione è la norma del vettore che esce dalla differenza tra centro e la posizione
        #projected_scaling = scaling * scaling_factor
        projected_scaling = scaling

        return projected_xyz, projected_scaling

    def rotation_matrix_to_quaternion(self, R):
        # R: (N, 3, 3)
        # Return: (N, 4)
        import numpy as np
        from scipy.spatial.transform import Rotation as SciRot
        R_np = R.detach().cpu().numpy()
        quats = SciRot.from_matrix(R_np).as_quat()  # (N, 4), format [x, y, z, w]
        quats = torch.from_numpy(quats).to(R.device, R.dtype)
        return quats
    
    # Computes a local orthonormal basis (radial, tangential1, tangential2)
    def get_local_basis(self, radial):
        # Choose an arbitrary vector not parallel to radial
        up = torch.tensor([0, 0, 1], device=radial.device, dtype=radial.dtype).expand_as(radial).clone()
        mask = (torch.abs(radial - up).sum(dim=1) < 1e-3)
        up[mask] = torch.tensor([0, 1, 0], device=radial.device, dtype=radial.dtype)
        tangent1 = torch.nn.functional.normalize(torch.cross(up, radial, dim=1), dim=1)
        tangent2 = torch.nn.functional.normalize(torch.cross(radial, tangent1, dim=1), dim=1)
        # Returns a matrix (N, 3, 3): [tangent1, tangent2, radial]
        return torch.stack([tangent1, tangent2, radial], dim=2)

    '''
    # le gaussiane sembrano fettucce lunghe e sottili verso la sfera! niente spike verso il centro almeno
    def project_gaussian_on_sphere(self, center, radius, xyz=None, scaling=None, rotation=None):
        #Projects each Gaussian onto the surface of the sphere with center `center` and radius `radius`.
        #Args:
        #    center (torch.Tensor): shape (3,) or (1,3), device same as self._xyz
        #    radius (float): radius of the sphere
        
        # Allows projecting an arbitrary xyz tensor, default: self._xyz
        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        # Allows projecting an arbitrary scaling tensor, default: self._scaling
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling
        # Allows projecting an arbitrary rotation tensor, default: self._rotation
        if hasattr(self, "_rotation") and rotation is None:
            rotation = self._rotation

        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected_xyz = center + radius * direction / norm
        radial = direction / norm  # (N, 3)

        local_basis = self.get_local_basis(radial)  # (N, 3, 3)

        # Build the rotation matrix: align z-axis with the radial direction
        projected_rotation = self.rotation_matrix_to_quaternion(local_basis)  # (N, 4)

        # Scaling: keep tangential values, squash the radial value
        if scaling is not None:
            # Keep the first two (tangential) scaling values unchanged
            scaling_tangential = scaling[:, :2]
            scaling_radial = torch.full((scaling.shape[0], 1), 0.01, device=scaling.device)  # squash the radial scaling
            projected_scaling = torch.cat([scaling_tangential, scaling_radial], dim=1)  # (N, 3)
        else:
            projected_scaling = None

        return projected_xyz, projected_scaling, projected_rotation

    # le gaussiane hanno i valori di scaling tutti uguali, quindi sono sferiche - non benissimo, percè le voglio libere
    def project_gaussian_on_sphere(self, center, radius, xyz=None, scaling=None, rotation=None):
        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling
        if hasattr(self, "_rotation") and rotation is None:
            rotation = self._rotation

        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected_xyz = center + radius * direction / norm
        radial = direction / norm  # (N, 3)

        local_basis = self.get_local_basis(radial)  # (N, 3, 3)
        projected_rotation = self.rotation_matrix_to_quaternion(local_basis)  # (N, 4)

        if scaling is not None:
            # Imposta tutti i valori uguali (sferica)
            mean_scale = scaling.mean(dim=1, keepdim=True)
            projected_scaling = mean_scale.repeat(1, 3)
        else:
            projected_scaling = None

        return projected_xyz, projected_scaling, projected_rotation    
    
    # le gaussiane sono ruotate per allineare l'asse radiale alla sfera, ma mantengono lo scaling originale
    def project_gaussian_on_sphere(self, center, radius, xyz=None, scaling=None, rotation=None):
        
        #Projects each Gaussian onto the surface of the sphere with center `center` and radius `radius`.
        #Rotates the gaussian so its main axis is tangent to the sphere, but keeps the original scaling.
        
        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling
        if hasattr(self, "_rotation") and rotation is None:
            rotation = self._rotation
    
        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected_xyz = center + radius * direction / norm
        radial = direction / norm  # (N, 3)
    
        local_basis = self.get_local_basis(radial)  # (N, 3, 3)
        projected_rotation = self.rotation_matrix_to_quaternion(local_basis)  # (N, 4)
    
        # Mantieni lo scaling originale
        projected_scaling = scaling
    
        return projected_xyz, projected_scaling, projected_rotation
    # '''
    
    # Devi ruotare lo scaling nella base locale della sfera, schiacciare la 
    # componente radiale, e poi riportare lo scaling nella base della 
    # gaussiana proiettata.
    # Così elimini gli spike, qualunque sia la rotazione originale.
    def squash_radial_scaling_with_transport(self, scaling, transported_rotation, radial): # NO
        # Costruisci la base locale: [tangent1, tangent2, radial]
        local_basis = self.get_local_basis(radial)  # (N, 3, 3)

        print("Local basis axes (first gaussian):")
        print("Tangent1:", local_basis[0, :, 0])
        print("Tangent2:", local_basis[0, :, 1])
        print("Radial  :", local_basis[0, :, 2])
        print("Radial direction (should point from sphere center to gaussian):", radial[0])

        '''
        # Costruisci la matrice di rotazione dalla quaternion
        rot_matrix = build_rotation(rotation)  # (N, 3, 3)
        # Porta lo scaling nella base locale della sfera
        scaling_global = torch.bmm(rot_matrix, scaling.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        scaling_local = torch.bmm(local_basis.transpose(1, 2), scaling_global.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        # Schiaccia la componente radiale (asse z)
        scaling_local[:, 2] = 0.01
        print("Scaling local (after squash):", scaling_local[0])
        # Riporta lo scaling nella base globale della sfera
        scaling_global_new = torch.bmm(local_basis, scaling_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        # Riporta lo scaling nella base della gaussiana proiettata
        scaling_final = torch.bmm(rot_matrix.transpose(1, 2), scaling_global_new.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        return scaling_final
        '''

        rot_matrix = build_rotation(transported_rotation)  # (N, 3, 3)
        # Porta lo scaling nella base locale della sfera
        scaling_local = torch.bmm(local_basis.transpose(1, 2), torch.bmm(rot_matrix, scaling.unsqueeze(-1))).squeeze(-1)  # (N, 3)
        # Schiaccia la componente radiale (asse z)
        scaling_local[:, 1] = scaling_local[:, 1].clamp(max=0.01)
        # Riporta lo scaling nella base della gaussiana proiettata (parallel transported)
        scaling_final = torch.bmm(local_basis, scaling_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        scaling_final = torch.bmm(rot_matrix.transpose(1, 2), scaling_final.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        return scaling_final

    #versione col parallel transport e squash radiale - NO
    '''
    def project_gaussian_on_sphere(self, center, radius, xyz=None, scaling=None, rotation=None):
        #Projects each Gaussian onto the surface of the sphere with center `center` and radius `radius`.
        #Uses parallel transport for rotation, keeps original scaling.

        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling
        if hasattr(self, "_rotation") and rotation is None:
            rotation = self._rotation

        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected_xyz = center + radius * direction / norm
        radial = direction / norm  # (N, 3)

        # Parallel transport the rotation
        transported_rotation = self.parallel_transport_rotation(xyz, projected_xyz, rotation)
        # Squash the radial scaling
        projected_scaling = self.squash_radial_scaling_with_transport(scaling, transported_rotation, radial)

        print("Scaling local (after squash):", projected_scaling[0])

        return projected_xyz, projected_scaling, transported_rotation
    '''  

    #versione col parallel transport SENZA squash radiale - NO
    def project_gaussian_on_sphere(self, center, radius, xyz=None, scaling=None, rotation=None):
        #Projects each Gaussian onto the surface of the sphere with center `center` and radius `radius`.
        #Uses parallel transport for rotation, keeps original scaling.

        if hasattr(self, "_xyz") and xyz is None:
            xyz = self._xyz
        if hasattr(self, "_scaling") and scaling is None:
            scaling = self._scaling
        if hasattr(self, "_rotation") and rotation is None:
            rotation = self._rotation

        direction = xyz - center
        norm = torch.norm(direction, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected_xyz = center + radius * direction / norm
        radial = direction / norm  # (N, 3)

        # Parallel transport the rotation
        transported_rotation = self.parallel_transport_rotation(xyz, projected_xyz, rotation)
        projected_scaling = scaling # NON schiacciare più la componente radiale

        return projected_xyz, projected_scaling, transported_rotation
    
    # Compose the original rotation with the transport quaternion
    def quat_mult(self, q, r):
        # q, r: (N, 4) in (w, x, y, z)
        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=1)
    
    # NO
    def parallel_transport_rotation(self, xyz, projected_xyz, rotation):
        """
        Parallel transport the rotation from xyz to projected_xyz on the sphere.
        xyz: (N, 3) original positions
        projected_xyz: (N, 3) projected positions on the sphere
        rotation: (N, 4) quaternion (w, x, y, z)
        Returns: (N, 4) quaternion after transport
        """
        v1 = torch.nn.functional.normalize(xyz, dim=1)
        v2 = torch.nn.functional.normalize(projected_xyz, dim=1)
        axis = torch.cross(v1, v2, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = torch.where(axis_norm > 1e-6, axis / axis_norm, torch.zeros_like(axis))
        dot = (v1 * v2).sum(dim=1, keepdim=True).clamp(-1, 1)
        angle = torch.acos(dot)

        # noooooooooooooooooooooooo non funziona così
        # Build quaternion for the rotation between v1 and v2
        half_angle = angle / 2
        sin_half_angle = torch.sin(half_angle)
        rot_quat = torch.cat([
            torch.cos(half_angle),           # w
            axis * sin_half_angle            # x, y, z
        ], dim=1)  # (N, 4)

        transported_rotation = self.quat_mult(rot_quat, rotation)
        transported_rotation = torch.nn.functional.normalize(transported_rotation, dim=1)
        return transported_rotation