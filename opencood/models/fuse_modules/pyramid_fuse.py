# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

from PIL import Image
import torchvision.transforms.functional as TF
import datetime


def gray_image_function(input_tensor, image_info):
    gray_tensor = input_tensor.sum(dim=0, keepdim=True)
    gray_tensor = gray_tensor.mul(255).byte()
    gray_image = TF.to_pil_image(gray_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
    gray_image.save(filename)

def color_image_function(input_tensor, image_info): #for 64 channels
    C, H, W = input_tensor.shape
    device = input_tensor.device
    tensor_1 = input_tensor[:22, :, :].mean(dim=0)
    tensor_2 = input_tensor[22:43, :, :].mean(dim=0)
    tensor_3 = input_tensor[43:64, :, :].mean(dim=0)
    color_mapped_tensor = torch.stack([tensor_1, tensor_2, tensor_3], dim=0)
    color_mapped_tensor = color_mapped_tensor.mul(255).byte()
    color_image = TF.to_pil_image(color_mapped_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_color_image_{now.strftime('%m%d%H%M%S')}.png"
    color_image.save(filename)

def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)

    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)

    record_len : list
        shape: (B)

    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        #print(N)
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        #print(t_matrix[i, :, :, :])
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)

        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device),
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    return out

def padded_weighted_fuse(x, score, record_len, affine_matrix, align_corners, pad_size):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)

    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)

    record_len : list
        shape: (B)

    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)

    pad_size: outer padding distance along each edge
        int
    """
    #padding x and score first
    x = nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size), value=0.0)
    score = nn.functional.pad(score, (pad_size, pad_size, pad_size, pad_size), value=0.0)
    #original part
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        #print(N)
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        #print(t_matrix[i, :, :, :])
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)

        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device),
                                    scores_in_ego)
        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    return out

def occ_projection(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)

    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)

    record_len : list
        shape: (B)

    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)
    """
    if record_len <= 1:
        return score
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    scores_in_ego_list = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        #print(N)
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        #print(t_matrix[i, :, :, :])
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego_list.append(scores_in_ego)
    scores_in_ego_list = torch.stack(scores_in_ego_list)
    return scores_in_ego_list

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck,
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)

        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )
        if 'outpadding' in model_cfg:
            self.outpadding = model_cfg['outpadding']
        else:
            self.outpadding = False

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list

    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list]
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor
                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}
        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            if self.outpadding:
                fused_feature_list.append(padded_weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners, 2**(2-i)))
            else:
                fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        return fused_feature, occ_map_list
