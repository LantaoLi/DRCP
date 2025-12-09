# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistribution

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

from PIL import Image
import torchvision.transforms.functional as TF
import datetime
import copy
import time

def eextract_rectangular(bev_map, fov_radians, h1, w1):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # 计算角度步长
    device = bev_map.device
    fov_radians = fov_radians.item() # to scaler
    half_fov = fov_radians / 2
    step = fov_radians / w1
    angles = torch.linspace(-half_fov, half_fov, w1, device=device)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    radius_max = torch.min(center_x/torch.abs(cos_angles), center_y/torch.abs(sin_angles))
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max.view(1,-1)
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def rrestore_bev_map(rectangular_grid, bev_map_shape, fov_radians):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    fov_radians = fov_radians.item() # to scaler
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    center_y = H // 2
    center_x = W // 2
    half_fov = fov_radians / 2
    # Precompute angles and radii
    angles = torch.linspace(-half_fov, half_fov, w1, device=device)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    # Max radius calculation
    radius_max = torch.min(center_x / torch.abs(cos_angles), center_y / torch.abs(sin_angles))
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1) * radius_max.view(1, -1)
    # Compute grid coordinates
    x_coords = (center_x + radii * cos_angles).round().long().clamp(0, W - 1)
    y_coords = (center_y - radii * sin_angles).round().long().clamp(0, H - 1)
    # Flatten the coordinates and values for scatter_add
    x_coords_flat = x_coords.view(-1)
    y_coords_flat = y_coords.view(-1)
    values_flat = rectangular_grid.view(C, -1)
    # Compute linear indices for scatter_add
    indices = y_coords_flat * W + x_coords_flat
    # Scatter add values into restore_bev_map
    restore_bev_map.view(C, -1).scatter_add_(1, indices.expand(C, -1), values_flat)
    count_map.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=count_map.dtype, device=device))
    # Avoid division by zero
    count_map[count_map == 0] = 1
    restore_bev_map /= count_map
    return restore_bev_map

def accu_extract_canvas(bev_map, fov_radians, h1, w1, rots, trans):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # 计算角度步长
    device = bev_map.device
    fov_radians = fov_radians.item() # to scaler
    half_fov = fov_radians / 2
    angles = torch.linspace(-half_fov, half_fov, w1, device=device)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots = rot[0,0]
    sin_rots = rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles)
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles)
    #trans offset
    camera_offset = trans[0][:2]/0.8
    offset_x, offset_y = camera_offset[0], camera_offset[1]
    #regarding theta
    adjusted_offset_y = torch.where(angles > 0, offset_y, -offset_y)
    radius_max = torch.min((center_x)/torch.abs(cos_angles), (center_y + adjusted_offset_y)/torch.abs(sin_angles))
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max.view(1,-1)
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y + offset_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def accu_restore_canvas(rectangular_grid, bev_map_shape, fov_radians, rots, trans):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    fov_radians = fov_radians.item() # to scaler
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    center_y = H // 2
    center_x = W // 2
    half_fov = fov_radians / 2
    # Precompute angles and radii
    angles = torch.linspace(-half_fov, half_fov, w1, device=device)
    #trans offset
    camera_offset = trans[0][:2]/0.8
    offset_x, offset_y = camera_offset[0], camera_offset[1]
    # rotation matrix for FOV
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots = rot[0,0]
    sin_rots = rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + +
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles)
    #regarding theta
    #adjusted_offset_x = torch.where(angles > 0, offset_x, -offset_x)
    adjusted_offset_y = torch.where(angles > 0, offset_y, -offset_y)
    # Max radius calculation
    # was radius_max = torch.min((center_x - offset_x)/ torch.abs(cos_angles), (center_y - adjusted_offset_y) / torch.abs(sin_angles))
    radius_max = torch.min((center_x)/ torch.abs(cos_angles), (center_y + adjusted_offset_y) / torch.abs(sin_angles))
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1) * radius_max.view(1, -1)
    # Compute grid coordinates
    x_coords = (center_x + radii * cos_angles).round().long().clamp(0, W - 1)
    y_coords = (center_y + offset_y - radii * sin_angles).round().long().clamp(0, H - 1)
    # Flatten the coordinates and values for scatter_add
    x_coords_flat = x_coords.view(-1)
    y_coords_flat = y_coords.view(-1)
    values_flat = rectangular_grid.view(C, -1)
    # Compute linear indices for scatter_add
    indices = y_coords_flat * W + x_coords_flat
    # Scatter add values into restore_bev_map
    restore_bev_map.view(C, -1).scatter_add_(1, indices.expand(C, -1), values_flat)
    count_map.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=count_map.dtype, device=device))
    # Avoid division by zero
    count_map[count_map == 0] = 1
    restore_bev_map /= count_map
    return restore_bev_map

def extract_canvas(bev_map, fov_radians, h1, w1, rots):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # angles and rotations
    device = bev_map.device
    half_fov = fov_radians / 2
    angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
    """
    radius_max = torch.min(center_x/torch.abs(cos_angles), center_y/torch.abs(sin_angles)) #was x/cos y/sin
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max.view(1,-1)
    """
    #static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def restore_canvas(rectangular_grid, bev_map_shape, fov_radians, rots):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    center_x, center_y = W // 2, H // 2
    half_fov = fov_radians / 2
    # Precompute angles and radii
    angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    # rotation matrix for FOV
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + +
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was - +
    # Max radius calculation
    """
    radius_max = torch.min(center_x / torch.abs(cos_angles), center_y / torch.abs(sin_angles)) #was x/cos y/sin
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1) * radius_max.view(1, -1)
    """
    #static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    # Compute grid coordinates
    x_coords = (center_x + radii * cos_angles).round().long().clamp(0, W - 1)
    y_coords = (center_y - radii * sin_angles).round().long().clamp(0, H - 1)
    # Flatten the coordinates and values for scatter_add
    x_coords_flat = x_coords.view(-1)
    y_coords_flat = y_coords.view(-1)
    values_flat = rectangular_grid.view(C, -1)
    # Compute linear indices for scatter_add
    indices = y_coords_flat * W + x_coords_flat
    # Scatter add values into restore_bev_map
    restore_bev_map.view(C, -1).scatter_add_(1, indices.expand(C, -1), values_flat)
    count_map.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=count_map.dtype, device=device))
    # Avoid division by zero
    count_map[count_map == 0] = 1
    restore_bev_map /= count_map
    return restore_bev_map

def extract_undistorted_canvas(bev_map, fov_radians, h1, w1, rots, fx, cx):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # angles and rotations
    device = bev_map.device
    half_fov = fov_radians / 2
    #angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    # calculating angles in undistored way
    pixel_indices = torch.arange(0, w1, device=device)
    pixel_width_original = 1920.0/w1
    original_pixel_indices = pixel_indices*pixel_width_original + pixel_width_original/2.0
    pixel_offsets = (original_pixel_indices - cx)/fx
    angles = torch.atan(pixel_offsets)
    #print(angles)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
    # static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def restore_undistorted_canvas(rectangular_grid, bev_map_shape, fov_radians, rots, fx, cx):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    center_x, center_y = W // 2, H // 2
    half_fov = fov_radians / 2
    # Precompute angles and radii
    #angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    # calculating angles in undistored way
    pixel_indices = torch.arange(0, w1, device=device)
    pixel_width_original = 1920.0/w1
    original_pixel_indices = pixel_indices*pixel_width_original + pixel_width_original/2.0
    pixel_offsets = (original_pixel_indices - cx)/fx
    angles = torch.atan(pixel_offsets)
    # rotation matrix for FOV
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + +
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was - +
    #static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    # Compute grid coordinates
    x_coords = (center_x + radii * cos_angles).round().long().clamp(0, W - 1)
    y_coords = (center_y - radii * sin_angles).round().long().clamp(0, H - 1)
    # Flatten the coordinates and values for scatter_add
    x_coords_flat = x_coords.view(-1)
    y_coords_flat = y_coords.view(-1)
    values_flat = rectangular_grid.view(C, -1)
    # Compute linear indices for scatter_add
    indices = y_coords_flat * W + x_coords_flat
    # Scatter add values into restore_bev_map
    restore_bev_map.view(C, -1).scatter_add_(1, indices.expand(C, -1), values_flat)
    count_map.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=count_map.dtype, device=device))
    # Avoid division by zero
    count_map[count_map == 0] = 1
    restore_bev_map /= count_map
    return restore_bev_map

class SlicingCrossAttention(nn.Module):
    def __init__(self, C1, C2, num_heads, dropout=0.1):
        super(SlicingCrossAttention, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.num_heads = num_heads
        #qkv linear trans
        self.query_proj = nn.Linear(C1, C1)
        self.key_proj = nn.Linear(C2, C1)
        self.value_proj = nn.Linear(C2, C1)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiheadAttention(C1, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            a_rect = eextract_rectangular(a, fov_radians, h1, w2)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = rrestore_bev_map(a_enhanced, a.shape, fov_radians)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list

class DepthAwareCrossAttention(nn.Module):
    def __init__(self, model_cfg):
        super(DepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        #fxs, cxs not used iin this function design
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            a_rect = extract_canvas(a, fov_radians, h1, w2, rots)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            a_rect += self.pos_encoding_a
            b_rect += self.pos_encoding_b
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list

class UndistortedDepthAwareCrossAttention(nn.Module):
    def __init__(self, model_cfg):
        super(UndistortedDepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            fx, cx = fxs[i], cxs[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            # Extract trapezoidal and convert to rectangular with width w2
            a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            a_rect += self.pos_encoding_a
            b_rect += self.pos_encoding_b
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list

class UndistortedDepthAwareMHAConvFusion(nn.Module):
    def __init__(self, model_cfg):
        super(UndistortedDepthAwareMHAConvFusion, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        #self.fusion_conv = nn.Conv2d(self.C1+self.C1, self.C1, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = nn.Conv2d(self.C1+self.C1, self.C1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            #for LCC train and eval
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            fx, cx = fxs[i], cxs[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            # Extract trapezoidal and convert to rectangular with width w2
            a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            a_rect += self.pos_encoding_a
            b_rect += self.pos_encoding_b
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
            new_a = torch.cat([a, a_enhanced], dim=0)
            new_a = self.fusion_conv(new_a.unsqueeze(0)).squeeze(0)
            enhanced_a_list.append(new_a)
        return enhanced_a_list

class PyramidDepthAwareCrossAttention(nn.Module):
    def __init__(self, model_cfg):
        super(PyramidDepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        self.pyramid_levels = model_cfg["pyramid_levels"]
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.qs = nn.ModuleList([nn.Linear(self.C1, self.C1) for i in range(self.pyramid_levels)])
        self.ks = nn.ModuleList([nn.Linear(self.C2, self.C1) for i in range(self.pyramid_levels)])
        self.vs = nn.ModuleList([nn.Linear(self.C2, self.C1) for i in range(self.pyramid_levels)])
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attentions = nn.ModuleList([MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True) for i in range(self.pyramid_levels)])
        #up and down sampling stuff
        self.downsamples_a = nn.ModuleList([nn.Identity() if i==0 else nn.Conv2d(self.C1, self.C1, kernel_size=3, stride=2, padding=1) for i in range(self.pyramid_levels)])
        self.upsamples = nn.ModuleList([nn.Identity() if i==0 else nn.ConvTranspose2d(self.C1, self.C1, kernel_size=3, stride=2, padding=1) for i in range(self.pyramid_levels)])
        self.downsamples_b = nn.ModuleList([nn.Identity() if i==0 else nn.Conv2d(self.C2, self.C2, kernel_size=3, stride=2, padding=1) for i in range(self.pyramid_levels)])

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []

        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            fx, cx = fxs[i], cxs[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2

            a_rect = a_rect + self.pos_encoding_a
            b_rect = b_rect + self.pos_encoding_b

            for level, (downsample_a, downsample_b, upsample, qi, ki, vi, atteni) in enumerate(zip(self.downsamples_a, self.downsamples_b, self.upsamples, self.qs, self.ks, self.vs, self.attentions)):
                if level == 0:
                    a_scale = a_rect
                    b_scale = b_rect
                else:
                    a_scale = downsample_a(a_rect.permute(2,1,0)).permute(2,1,0)
                    b_scale = downsample_b(b_rect.permute(2,1,0)).permute(2,1,0)
                q = qi(a_scale) #whc
                k = ki(b_scale) #whc
                v = vi(b_scale) #whc
                attn_output, _ = atteni(q, k, v) #whc
                attn_output = attn_output.permute(2,1,0) #chw
                attn_output = upsample(attn_output.unsqueeze(0)).squeeze(0)

                if level != 0:
                    attn_output = F.interpolate(attn_output.unsqueeze(0), size = (128, 256), mode = 'bilinear', align_corners=True)
                    attn_output = attn_output.squeeze(0)

                a_enhanced = a_enhanced + attn_output

            a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list

class LevelDepthAwareCrossAttention(nn.Module): # this module will be used as a sub-module of pyramidcrossfusion module
    def __init__(self, model_cfg):
        super(LevelDepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        self.pyramid_levels = model_cfg["pyramid_levels"]
        if "distorted" not in model_cfg or model_cfg["distorted"] == False:
            self.distorted = False
        else:
            self.distorted = True
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.qs = nn.ModuleList([nn.Linear(self.C1*(2**i), self.C1*(2**i)) for i in range(self.pyramid_levels)])
        self.ks = nn.ModuleList([nn.Linear(self.C2*(2**i), self.C1*(2**i)) for i in range(self.pyramid_levels)])
        self.vs = nn.ModuleList([nn.Linear(self.C2*(2**i), self.C1*(2**i)) for i in range(self.pyramid_levels)])
        self.pos_encoding_as = nn.ParameterList([nn.Parameter(torch.randn(1, int(self.h1/(2**i)), self.C1*(2**i))) for i in range(self.pyramid_levels)])
        self.pos_encoding_bs = nn.ParameterList([nn.Parameter(torch.randn(1, int(self.h2/(2**i)), self.C2*(2**i))) for i in range(self.pyramid_levels)])
        self.attentions = nn.ModuleList([MultiheadAttention(self.C1*(2**i), num_heads=self.num_heads, dropout=self.dropout, batch_first=True) for i in range(self.pyramid_levels)])

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        """
        list_a in form of [3*(t_i)], t_i in shape (sum_num_agent_perbatch, Ci, Hi, Wi)
        list_b in form of [sum_num_agent_perbatch*[t_1, t_2, t_3]], t_i in shape (Ci, Hi, Wi)
        """
        if self.pyramid_levels != len(list_a) or self.pyramid_levels != len(list_b): #levels error
            print("levels error")
            return list_a
        n = len(list_b[0])
        if n != list_a[0].shape[0]: #sum of agents per batch error
            print(len(list_b[0]))
            print(list_a[0].shape)
            print("sum of agents per batch error")
            return list_a
        enhanced_a_list = []

        for i, (pos_encoding_a, pos_encoding_b, qi, ki, vi, atteni) in enumerate(zip(self.pos_encoding_as, self.pos_encoding_bs, self.qs, self.ks, self.vs, self.attentions)):
            a_i = list_a[i]
            b_i = list_b[i]
            a_i_list = []
            for j in range(n):
                a = a_i[j,:,:,:]
                b = b_i[j]
                fov_radians = fov_list[j]
                rots = rots_list[j]
                trans = trans_list[j]
                fx, cx = fxs[j], cxs[j]
                C1, h1, w1 = a.shape
                C2, h2, w2 = b.shape
                if self.distorted:
                    a_rect = extract_canvas(a, fov_radians, h1, w2, rots)
                else:
                    a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
                a_enhanced = torch.zeros_like(a_rect)
                a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
                b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2

                a_scale = a_rect + pos_encoding_a
                b_scale = b_rect + pos_encoding_b
                q = qi(a_scale) #whc
                k = ki(b_scale) #whc
                v = vi(b_scale) #whc
                attn_output, _ = atteni(q, k, v) #whc
                attn_output = attn_output.permute(2,1,0) #chw
                a_enhanced = a_enhanced + attn_output
                if self.distorted:
                    a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots)
                else:
                    a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
                a_i_list.append(a+a_enhanced)
            #gray_image_function(a_enhanced, "debug_feature_map/m1_enhanced")
            enhanced_a_list.append(torch.stack(a_i_list))
            #gray_image_function(a + a_enhanced, "debug_feature_map/m1_enhanced_combined")
        return enhanced_a_list

class UndistortedDepthAwareX2XAttention(nn.Module):
    def __init__(self, model_cfg):
        super(UndistortedDepthAwareX2XAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #self.depth_levels = depth_levels
        #QKV config for lidar-enhancing-camera
        self.query_proj_lc = nn.Linear(self.C2, self.C2)
        self.key_proj_lc = nn.Linear(self.C1, self.C2)
        self.value_proj_lc = nn.Linear(self.C1, self.C2)
        self.pos_encoding_l_lc = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_c_lc = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention_lc = MultiheadAttention(self.C2, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        #QKV config for camera-enhancing-lidar
        self.query_proj_cl = nn.Linear(self.C1, self.C1)
        self.key_proj_cl = nn.Linear(self.C2, self.C1)
        self.value_proj_cl = nn.Linear(self.C2, self.C1)
        self.pos_encoding_l_cl = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_c_cl = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention_cl = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            fx, cx = fxs[i], cxs[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            # Extract trapezoidal and convert to rectangular with width w2
            a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2

            a_rect_lc = a_rect + self.pos_encoding_l_lc
            b_rect_lc = b_rect + self.pos_encoding_c_lc
            q_lc = self.query_proj_lc(b_rect_lc) #w1h2c2
            k_lc = self.key_proj_lc(a_rect_lc) #w1h1c2
            v_lc = self.value_proj_lc(a_rect_lc) #w1h1c2
            attn_output_lc, _ = self.attention_lc(q_lc, k_lc, v_lc) #w1h2c2

            b_enhanced_rect = b_rect + attn_output_lc

            a_rect_cl = a_rect + self.pos_encoding_l_cl
            b_rect_cl = b_enhanced_rect + self.pos_encoding_c_cl
            q_cl = self.query_proj_cl(a_rect_cl) #w1h1c1
            k_cl = self.key_proj_cl(b_rect_cl) #w1h2c1
            v_cl = self.value_proj_cl(b_rect_cl) #w1h2c1
            attn_output_cl, _ = self.attention_cl(q_cl, k_cl, v_cl) #w1h1c1

            a_enhanced = attn_output_cl.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list

class UndistortedDepthAwareCrossConvAttention(nn.Module):
    def __init__(self, model_cfg):
        super(UndistortedDepthAwareCrossConvAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        #new conv for cross-attn genrated BEV
        self.conv = nn.Conv2d(self.C1, self.C1, kernel_size=3, padding=1)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            fx, cx = fxs[i], cxs[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            # Extract trapezoidal and convert to rectangular with width w2
            a_rect = extract_undistorted_canvas(a, fov_radians, h1, w2, rots, fx, cx)
            # Initialize enhanced version of a_rect
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            a_rect += self.pos_encoding_a
            b_rect += self.pos_encoding_b
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1

            #conv on a_enhanced
            a_enhanced = self.conv(a_enhanced)

            a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list
