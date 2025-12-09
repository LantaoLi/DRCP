# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistribution

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
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

def gray_image_function(input_tensor, image_info):
    gray_tensor = input_tensor.sum(dim=0, keepdim=True)
    gray_tensor = gray_tensor.mul(255).byte()
    gray_image = TF.to_pil_image(gray_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
    gray_image.save(filename)

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
    rot = rots[:2,:2].to(torch.float32)
    rot = torch.tensor([[0, 1],[-1, 0]], device = device, dtype=torch.float32)@rot #was 0 -1 1 0
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
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
    rot = rots[:2,:2].to(torch.float32)
    rot = torch.tensor([[0, 1],[-1, 0]], device = device, dtype=torch.float32)@rot #was 0 -1 1 0
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

def extract_undistorted_canvas(bev_map, fov_radians, h1, w1, rots, fx, cx):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # angles and rotations
    device = bev_map.device
    half_fov = fov_radians / 2
    # calculating angles in undistored way
    pixel_indices = torch.arange(0, w1, device=device)
    pixel_width_original = 800.0/w1
    original_pixel_indices = pixel_indices*pixel_width_original + pixel_width_original/2.0
    pixel_offsets = (original_pixel_indices - cx)/fx
    angles = torch.atan(pixel_offsets)
    rot = rots[:2,:2].to(torch.float32)
    rot = torch.tensor([[0, 1],[-1, 0]], device = device, dtype=torch.float32)@rot #was 0 -1 1 0
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
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
    # calculating angles in undistored way
    pixel_indices = torch.arange(0, w1, device=device)
    pixel_width_original = 800.0/w1
    original_pixel_indices = pixel_indices*pixel_width_original + pixel_width_original/2.0
    pixel_offsets = (original_pixel_indices - cx)/fx
    angles = torch.atan(pixel_offsets)
    # rotation matrix for FOV
    rot = rots[:2,:2].to(torch.float32)
    rot = torch.tensor([[0, 1],[-1, 0]], device = device, dtype=torch.float32)@rot #was 0 -1 1 0
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
        cam_num = 4
        print(n)
        print(len(list_b))
        if len(list_b) != n*cam_num or len(fov_list) != n: #"All input lists must have the same length."
            print("not aligned length of data!!!")
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            for j in range(cam_num):
                b = list_b[i*cam_num + j]
                fov_radians = fov_list[i][j]
                rots = rots_list[i][j]
                trans = trans_list[i][j]

                # determine whether apply side cameras and reverse rots
                if j==1 or j==2:
                    continue
                    rots = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]], device = rots.device, dtype=torch.float32)@rots
                    #continue
                if len(a.shape) != 3:
                    print("a.shape != 3, a.shape is:")
                    print(a.shape)
                    a = a[0]
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
                a = a + a_enhanced
            enhanced_a_list.append(a + a_enhanced) # a or a + a_enhanced?, could lead to minor performance differences
        return enhanced_a_list

class UndistortedDepthAwareCrossAttention(nn.Module):
    def __init__(self, model_cfg):
        super(UndistortedDepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list, fxs, cxs):
        n = len(list_a)
        cam_num = 4
        print(n)
        print(len(list_b))
        if len(list_b) != n*cam_num or len(fov_list) != n: #"All input lists must have the same length."
            print("not aligned length of data!!!")
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            for j in range(cam_num):
                b = list_b[i*cam_num + j]
                fov_radians = fov_list[i][j]
                rots = rots_list[i][j]
                trans = trans_list[i][j]
                fx, cx = fxs[i][j], cxs[i][j]
                # determine whether apply side cameras and reverse rots
                if j==1 or j==2:
                    continue
                    rots = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]], device = rots.device, dtype=torch.float32)@rots
                if len(a.shape) != 3:
                    print("a.shape != 3, a.shape is:")
                    print(a.shape)
                    a = a[0]
                C1, h1, w1 = a.shape
                C2, h2, w2 = b.shape
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
                a = a + a_enhanced
            enhanced_a_list.append(a + a_enhanced) # a or a + a_enhanced?, could lead to minor performance differences
        return enhanced_a_list

class SideDepthAwareCrossAttention(nn.Module):
    def __init__(self, C1, C2, num_heads, h1, h2, dropout=0.1):
        super(SideDepthAwareCrossAttention, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.num_heads = num_heads
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.query_proj = nn.Linear(C1, C1)
        self.key_proj = nn.Linear(C2, C1)
        self.value_proj = nn.Linear(C2, C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, h1, C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, h2, C2))
        self.attention = MultiheadAttention(C1, num_heads=num_heads, dropout=dropout, batch_first=True)
        #for side attention
        self.side_query_proj = nn.Linear(C1, C1)
        self.side_key_proj = nn.Linear(C2, C1)
        self.side_value_proj = nn.Linear(C2, C1)
        self.side_pos_encoding_a = nn.Parameter(torch.randn(1, h1, C1))
        self.side_pos_encoding_b = nn.Parameter(torch.randn(1, h2, C2))
        self.side_attention = MultiheadAttention(C1, num_heads=num_heads, dropout=dropout, batch_first=True)
        #Auxiliary Detection Head
        self.detection_head = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size = 3, padding = 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list):
        n = len(list_a)
        cam_num = 4
        print(n)
        print(len(list_b))
        if len(list_b) != n*cam_num or len(fov_list) != n: #"All input lists must have the same length."
            print("not aligned legnth of data!!!")
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            for j in range(cam_num):
                b = list_b[i*cam_num + j]
                fov_radians = fov_list[i][j]
                rots = rots_list[i][j]
                trans = trans_list[i][j]

                # determine whether apply side cameras and reverse rots
                if j==1 or j==2:
                    #continue
                    rots = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]], device = rots.device, dtype=torch.float32)@rots
                if len(a.shape) != 3:
                    print("a.shape != 3, a.shape is:")
                    print(a.shape)
                    a = a[0]
                C1, h1, w1 = a.shape
                C2, h2, w2 = b.shape
                a_rect = extract_canvas(a, fov_radians, h1, w2, rots)
                # Initialize enhanced version of a_rect
                a_enhanced = torch.zeros_like(a_rect)
                a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
                b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
                if j==0 or j==3:
                    a_rect += self.pos_encoding_a
                    b_rect += self.pos_encoding_b
                    q = self.query_proj(a_rect) #w1h1c1
                    k = self.key_proj(b_rect) #w1h2c1
                    v = self.value_proj(b_rect) #w1h2c1
                    attn_output, _ = self.attention(q, k, v) #w1h1c1
                if j==1 or j==2:
                    #auxiliary determination:
                    detection_result = self.detection_head(b).view(1)
                    if torch.sigmoid(detection_result) <= 0.5:
                        continue
                    #if vehicle exist
                    a_rect += self.side_pos_encoding_a
                    b_rect += self.side_pos_encoding_b
                    q = self.side_query_proj(a_rect) #w1h1c1
                    k = self.side_key_proj(b_rect) #w1h2c1
                    v = self.side_value_proj(b_rect) #w1h2c1
                    attn_output, _ = self.side_attention(q, k, v) #w1h1c1
                a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
                a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots)
                a = a + a_enhanced
            enhanced_a_list.append(a+a_enhanced) # a or a + a_enhanced?, could lead to minor performance differences
        return enhanced_a_list

class PyramidDepthAwareCrossAttention(nn.Module):
    def __init__(self, C1, C2, num_heads, h1, h2, pyramid_levels=3, dropout=0.1):
        super(PyramidDepthAwareCrossAttention, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.num_heads = num_heads
        self.pyramid_levels = pyramid_levels
        #self.depth_levels = depth_levels
        #qkv linear trans
        self.qs = nn.ModuleList([nn.Linear(C1, C1) for i in range(pyramid_levels)])
        self.ks = nn.ModuleList([nn.Linear(C2, C1) for i in range(pyramid_levels)])
        self.vs = nn.ModuleList([nn.Linear(C2, C1) for i in range(pyramid_levels)])
        self.pos_encoding_a = nn.Parameter(torch.randn(1, h1, C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, h2, C2))
        self.attentions = nn.ModuleList([MultiheadAttention(C1, num_heads=num_heads, dropout=dropout, batch_first=True) for i in range(pyramid_levels)])
        #up and down sampling stuff
        self.pools = nn.ModuleList([nn.AvgPool2d(2**i) for i in range(pyramid_levels)])
        self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(pyramid_levels)])

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
            a_rect = extract_canvas(a, fov_radians, h1, w2, rots)
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2

            a_rect = a_rect + self.pos_encoding_a
            b_rect = b_rect + self.pos_encoding_b

            for level, (pool, upsample, qi, ki, vi, atteni) in enumerate(zip(self.pools, self.upsamples, self.qs, self.ks, self.vs, self.attentions)):
                a_scale = pool(a_rect.permute(2,1,0)).permute(2,1,0)
                b_scale = pool(b_rect.permute(2,1,0)).permute(2,1,0)

                q = qi(a_scale) #whc
                k = ki(b_scale) #whc
                v = vi(b_scale) #whc
                attn_output, _ = atteni(q, k, v) #whc
                attn_output = attn_output.permute(2,1,0) #chw
                attn_output = upsample(attn_output.unsqueeze(0)).squeeze(0)
                a_enhanced = a_enhanced + attn_output
            a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots)
            enhanced_a_list.append(a+a_enhanced) # a or a + a_enhanced?, could lead to minor performance differences
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
        n = list_a[0].shape[0]
        print("list_b[0] length:"+str(n))
        print("list_a[0] shape:" + str(list_a[0].shape))
        if len(list_b[0]) != n*4: #sum of agents per batch error
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
                for k in range(4): #cam_num = 4
                    b = b_i[j*4+k]
                    if k == 1 or k == 2:
                        continue
                    fov_radians = fov_list[j][k]
                    rots = rots_list[j][k]
                    trans = trans_list[j][k]
                    fx, cx = fxs[j][k], cxs[j][k]

                    C1, h1, w1 = a.shape
                    C2, h2, w2 = b.shape
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
                    a_enhanced = restore_undistorted_canvas(a_enhanced, a.shape, fov_radians, rots, fx, cx)
                    a = a + a_enhanced
                a_i_list.append(a)

            enhanced_a_list.append(torch.stack(a_i_list))
        return enhanced_a_list
