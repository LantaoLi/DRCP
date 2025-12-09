# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet101
import torch.nn.functional as F
from opencood.utils.camera_utils import bin_depths
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import \
    MaxFusion, AttFusion, V2VNetFusion, V2XViTFusion, DiscoFusion

class BEVDiffuseMasker(nn.Module):  # 提取图像特征进行mask generation
    def __init__(self, in_channels=256, groups=1):
        super(BEVDiffuseMasker, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feature):
        x = F.relu(self.conv1(bev_feature))
        mask = self.sigmoid(self.conv2(x))
        return mask

class BEVDiffusePerMasker(nn.Module):  # per channel weights from original bev
    def __init__(self, in_channels=256, groups=1):
        super(BEVDiffusePerMasker, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feature):
        mask = self.sigmoid(self.conv1(bev_feature))
        return mask

class BEVDiffusePer2ChMasker(nn.Module):  # per channel weights from original and diffused bev both
    def __init__(self, in_channels=256):
        super(BEVDiffusePer2ChMasker, self).__init__()
        """
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        """
        self.conv_original = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_diffusion = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, bev_feature, diffuse_feature):
        """
        fused_input = torch.cat([bev_feature, diffuse_feature], dim=1)
        mask = self.sigmoid(self.conv1(fused_input))
        """
        weight_original = self.sigmoid(self.conv_original(bev_feature))
        weight_diffusion = self.sigmoid(self.conv_diffusion(diffuse_feature))
        mask = weight_original/(weight_original + weight_diffusion) # + 1e-8 might be necessary

        return mask

class BEVDiffusePerMaskFuser(nn.Module):  # per channel weights from original and diffused bev both
    def __init__(self, in_channels=256):
        super(BEVDiffusePerMaskFuser, self).__init__()
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, bev_feature, diffuse_feature):
        fused_input = torch.cat([bev_feature, diffuse_feature], dim=1)
        fused_output = self.conv1(fused_input)
        return fused_output

class BEVDiffuseAttnFuser(nn.Module):  # per channel weights from original and diffused bev both
    def __init__(self, in_channels=256, num_heads=8):
        super(BEVDiffuseAttnFuser, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim = in_channels, num_heads = num_heads, batch_first=True)
        #self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, bev_feature, diffuse_feature):
        B, C, H, W = bev_feature.shape
        bev_flat = bev_feature.view(B, C, -1).permute(0, 2, 1) #(B,C,H,W) -> (B, HW, C)
        diffuse_flat = diffuse_feature.view(B, C, -1).permute(0, 2, 1)

        attn_feature, _ = self.mha(query = bev_flat, key = diffuse_flat, value = diffuse_flat)
        attn_feature = attn_feature.permute(0, 2, 1).view(B, C, H, W)

        #attn_feature = self.value_proj(attn_feature)
        fused_output = attn_feature + bev_feature
        return fused_output

class LidarDiffuseMasker(nn.Module):  # 提取图像特征进行mask generation
    def __init__(self, in_channels=64):
        super(LidarDiffuseMasker, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feature):
        x = F.relu(self.conv1(bev_feature))
        mask = self.sigmoid(self.conv2(x))
        return mask

class CameraDiffuseMasker(nn.Module):  # 提取图像特征进行mask generation
    def __init__(self, in_channels=8):
        super(CameraDiffuseMasker, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feature):
        x = F.relu(self.conv1(bev_feature))
        mask = self.sigmoid(self.conv2(x))
        return mask
