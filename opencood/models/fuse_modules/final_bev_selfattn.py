# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistribution

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from opencood.visualization.debug_plot import plot_feature
#from flash_attn.flash_attention import FlashAttention

from PIL import Image
import torchvision.transforms.functional as TF
import copy


class BEVSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4):
        """
        初始化 BEV Self-Attention 模块。
        :param embed_dim: 输入特征的维度（C）
        :param n_heads: 多头数量
        """
        super(BEVSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        #torch.nn.MultiheadAttention no ganna work
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        # 残差连接 + LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 自注意力处理后的特征图，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 将特征图展平为序列
        x_flatten = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        # Self-Attention (query, key, value 都是 x_flatten)
        attn_output, _ = self.attention(x_flatten, x_flatten, x_flatten)  # (B, H*W, C)
        # 残差连接 + LayerNorm
        attn_output = self.layer_norm(attn_output + x_flatten)  # (B, H*W, C)
        # 恢复到原始形状
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        return attn_output
"""
class BEVFlashAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4):
        super(BEVFlashAttention, self).__init__()
        self.flash_attn = FlashAttention()
        self.n_heads = n_heads
        self.head_dim = embed_dim//n_heads
        self.embed_dim = embed_dim
        self.flash_attn.softmax_scale = (self.head_dim)**(-0.5)

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        T = H*W
        q = q.permute(0, 2, 3, 1).reshape(B, T, C)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = q
        v = q
        attn_output = self.flash_attn(q, k, v, attn_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        attn_output = attn_output.view(B, H, W, C).permute(0, 3, 1, 2)
        return attn_output
"""


class BEVConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size = 3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class BEVMultiConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(BEVMultiConv, self).__init__()
        # 多尺度卷积核
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        # 融合层
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 多尺度卷积
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        # 拼接后融合
        out = torch.cat([x3, x5, x7], dim=1)  # 在通道维度拼接
        out = self.relu(self.fusion(out))
        return out

class BEVMultiDepthwiseConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(BEVMultiDepthwiseConv, self).__init__()
        # 深度可分离卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.relu(self.fusion(out))
        return out

class BEVDynamicMultiConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(BEVDynamicMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        # 动态权重生成器
        self.weight_gen = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),  # 全局池化
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 动态生成权重
        w = self.weight_gen(x)# 多尺度卷积
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        # 融合特征
        out = w[:,0:1,:,:]*x3 + w[:,1:2,:,:]*x5 + w[:,2:3,:,:]*x7
        out = self.relu(out)
        return out

class MultiPerChaConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(MultiPerChaConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=256)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=256)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=256)
        # 动态权重生成器
        self.weight_gen = nn.Conv2d(in_channels, in_channels*3, kernel_size=1, groups=in_channels)  # 全局池化
        self.relu = nn.ReLU()

    def forward(self, x):
        # 动态生成权重
        B, C, H, W = x.shape
        w = self.weight_gen(x)# 多尺度卷积
        w = w.view(B,C,3,H,W)
        w = F.softmax(w, dim=2)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        # 融合特征
        out = w[:,:,0,:,:]*x3 + w[:,:,1,:,:]*x5 + w[:,:,2,:,:]*x7
        out = self.relu(out)
        return out

class BEVInceptionStyleConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(BEVInceptionStyleConv, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x_pool = self.pool(x)
        out = torch.cat([x1, x3, x5, x_pool], dim=1)
        out = self.relu(self.fusion(out))
        return out
