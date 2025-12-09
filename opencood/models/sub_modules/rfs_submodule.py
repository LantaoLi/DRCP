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

class EfficientNetFeatureExtractor(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, output_channels, target_H, target_W):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        self.features = nn.Sequential(self.trunk.features[:5]) #sequential 5 layers will be used
        self.conv = nn.Conv2d(in_channels = 112, out_channels=output_channels, kernel_size=1) #112 due to the layers' setting of efficientnet
        self.upsample = nn.Upsample(size=(target_H, target_W), mode='bilinear', align_corners=False)

    def forward(self, x):
        N, M, C, H, W = x.size()
        x = x.view(-1, C, H, W) #resizing to N*M C H W
        x = self.features(x) #feature extraction
        x = self.conv(x) #channel settings
        x = self.upsample(x) #upsamling to target size
        return x

class ResNetFeatureExtractor(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, output_channels, target_H, target_W):
        super(ResNetFeatureExtractor, self).__init__()
        self.trunk = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        self.features = nn.Sequential(*list(self.trunk.children())[:-3]) #sequential 5 layers will be used
        self.conv = nn.Conv2d(in_channels = 1024, out_channels=output_channels, kernel_size=1) #1024 due to the layers' setting of resnet101
        self.upsample = nn.Upsample(size=(target_H, target_W), mode='bilinear', align_corners=False)

    def forward(self, x):
        N, M, C, H, W = x.size()
        x = x.view(-1, C, H, W) #resizing to N*M C H W
        x = self.features(x) #feature extraction
        x = self.conv(x) #channel settings
        x = self.upsample(x) #upsamling to target size
        return x

class MulLeResNetFeatureExtractor(nn.Module):  # 提取图像特征进行图像编码 at 3-levels scale
    def __init__(self, output_channels, target_H, target_W, target_levels = 2):
        super(MulLeResNetFeatureExtractor, self).__init__()
        # 1st level at target_H, target_W
        self.trunk1 = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        self.features1 = nn.Sequential(*list(self.trunk1.children())[:-3]) #sequential 5 layers will be used
        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels=output_channels, kernel_size=1) #1024 due to the layers' setting of resnet101
        self.upsample1 = nn.Upsample(size=(target_H, target_W), mode='bilinear', align_corners=False)
        # 2nd level
        self.trunk2 = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        self.features2 = nn.Sequential(*list(self.trunk2.children())[:-3]) #sequential 5 layers will be used
        self.conv2 = nn.Conv2d(in_channels = 1024, out_channels=output_channels*2, kernel_size=1) #1024 due to the layers' setting of resnet101
        self.upsample2 = nn.Upsample(size=(int(target_H/2), int(target_W/2)), mode='bilinear', align_corners=False)
        # 3rd level
        self.trunk3 = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        self.features3 = nn.Sequential(*list(self.trunk3.children())[:-3]) #sequential 5 layers will be used
        self.conv3 = nn.Conv2d(in_channels = 1024, out_channels=output_channels*4, kernel_size=1) #1024 due to the layers' setting of resnet101
        self.upsample3 = nn.Upsample(size=(int(target_H/4), int(target_W/4)), mode='bilinear', align_corners=False)

    def forward(self, x):
        N, M, C, H, W = x.size()
        x = x.view(-1, C, H, W) #resizing to N*M C H W

        x1 = self.features1(x) #feature extraction
        x1 = self.conv1(x1) #channel settings
        x1 = self.upsample1(x1) #upsamling to target size

        x2 = self.features2(x) #feature extraction
        x2 = self.conv2(x2) #channel settings
        x2 = self.upsample2(x2) #upsamling to target size

        x3 = self.features3(x) #feature extraction
        x3 = self.conv3(x3) #channel settings
        x3 = self.upsample3(x3) #upsamling to target size
        return [x1, x2, x3]
