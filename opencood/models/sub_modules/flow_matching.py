import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DModel

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class UNetBEVFlow(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        #encoders
        self.enc1 = DoubleConv(in_channels, 128)
        self.enc2 = DoubleConv(128, 256)
        self.enc3 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        #decoders (up + conv)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)

        if self.use_attention:
            self.att3 = AttentionGate(F_g=512, F_l=512, F_int=256)
            self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
            self.att1 = AttentionGate(F_g=128, F_l=128, F_int=64)

        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        if self.use_attention:
            e3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if self.use_attention:
            e2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if self.use_attention:
            e1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out

class PlainFlowMatchingBEVNet(nn.Module):
    def __init__(self,
                 in_channels=256,   # 没有 mask
                 base_channels=128,
                 out_channels=2     # dx, dy
                 ):
        super().__init__()

        # 使用 diffusers UNet2DModel 做 Flow Field Estimator
        """
        self.flow_net = UNet2DModel(
            sample_size=None,  # 动态 shape
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        """
        self.flow_net = UNetBEVFlow()

    def forward(self, bev_feature):
        """
        bev_feature: [B, 256, H, W]
        """
        # Predict Flow Field
        #flow = self.flow_net(bev_feature, timestep=0).sample  # [B, 2, H, W]
        flow = self.flow_net(bev_feature)

        # Base grid: [-1, 1] normalized
        B, _, H, W = flow.shape
        base_grid = self.make_base_grid(B, H, W, bev_feature.device)  # [B, H, W, 2]

        # Normalize flow to [-1, 1] range
        flow_norm = flow.permute(0, 2, 3, 1) / torch.tensor([W/2, H/2], device=bev_feature.device)

        sampling_grid = base_grid + flow_norm

        # Warp the BEV feature
        warped_bev = F.grid_sample(
            bev_feature, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return warped_bev, flow, None

    def make_base_grid(self, B, H, W, device):
        """
        返回 [-1, 1] normalized grid
        """
        theta = torch.linspace(-1, 1, W, device=device)
        phi = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(phi, theta, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, H, W, 2]
        return grid

class BEVFMMasker(nn.Module):
    def __init__(self, in_channels=256):
        super(BEVFMMasker, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feature):
        mask = self.sigmoid(self.conv1(bev_feature))
        return mask  # [B, 1, H, W]

class MaskFlowMatchingBEVNet(nn.Module):
    def __init__(self,
                 in_channels=256,  # 不需要改
                 base_channels=128,
                 out_channels=2):
        super(MaskFlowMatchingBEVNet, self).__init__()

        # Flow Field Estimator: UNet2D
        self.flow_net = UNet2DModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

        # Mask branch
        self.masker = BEVFMMasker(in_channels=in_channels)

    def forward(self, bev_feature):
        """
        bev_feature: [B, 256, H, W]
        """
        # 1. Predict flow
        flow = self.flow_net(bev_feature, timestep=0).sample  # [B, 2, H, W]

        # 2. Predict mask
        mask = self.masker(bev_feature)  # [B, 1, H, W]

        # 3. Make base grid
        B, _, H, W = flow.shape
        base_grid = self.make_base_grid(B, H, W, bev_feature.device)

        flow_norm = flow.permute(0, 2, 3, 1) / torch.tensor([W/2, H/2], device=bev_feature.device)
        sampling_grid = base_grid + flow_norm

        # 4. Warp
        warped_bev = F.grid_sample(
            bev_feature, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # 5. Use mask: Known region keeps original, unknown region uses warp
        pseudo_bev = warped_bev * (1 - mask) + bev_feature * mask

        return pseudo_bev, flow, mask

    def make_base_grid(self, B, H, W, device):
        theta = torch.linspace(-1, 1, W, device=device)
        phi = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(phi, theta, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, H, W, 2]
        return grid
