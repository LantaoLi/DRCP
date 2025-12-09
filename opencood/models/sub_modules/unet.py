import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """Double Conv Block"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, base_channels=128):
        super(LightweightUNet, self).__init__()

        # Encoder (downsample once)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)

        # Bottleneck with Attention
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.se_bottleneck = SEBlock(base_channels * 4)

        # Decoder (upsample once)
        self.up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec = ConvBlock(base_channels * 3, base_channels)  # Concat: (base*2 from up) + (base from enc1) = base*3
        self.se_dec = SEBlock(base_channels)

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() if out_channels == 1 else None

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [batch, base, H, W]
        p = self.pool(e1)  # [batch, base, H/2, W/2]
        e2 = self.enc2(p)  # [batch, base*2, H/2, W/2]

        # Bottleneck
        b = self.bottleneck(e2)  # [batch, base*4, H/2, W/2]
        b = self.se_bottleneck(b)

        # Decoder
        u = self.up(b)  # [batch, base*2, H, W]
        cat = torch.cat([u, e1], dim=1)  # [batch, base*3, H, W]
        d = self.dec(cat)  # [batch, base, H, W]
        d = self.se_dec(d)

        # Output
        out = self.out_conv(d)  # [batch, out_channels, H, W]
        if self.sigmoid:
            out = self.sigmoid(out)
        return out
