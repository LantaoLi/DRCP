import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
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

class TimeEmbedding(nn.Module):
    """Diffusion-style sinusoidal timestep embedding + MLP"""
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        # t: [B] or [B, 1]
        if t.dim() == 1:
            t = t[:, None]
        half_dim = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, base_channels=256, time_emb_dim=256):
        super().__init__()

        self.time_mlp = TimeEmbedding(time_emb_dim, hidden_dim=time_emb_dim)

        # Encoder
        self.enc1 = ConvBlock(in_channels + in_channels, base_channels)  # concat with cond tensor
        self.film1 = FiLM(base_channels, time_emb_dim)

        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.film2 = FiLM(base_channels * 2, time_emb_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.se_bottleneck = SEBlock(base_channels * 4)
        self.film_b = FiLM(base_channels * 4, time_emb_dim)

        # Decoder
        self.up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec = ConvBlock(base_channels * 3, base_channels)
        self.se_dec = SEBlock(base_channels)
        self.film_d = FiLM(base_channels, time_emb_dim)

        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() if out_channels == 1 else None

    def forward(self, x, t, cond):
        """
        x: [B, C, H, W]  - noisy input
        t: [B] or [B, 1] - timesteps
        cond: [B, C, H, W] - same shape condition tensor
        """
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # Encoder
        e1 = self.enc1(torch.cat([x, cond], dim=1))
        e1 = self.film1(e1, t_emb)
        p = self.pool(e1)

        e2 = self.enc2(p)
        e2 = self.film2(e2, t_emb)

        # Bottleneck
        b = self.bottleneck(e2)
        b = self.se_bottleneck(b)
        b = self.film_b(b, t_emb)

        # Decoder
        u = self.up(b)
        cat = torch.cat([u, e1], dim=1)
        d = self.dec(cat)
        d = self.se_dec(d)
        d = self.film_d(d, t_emb)

        out = self.out_conv(d)
        if self.sigmoid:
            out = self.sigmoid(out)
        return out
