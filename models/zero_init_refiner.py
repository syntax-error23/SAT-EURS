"""
Zero-Initialized RAM Residual Refiner CNN
Stage 1 of Track 1 V17 Apex Breakthrough Pipeline
MaCVi @ WACV 2027 Challenge

Architecture:
- Input (7 Channels):
  * Degraded RGB (3ch, [0, 1])
  * Physical RSUIGM Prior (3ch, [0, 1])
  * Depth Map (1ch, depth / 100.0)
- Dilated Residual Convolutional Backbone:
  * 4 Dilated Conv2d Layers (width=64, receptive field ~29px)
  * GELU activations
- Zero-Initialization:
  * Final Conv2d weights and bias initialized strictly to 0.0.
  * Guarantees at Step 0 that J_final = J_rsuigm (PSNR = 25.41 dB exact start).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitResidualRefiner(nn.Module):
    def __init__(self, in_channels: int = 7, width: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=2, dilation=2)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=4, dilation=4)
        self.act3 = nn.GELU()
        
        self.conv4 = nn.Conv2d(width, width, kernel_size=3, padding=1, dilation=1)
        self.act4 = nn.GELU()
        
        self.final_conv = nn.Conv2d(width, 3, kernel_size=3, padding=1)
        
        # Mathematical guarantee: Zero-init final layer
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, d_t: torch.Tensor, j_rsuigm: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_t: [B, 3, H, W] degraded image in [0, 1]
            j_rsuigm: [B, 3, H, W] physical RSUIGM prior in [0, 1]
            depth_map: [B, 1, H, W] normalized depth map (depth / 100.0)
        Returns:
            j_final: [B, 3, H, W] refined restored image in [0, 1]
        """
        x = torch.cat([d_t, j_rsuigm, depth_map], dim=1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        delta = self.final_conv(x)
        return torch.clamp(j_rsuigm + delta, 0.0, 1.0)


def build_zero_init_refiner(width: int = 64) -> ZeroInitResidualRefiner:
    return ZeroInitResidualRefiner(in_channels=7, width=width)
