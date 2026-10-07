"""
EURS-Net V2 Common Layers
Includes:
- FastWhiteBalance: Learnable 3-channel affine correction initialized as identity.
- CoordinateAttention: 1D directional pooling (H, W) for lightweight positional attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWhiteBalance(nn.Module):
    """
    Learnable 3-channel affine scale and bias layer:
        out_c = alpha_c * in_c + beta_c
    Initialized as an identity mapping before training (alpha=1, beta=0).
    Uses only 6 trainable parameters to learn per-channel color correction
    before deep feature extraction layers.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 3, 1, 1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, 3, 1, 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Feature Encoders.
    Produces:
        Horizontal pooling -> [B, C, H, 1]
        Vertical pooling   -> [B, C, 1, W]
    Uses standard, ONNX/TensorRT-friendly operations to encode precise
    spatial coordinate information with negligible FLOP overhead.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mip = max(8, in_channels // reduction)

        # 1x1 conv shared across concatenated coordinate vectors
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # Separate 1x1 convs to recover channel dimension
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # Direction-aware pooling: produces [B, C, H, 1] and [B, C, 1, W]
        x_h = torch.mean(x, dim=3, keepdim=True)  # [B, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True)  # [B, C, 1, W]

        # Concatenate along spatial dimension after transposing x_w
        x_w_permuted = x_w.permute(0, 1, 3, 2)    # [B, C, W, 1]
        y = torch.cat([x_h, x_w_permuted], dim=2) # [B, C, H + W, 1]

        y = self.act(self.bn1(self.conv1(y)))

        # Split back to H and W components
        x_h_feat = y[:, :, :h, :]
        x_w_feat = y[:, :, h:, :].permute(0, 1, 3, 2) # [B, C, 1, W]

        a_h = torch.sigmoid(self.conv_h(x_h_feat))
        a_w = torch.sigmoid(self.conv_w(x_w_feat))

        return identity * a_h * a_w
