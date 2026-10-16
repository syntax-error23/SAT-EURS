"""
Canonical NAFNet: Nonlinear Activation Free Network for Image Restoration
Reference: Chen et al., 'Simple Baselines for Image Restoration', ECCV 2022.
Clean, modular PyTorch implementation for SAT-EURS Track 1 Restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True)
        )

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    Canonical NAFNet architecture for Image Restoration.
    Default configuration:
    - width = 32 (Practical ~7.6M params) or 48 (~17.1M params)
    - enc_blk_nums = [2, 2, 4, 8]
    - middle_blk_num = 4
    - dec_blk_nums = [2, 2, 2, 2]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        enc_blk_nums: list = [2, 2, 4, 8],
        middle_blk_num: int = 4,
        dec_blk_nums: list = [2, 2, 2, 2],
        use_residual_connection: bool = True
    ):
        super().__init__()
        self.use_residual_connection = use_residual_connection
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan // 2, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)
        if self.use_residual_connection:
            nn.init.zeros_(self.ending.weight)
            nn.init.zeros_(self.ending.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.intro(x)

        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_skips.append(feat)
            feat = down(feat)

        feat = self.middle_blks(feat)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(enc_skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        res = self.ending(feat)
        if self.use_residual_connection:
            out = torch.clamp(x + res, 0.0, 1.0)
        else:
            out = torch.clamp(res, 0.0, 1.0)
        return out


def build_nafnet_restorer(width: int = 32, in_channels: int = 3) -> NAFNet:
    return NAFNet(
        in_channels=in_channels,
        out_channels=3,
        width=width,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=4,
        dec_blk_nums=[2, 2, 2, 2],
        use_residual_connection=True
    )
