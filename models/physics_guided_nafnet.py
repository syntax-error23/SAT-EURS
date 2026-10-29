"""
PhysicsGuidedNAFNet: 8-Channel Single-Image Residual Underwater Neural Refiner
Stage 2 of SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Architecture:
1. Inputs (8 channels):
   - Degraded RGB image I_deg in [0, 1] (3 channels)
   - Fold-leak-free RSUIGM physical inversion prior J_rsuigm in [0, 1] (3 channels)
   - Normalized water-type map wt / 10.0 (1 channel)
   - Normalized depth map depth / 100.0 (1 channel)
2. NAFNet Hierarchical Backbone (SimpleGate + Simplified Channel Attention, ~21M params).
3. Bounded Residual Refinement:
   J_hat = clamp(J_rsuigm + 0.25 * tanh(net(x)), 0.0, 1.0)
   Guarantees physical plausibility and stable training anchored on the RSUIGM solution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True)
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.norm1(x)
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

class PhysicsGuidedNAFNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 48,
        enc_blk_nums: list = [2, 2, 4, 6],
        dec_blk_nums: list = [4, 2, 2, 2],
        residual_scale: float = 0.25
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(4)])

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
        # Initialize ending layer close to zero so model starts smoothly from the RSUIGM prior
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(
        self,
        i_deg: torch.Tensor,
        j_rsuigm: torch.Tensor,
        wt_norm: torch.Tensor,
        depth_norm: torch.Tensor,
        return_residual: bool = False
    ) -> torch.Tensor:
        """
        i_deg:      [B, 3, H, W] in [0, 1]
        j_rsuigm:   [B, 3, H, W] in [0, 1]
        wt_norm:    [B, 1, H, W] in [0.1, 1.0]
        depth_norm: [B, 1, H, W] in [0.05, 1.0]
        """
        x_in = torch.cat([i_deg, j_rsuigm, wt_norm, depth_norm], dim=1)  # [B, 8, H, W]

        feat = self.intro(x_in)
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

        residual = torch.tanh(self.ending(feat)) * self.residual_scale
        out = torch.clamp(j_rsuigm + residual, 0.0, 1.0)
        if return_residual:
            return out, residual
        return out

if __name__ == "__main__":
    net = PhysicsGuidedNAFNet().cuda()
    b, h, w = 2, 256, 256
    i_deg = torch.randn(b, 3, h, w, device="cuda").clamp(0, 1)
    j_rsuigm = torch.randn(b, 3, h, w, device="cuda").clamp(0, 1)
    wt_norm = torch.ones(b, 1, h, w, device="cuda") * 0.5
    dp_norm = torch.ones(b, 1, h, w, device="cuda") * 0.3

    out = net(i_deg, j_rsuigm, wt_norm, dp_norm)
    params = sum(p.numel() for p in net.parameters())
    print(f"PhysicsGuidedNAFNet created successfully! Parameters: {params / 1e6:.2f}M | Output shape: {out.shape}")
    # Verify zero-residual start
    diff = (out - j_rsuigm).abs().max().item()
    print(f"Initial deviation from RSUIGM prior: {diff:.8f} (Zero-Anchor Guarantee: {'PASS' if diff < 1e-6 else 'FAIL'})")
