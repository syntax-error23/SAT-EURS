import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True)
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

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

class EURSUnderwaterRestorer(nn.Module):
    """
    High-Capacity Dedicated Physical Underwater Restorer (Track 1).
    Unconstrained parameter size (~21M parameters).
    Combines:
    1. Physical Optical Estimation (Veiling light B_inf and Transmission T)
    2. Multi-Scale NAFNet Hierarchical Residual Architecture
    3. Global Dynamic Color-Channel Rebalancing
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, width: int = 48, enc_blk_nums=[2, 2, 4, 6], dec_blk_nums=[4, 2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        # Physics Guidance Subnet (Estimates Veiling Light and Attenuation priors)
        self.phys_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 32, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 6, 1), # 3 for B_inf, 3 for global attenuation weight
            nn.Sigmoid()
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(6)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan // 2, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Physical Guidance Extraction
        phys = self.phys_guidance(x)
        b_inf, atten = phys[:, :3], phys[:, 3:]
        # Normalize/de-haze input feature representation
        x_norm = (x - b_inf) * (1.0 + atten) + b_inf

        feat = self.intro(x_norm)

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

        residual = self.ending(feat)
        out = torch.clamp(x + residual, 0.0, 1.0)
        return out

if __name__ == "__main__":
    net = EURSUnderwaterRestorer().cuda()
    x = torch.randn(2, 3, 512, 512, device='cuda')
    y = net(x)
    total_p = sum(p.numel() for p in net.parameters())
    print(f"EURS Dedicated Restorer created successfully! Params: {total_p / 1e6:.2f}M | Output shape: {y.shape}")
