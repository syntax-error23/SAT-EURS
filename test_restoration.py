import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6, use_ca=False):
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish()
        ])
        if use_ca:
            layers.append(CoordinateAttention(hidden_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class ProgressiveRestorationHead(nn.Module):
    """
    Progressive Multi-Scale Guided Restoration Decoder:
    Starts at 128x128 (F128) -> Fuses F256 (256x256) -> Fuses Stem (512x512)
    Predicts high-frequency & color residual Delta I at 512x512, then gently upsamples to 1024x1024.
    """
    def __init__(self):
        super().__init__()
        # Level 1: 128x128
        self.proj_f128 = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Level 2: 256x256
        self.proj_f256 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.dw_conv256 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        # Level 3: 512x512
        self.proj_stem = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24)
        )
        self.dw_conv512 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Final residual prediction head
        self.head = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, f128, f256, stem, raw_input):
        # Step 1: 128x128 -> 256x256
        x = self.proj_f128(f128)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_f256(f256)
        x = self.dw_conv256(x)

        # Step 2: 256x256 -> 512x512
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_stem(stem)
        x = self.dw_conv512(x)

        # Step 3: Residual map at 512x512 -> 1024x1024
        delta_512 = self.head(x)
        delta_1024 = F.interpolate(delta_512, size=raw_input.shape[2:], mode='bilinear', align_corners=False)

        # Add residual to original degraded input
        restored = torch.clamp(raw_input + delta_1024, 0.0, 1.0)
        return restored

def count_flops_and_params():
    # Test tensor shapes
    b = 1
    raw_input = torch.randn(b, 3, 1024, 1024)
    stem = torch.randn(b, 16, 512, 512)
    f256 = torch.randn(b, 24, 256, 256)
    f128 = torch.randn(b, 40, 128, 128)
    model = ProgressiveRestorationHead()
    model.train()
    out = model(f128, f256, stem, raw_input)
    loss = out.mean()
    loss.backward()
    print("Backward pass & gradient flow successful! All gradients computed.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Restoration Branch Output Shape: {out.shape}")
    print(f"Restoration Branch Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")

if __name__ == '__main__':
    count_flops_and_params()
