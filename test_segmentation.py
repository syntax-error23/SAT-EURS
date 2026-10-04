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

class ProgressiveSegmentationDecoder(nn.Module):
    """
    Upgraded Progressive Segmentation Decoder:
    Fuses F32 -> F64 -> F128 -> F256 (Small-Object Neck with CoordAttn) -> Stem (512x512 Boundary Refinement)
    Outputs high-precision 5-class logits at 512x512, then gently upsamples 2x to 1024x1024.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        # FPN Top-Down Stages
        self.lat_f32 = nn.Conv2d(96, 64, kernel_size=1)
        self.lat_f64 = nn.Conv2d(64, 64, kernel_size=1)
        self.lat_f128 = nn.Conv2d(40, 48, kernel_size=1)
        self.lat_f256 = nn.Conv2d(24, 32, kernel_size=1)

        self.smooth_64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish()
        )
        self.smooth_128 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish()
        )
        
        # Small-Object Neck at 256x256
        self.small_obj_neck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            CoordinateAttention(32)
        )

        # Boundary Refinement Level at 512x512
        self.lat_stem = nn.Conv2d(16, 24, kernel_size=1)
        self.boundary_refine_512 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish()
        )
        self.dw_boundary_512 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
            CoordinateAttention(24),
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        # Segmentation Head predicting logits at 512x512
        self.head_512 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, f32, f64, f128, f256, stem):
        # 1. 32x32 -> 64x64
        p32 = self.lat_f32(f32)
        p64 = F.interpolate(p32, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f64(f64)
        p64 = self.smooth_64(p64)

        # 2. 64x64 -> 128x128
        p128 = F.interpolate(p64, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f128(f128)
        p128 = self.smooth_128(p128)

        # 3. 128x128 -> 256x256
        p256 = F.interpolate(p128, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f256(f256)
        p256 = self.small_obj_neck(p256)

        # 4. 256x256 -> 512x512 (Boundary Refinement with Stem skip)
        p512 = F.interpolate(p256, scale_factor=2.0, mode='bilinear', align_corners=False)
        p512 = self.boundary_refine_512(p512) + self.lat_stem(stem)
        p512 = self.dw_boundary_512(p512)

        # 5. Predict at 512x512, then gently upsample 2x to full 1024x1024
        logits_512 = self.head_512(p512)
        logits_1024 = F.interpolate(logits_512, scale_factor=2.0, mode='bilinear', align_corners=False)
        return logits_1024

def test_segmentation():
    b = 1
    stem = torch.randn(b, 16, 512, 512)
    f256 = torch.randn(b, 24, 256, 256)
    f128 = torch.randn(b, 40, 128, 128)
    f64  = torch.randn(b, 64, 64, 64)
    f32  = torch.randn(b, 96, 32, 32)

    model = ProgressiveSegmentationDecoder(num_classes=5)
    model.train()
    out = model(f32, f64, f128, f256, stem)

    loss = out.mean()
    loss.backward()
    print("Segmentation Backward pass & gradient flow successful!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Segmentation Output Shape: {out.shape}")
    print(f"Segmentation Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")

if __name__ == '__main__':
    test_segmentation()
