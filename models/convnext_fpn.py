"""
ConvNeXt-Base FPN: Complementary Hierarchical Stride-4 Architecture
Stage 4 of SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Architecture:
1. Backbone: tu-convnext_base (ImageNet-22k pretrained, fine-tuned on ImageNet-1k).
   - Natively extracts multi-scale pyramid features at Stride 4, Stride 8, Stride 16, and Stride 32.
   - Stem and Stage 0 frozen to prevent 20-scene low-level texture memorization.
2. Decoder: Feature Pyramid Network (FPN) fusing multi-scale pyramidal features.
3. Completely breaks the ViT patch-14 monoculture, providing orthogonal spatial representations.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_convnext_fpn(num_classes: int = 5, freeze_stage0: bool = True) -> nn.Module:
    model = smp.FPN(
        encoder_name="tu-convnext_base",
        encoder_weights="imagenet",
        classes=num_classes,
        in_channels=3
    )

    if freeze_stage0:
        # Freeze stem and stage 0
        for name, p in model.encoder.model.named_parameters():
            if name.startswith("stem_") or name.startswith("stages_0."):
                p.requires_grad = False
            else:
                p.requires_grad = True

    return model

if __name__ == '__main__':
    m = build_convnext_fpn(num_classes=5, freeze_stage0=True)
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    total_p = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"ConvNeXt-FPN created! Total params: {total_p/1e6:.2f}M | Trainable: {train_p/1e6:.2f}M | Output: {y.shape}")
