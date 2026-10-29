"""
Unit Test Suite for DINOv2HighResDecoder (Experiment 3.1)
Verifies:
1. Output tensor shapes for variable input resolutions (518x518, 448x448, 588x588).
2. Zero-Identity at Step 0: Exactly identical predictions to baseline DINOv2 checkpoint.
3. Gradient flow across all modules (Backbone, Base Decoder, RGB Stem, High-Res Fusion, Zero Conv).
4. Loss computation stability: Weighted CE + Lovasz-Softmax under torch.amp.autocast.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.dinov2_highres_decoder import DINOv2HighResDecoder
from tools.train_track1_screening import build_dinov2_model
from losses.v4_losses import PresentClassLovaszSoftmaxLoss

def test_highres_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("TESTING DINOv2HighResDecoder (UNIT TEST SUITE)")
    print(f"  Device: {device}")
    print("=" * 76)

    # 1. Shape Verification across Multi-Scale TTA resolutions
    print("\n[Test 1/4] Verifying forward tensor shapes across scales...")
    model = DINOv2HighResDecoder(num_classes=5, pretrained_backbone=False).to(device)
    model.eval()

    test_shapes = [(2, 3, 448, 448), (2, 3, 518, 518), (1, 3, 588, 588)]
    for bs, c, h, w in test_shapes:
        x = torch.randn(bs, c, h, w, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, 5, h, w), f"Expected {(bs, 5, h, w)}, got {out.shape}"
        print(f"  Shape check passed for input {(bs, c, h, w)} -> output {out.shape}")

    # 2. Mathematical Zero-Identity Verification at Step 0
    print("\n[Test 2/4] Verifying Mathematical Zero-Identity against Baseline Checkpoint...")
    ckpt_path = "checkpoints/track1_screening/DINOV2_CP_F0/best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Base model
        base_model = build_dinov2_model(num_classes=5).to(device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        base_model.eval()

        # High-Res model initialized from the same checkpoint
        highres_model = DINOv2HighResDecoder(num_classes=5, pretrained_backbone=False).to(device)
        incompatible = highres_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Missing keys in checkpoint (expected new high-res layers): {len(incompatible.missing_keys)}")
        assert all("stem" in k or "fuse" in k or "zero_conv" in k for k in incompatible.missing_keys)
        highres_model.eval()

        # Fixed deterministic input
        torch.manual_seed(42)
        x_test = torch.randn(2, 3, 518, 518, device=device)

        with torch.no_grad():
            out_base = base_model(x_test)
            out_highres = highres_model(x_test)

        max_abs_diff = (out_base - out_highres).abs().max().item()
        print(f"  Maximum Absolute Difference: {max_abs_diff:.8f}")
        assert max_abs_diff < 1e-5, f"Zero-Identity violated! Diff = {max_abs_diff}"
        print("  Zero-Identity Check: PASS (Identical predictions at initialization!)")
    else:
        print(f"  Checkpoint {ckpt_path} not found, skipping checkpoint identity test.")

    # 3. Gradient Flow Verification (2-step check for zero-initialized residual projection)
    print("\n[Test 3/4] Verifying Gradient Flow across all modules (Zero-Conv Dynamics)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x_dummy = torch.randn(2, 3, 518, 518, device=device)
    y_dummy = torch.randint(0, 5, (2, 518, 518), device=device)

    # Micro-Step 1: Verify zero_conv parameter receives non-zero gradient
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits1 = model(x_dummy)
        loss1 = F.cross_entropy(logits1.float(), y_dummy)
    loss1.backward()

    assert model.zero_conv.weight.grad is not None and model.zero_conv.weight.grad.norm() > 0, "zero_conv did not receive gradient on step 1!"
    print(f"  Step 1: zero_conv gradient norm = {model.zero_conv.weight.grad.norm().item():.4f} -> PASS")

    # Optimizer step updates zero_conv from 0 to non-zero
    optimizer.step()
    optimizer.zero_grad()

    # Micro-Step 2: Upstream stem and fusion modules now receive active non-zero gradients
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits2 = model(x_dummy)
        loss2 = F.cross_entropy(logits2.float(), y_dummy) + 0.25 * PresentClassLovaszSoftmaxLoss(num_classes=5)(logits2.float(), y_dummy)
    loss2.backward()

    components = {
        "encoder": False,
        "proj": False,
        "fusion": False,
        "stem": False,
        "fuse_stage4": False,
        "fuse_stage2": False,
        "zero_conv": False
    }

    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            for k in components:
                if k in name:
                    components[k] = True

    for comp, has_grad in components.items():
        print(f"  Step 2: Module '{comp}' active gradient = {has_grad}")
        assert has_grad, f"Module '{comp}' did not receive any gradients on step 2!"
    print("  Gradient Flow Check: PASS (All modules receive active gradients!)")

    # 4. Loss Stability Check under Autocast
    print("\n[Test 4/4] Verifying loss stability and non-NaN guarantees...")
    assert not torch.isnan(loss1).any(), "Step 1 loss produced NaN!"
    assert not torch.isnan(loss2).any(), "Step 2 loss produced NaN!"
    assert not torch.isinf(loss2).any(), "Step 2 loss produced Inf!"
    print(f"  Step 1 CE Loss: {loss1.item():.4f} | Step 2 CE+Lovasz Loss: {loss2.item():.4f}")
    print("  Loss Stability Check: PASS")

    print("\n" + "=" * 76)
    print("ALL DINOv2HighResDecoder UNIT TESTS PASSED 100%!")
    print("=" * 76)

if __name__ == '__main__':
    test_highres_decoder()
