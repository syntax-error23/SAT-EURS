"""
Unit Test Suite for DINOv2HighResDPT & Frozen Blocks
Stage 3 of SAT-EURS Track 1 V7 Plan
"""

import sys
sys.path.insert(0, ".")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dinov2_highres_dpt import DINOv2HighResDPT
from losses.segmentation_v7_loss import SegmentationV7Loss

def test_dpt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("TESTING DINOv2HighResDPT (STAGE 3 ARCHITECTURE)")
    print(f"  Device: {device}")
    print("=" * 76)

    model = DINOv2HighResDPT(num_classes=5, freeze_blocks_up_to=15, pretrained=False).to(device)

    # 1. Shape Verification across resolutions
    print("\n[Test 1/4] Verifying forward shapes at 700x700 and native aspect 588x784...")
    test_shapes = [(1, 3, 518, 518), (1, 3, 700, 700), (1, 3, 588, 784)]
    model.eval()
    for bs, c, h, w in test_shapes:
        x = torch.randn(bs, c, h, w, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, 5, h, w), f"Expected {(bs, 5, h, w)}, got {out.shape}"
        print(f"  Shape check passed: {(bs, c, h, w)} -> {out.shape}")

    # 2. Frozen Blocks Verification
    print("\n[Test 2/4] Verifying Early Blocks 0..15 are strictly frozen...")
    for i, blk in enumerate(model.encoder.blocks):
        grads = [p.requires_grad for p in blk.parameters()]
        if i <= 15:
            assert not any(grads), f"Block {i} should be frozen, but has trainable parameters!"
        else:
            assert all(grads), f"Block {i} should be trainable, but is frozen!"
    print("  Blocks 0..15: FROZEN (requires_grad = False)")
    print("  Blocks 16..23: TRAINABLE (requires_grad = True)")
    print("  Early Blocks Freeze Check: PASS")

    # 3. Zero-Identity against Baseline Checkpoint
    print("\n[Test 3/4] Verifying Zero-Identity against Baseline Checkpoint...")
    ckpt_path = "checkpoints/track1_screening/DINOV2_CP_F0/best_model.pt"
    if os.path.exists(ckpt_path):
        from tools.train_track1_screening import build_dinov2_model
        base_model = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        base_model.eval()

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        torch.manual_seed(42)
        x_test = torch.randn(1, 3, 518, 518, device=device)
        with torch.no_grad():
            out_base = base_model(x_test)
            out_dpt = model(x_test)

        diff = (out_base - out_dpt).abs().max().item()
        print(f"  Max absolute deviation at Step 0: {diff:.8f}")
        assert diff < 1e-5, f"Zero-identity violated! diff={diff}"
        print("  Zero-Identity Check: PASS (Identical predictions at initialization!)")

    # 4. 2-Step Gradient Flow with SegmentationV7Loss
    print("\n[Test 4/4] Verifying 2-step gradient flow across DPT detail stems...")
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    criterion = SegmentationV7Loss().to(device)

    x_dummy = torch.randn(2, 3, 518, 518, device=device)
    y_dummy = torch.randint(0, 5, (2, 518, 518), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out1 = model(x_dummy)
        loss1, _ = criterion(out1, y_dummy)
    loss1.backward()

    assert model.detail_head.weight.grad is not None and model.detail_head.weight.grad.norm() > 0
    print(f"  Step 1 detail_head gradient norm: {model.detail_head.weight.grad.norm():.4f} -> PASS")

    optimizer.step()
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out2 = model(x_dummy)
        loss2, _ = criterion(out2, y_dummy)
    loss2.backward()

    has_stem_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stem.parameters())
    has_fuse4_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fuse_stage4.parameters())
    has_fuse2_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fuse_stage2.parameters())
    has_blk23_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.blocks[23].parameters())
    has_blk0_grad = any(p.grad is not None for p in model.encoder.blocks[0].parameters())

    print(f"  Block 23 Grad: {has_blk23_grad} | Block 0 Grad: {has_blk0_grad} (expected False)")
    print(f"  Stem Grad: {has_stem_grad} | Fuse4 Grad: {has_fuse4_grad} | Fuse2 Grad: {has_fuse2_grad}")
    assert has_stem_grad and has_fuse4_grad and has_fuse2_grad and has_blk23_grad and not has_blk0_grad
    print("  Gradient Flow Check: PASS (Only trainable blocks & detail stems receive gradients!)")

    print("\n" + "=" * 76)
    print("ALL DINOv2HighResDPT TESTS PASSED 100%!")
    print("=" * 76)

if __name__ == '__main__':
    test_dpt()
