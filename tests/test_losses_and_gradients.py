"""
Loss Contracts & Module-Level Isolated Gradient Audits (Stage 4 Verification)
Tests:
1. Individual loss unit tests (Focal, Dice, Charbonnier, MS-SSIM, Laplacian).
2. MS-SSIM known-value synthetic fixtures.
3. Module-level isolated backward tests:
   - Seg-only backward -> Rest decoder norm == 0.0
   - Rest-only backward -> Seg decoder norm == 0.0
   - Total loss backward -> All module norms > 0.0
4. Shared backbone gradient ratio logging.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EURSNetV2
from losses import (
    ClassWeightedFocalLoss,
    SoftDiceLoss,
    SegmentationLoss,
    ms_ssim,
    MSSSIMLoss,
    CharbonnierLoss,
    LaplacianLoss,
    RestorationLoss,
    MultiTaskLoss
)


def get_module_grad_norm(module: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all parameters of a module."""
    total_norm_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5


def test_individual_loss_contracts():
    print("=" * 60)
    print("Test 4.1: Individual Loss Unit Tests & Fixtures")
    print("=" * 60)

    # Dummy inputs: [2, 3, 256, 256] and [2, 256, 256]
    b, c, h, w = 2, 3, 256, 256
    num_classes = 5
    pred_rgb = torch.rand(b, c, h, w, requires_grad=True)
    target_rgb = torch.rand(b, c, h, w)

    logits = torch.randn(b, num_classes, h, w, requires_grad=True)
    targets_mask = torch.randint(0, num_classes, (b, h, w))

    # 1. Focal Loss
    alpha = torch.tensor([0.1, 0.3, 0.2, 0.3, 0.1])
    focal_fn = ClassWeightedFocalLoss(alpha=alpha, gamma=2.0)
    l_focal = focal_fn(logits, targets_mask)
    assert torch.isfinite(l_focal) and l_focal.item() > 0, "Focal loss invalid!"
    print(f"  [PASS] Focal Loss computed: {l_focal.item():.4f}")

    # 2. Soft Dice Loss
    dice_fn = SoftDiceLoss(num_classes=num_classes)
    l_dice = dice_fn(logits, targets_mask)
    assert torch.isfinite(l_dice) and 0.0 <= l_dice.item() <= 1.0, f"Dice loss out of bounds: {l_dice.item()}"
    print(f"  [PASS] Soft Dice Loss computed: {l_dice.item():.4f}")

    # 3. Charbonnier Loss
    charb_fn = CharbonnierLoss(eps=1e-3)
    l_charb = charb_fn(pred_rgb, target_rgb)
    assert torch.isfinite(l_charb) and l_charb.item() > 0, "Charbonnier loss invalid!"
    print(f"  [PASS] Charbonnier Loss computed: {l_charb.item():.4f}")

    # 4. MS-SSIM Loss & Fixtures
    # Fixture A: Identical images -> MS-SSIM == 1.0 -> Loss == 0.0
    identical_loss = MSSSIMLoss(data_range=1.0)(target_rgb, target_rgb)
    assert identical_loss.item() < 1e-4, f"Identical MS-SSIM loss should be ~0, got {identical_loss.item()}"
    print(f"  [PASS] MS-SSIM Fixture A (Identical): loss = {identical_loss.item():.6f} (< 1e-4)")

    # Fixture B: Inverted images -> MS-SSIM << 1.0 -> Loss > 0.4
    inverted_rgb = 1.0 - target_rgb
    inverted_loss = MSSSIMLoss(data_range=1.0)(inverted_rgb, target_rgb)
    assert inverted_loss.item() > 0.4, f"Inverted MS-SSIM loss should be large, got {inverted_loss.item()}"
    print(f"  [PASS] MS-SSIM Fixture B (Inverted):  loss = {inverted_loss.item():.4f} (> 0.4)")

    # 5. Laplacian Loss
    lap_fn = LaplacianLoss()
    l_lap = lap_fn(pred_rgb, target_rgb)
    assert torch.isfinite(l_lap) and l_lap.item() > 0, "Laplacian loss invalid!"
    print(f"  [PASS] Laplacian Loss computed: {l_lap.item():.4f}")
    print("All individual loss contracts passed!\n")


def test_isolated_module_gradient_audits():
    print("=" * 60)
    print("Test 4.2: Module-Level Isolated Backward Audits")
    print("=" * 60)

    # Initialize model
    model = EURSNetV2(num_classes=5)
    model.train()

    dummy_input = torch.rand(1, 3, 256, 256)
    clean_target = torch.rand(1, 3, 256, 256)
    mask_target = torch.randint(0, 5, (1, 256, 256))

    mt_loss_fn = MultiTaskLoss(weight_seg=1.0, weight_rest=0.5, num_classes=5)

    # --- Part 1: Segmentation-Only Backward ---
    model.zero_grad()
    restored, seg_logits = model(dummy_input)
    # Resize logits to match dummy target
    seg_logits_resized = torch.nn.functional.interpolate(seg_logits, size=(256, 256), mode='bilinear', align_corners=False)
    l_seg = mt_loss_fn.seg_loss_fn(seg_logits_resized, mask_target)
    l_seg.backward(retain_graph=True)

    norm_seg_decoder_s = get_module_grad_norm(model.segmentation_decoder)
    norm_backbone_s    = get_module_grad_norm(model.backbone)
    norm_rest_decoder_s = get_module_grad_norm(model.restoration_decoder)

    assert norm_seg_decoder_s > 0, "Segmentation decoder has 0 gradient from L_seg!"
    assert norm_backbone_s > 0, "Backbone has 0 gradient from L_seg!"
    assert norm_rest_decoder_s == 0.0, f"Restoration decoder improperly received gradient from L_seg: {norm_rest_decoder_s}"
    print(f"  [PASS] L_seg Isolated Backward:")
    print(f"    - Segmentation Decoder Grad Norm: {norm_seg_decoder_s:.6f} (> 0)")
    print(f"    - Shared Backbone Grad Norm:      {norm_backbone_s:.6f} (> 0)")
    print(f"    - Restoration Decoder Grad Norm:  {norm_rest_decoder_s:.6f} (== 0.0, ISOLATED)")

    # --- Part 2: Restoration-Only Backward ---
    model.zero_grad()
    restored_resized = torch.nn.functional.interpolate(restored, size=(256, 256), mode='bilinear', align_corners=False)
    l_rest = mt_loss_fn.rest_loss_fn(restored_resized, clean_target)
    l_rest.backward(retain_graph=True)

    norm_rest_decoder_r = get_module_grad_norm(model.restoration_decoder)
    norm_backbone_r    = get_module_grad_norm(model.backbone)
    norm_seg_decoder_r = get_module_grad_norm(model.segmentation_decoder)

    assert norm_rest_decoder_r > 0, "Restoration decoder has 0 gradient from L_rest!"
    assert norm_backbone_r > 0, "Backbone has 0 gradient from L_rest!"
    assert norm_seg_decoder_r == 0.0, f"Segmentation decoder improperly received gradient from L_rest: {norm_seg_decoder_r}"
    print(f"  [PASS] L_rest Isolated Backward:")
    print(f"    - Restoration Decoder Grad Norm:  {norm_rest_decoder_r:.6f} (> 0)")
    print(f"    - Shared Backbone Grad Norm:      {norm_backbone_r:.6f} (> 0)")
    print(f"    - Segmentation Decoder Grad Norm: {norm_seg_decoder_r:.6f} (== 0.0, ISOLATED)")

    # --- Part 3: Total Multi-Task Backward ---
    model.zero_grad()
    l_total, loss_dict = mt_loss_fn(restored_resized, seg_logits_resized, clean_target, mask_target)
    l_total.backward()

    norm_rest_tot = get_module_grad_norm(model.restoration_decoder)
    norm_seg_tot  = get_module_grad_norm(model.segmentation_decoder)
    norm_bb_tot   = get_module_grad_norm(model.backbone)

    assert norm_rest_tot > 0, "Restoration decoder has 0 gradient from L_total!"
    assert norm_seg_tot > 0, "Segmentation decoder has 0 gradient from L_total!"
    assert norm_bb_tot > 0, "Backbone has 0 gradient from L_total!"

    print(f"  [PASS] L_total Multi-Task Backward:")
    print(f"    - Restoration Decoder Grad Norm:  {norm_rest_tot:.6f} (> 0)")
    print(f"    - Segmentation Decoder Grad Norm: {norm_seg_tot:.6f} (> 0)")
    print(f"    - Shared Backbone Grad Norm:      {norm_bb_tot:.6f} (> 0)")

    # --- Part 4: Backbone Gradient Ratio ---
    ratio_s_to_r = norm_backbone_s / max(norm_backbone_r, 1e-8)
    print(f"  [AUDIT] Backbone Gradient Ratio (||g_seg|| / ||g_rest||): {ratio_s_to_r:.4f}")
    print("Isolated module-level gradient audits passed!\n")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  RUNNING STAGE 4 MULTI-TASK LOSS & GRADIENT AUDITS")
    print("#" * 60 + "\n")

    test_individual_loss_contracts()
    test_isolated_module_gradient_audits()

    print("=" * 60)
    print("STAGE 4 VERIFICATION PASSED: CLEAN ISOLATION & JOINT GRADIENT FLOW")
    print("=" * 60 + "\n")
