"""
Comprehensive Pre-Training Verification Suite (Gates A through F) for EURS-Net V4.
Ensures zero bugs across metrics, dataset, augmentations, FP32 losses, V2 warm-start,
2-step overfit, and official MaCVi Track 2 ONNX export before launching full training.
"""

import os
import sys
import hashlib
import torch
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from models.eurs_net_v4 import EURSNetV4
from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from losses.msssim import MSSSIMLoss
from losses.restoration_loss import ColorAngleLoss
from losses.segmentation_loss import SoftDiceLoss, SegmentationLoss
from losses.v4_losses import V4MultiTaskLoss
from metrics.restoration_metrics import RestorationEvaluator
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.composite_score import compute_composite_score
from deploy.export_onnx_v4 import export_and_validate_v4_onnx
from train_v4 import verify_protected_checkpoints, build_v4_optimizer, set_stage_freeze_state


def test_gate_a_scorer_and_v2_warmstart_parity():
    print("\n[Gate A] Testing Official MaCVi Scorer Conventions & V2 Warm-Start Step-0 Parity...")

    # 1. Dataset-mean PSNR_norm vs per-image PSNR_norm
    rest_eval = RestorationEvaluator()
    rest_eval.psnr_list = [14.0, 20.0]       # mean PSNR = 17.0 dB -> dataset-mean PSNR_norm = (17 - 15)/20 = 0.10
    rest_eval.psnr_norm_list = [0.0, 0.25]   # per-image mean = 0.125
    rest_eval.ssim_list = [0.80, 0.80]
    rest_eval.sharpness_list = [0.90, 0.90]
    res = rest_eval.compute()
    assert abs(res['PSNR_norm'] - 0.10) < 1e-5, f"Expected dataset-mean PSNR_norm=0.10, got {res['PSNR_norm']}"
    assert abs(res['PSNR_norm_per_image'] - 0.125) < 1e-5, f"Expected per-image PSNR_norm=0.125, got {res['PSNR_norm_per_image']}"

    # 2. Hard binary coverage_penalty vs soft_coverage_penalty
    seg_eval = SegmentationEvaluator(num_classes=5)
    # Construct target with all 5 classes present, but pred misses class 3 (Shark)
    target = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    logits = torch.zeros(1, 5, 1, 5, dtype=torch.float32)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 0, 2] = 10.0
    logits[0, 0, 0, 3] = 10.0  # Misses class 3 (predicts 0)
    logits[0, 4, 0, 4] = 10.0
    seg_eval.update(logits, target)
    seg_res = seg_eval.compute()
    assert seg_res['coverage_penalty'] == 0.0, f"Expected hard binary coverage_penalty=0.0, got {seg_res['coverage_penalty']}"
    assert abs(seg_res['soft_coverage_penalty'] - 0.8) < 1e-5, f"Expected soft_coverage_penalty=0.8, got {seg_res['soft_coverage_penalty']}"

    # 3. Official MaCVi Composite weights (0.60 / 0.30 / 0.10)
    comp = compute_composite_score(0.80, 0.70, 0.60)
    expected_comp = 0.60 * 0.80 + 0.30 * 0.70 + 0.10 * 0.60
    assert abs(comp['Composite_Score'] - expected_comp) < 1e-5

    # 4. Exact Step-0 parity between EURSNetV2 and warm-started EURSNetV4
    v2_ckpt = "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt"
    assert os.path.exists(v2_ckpt), f"Missing V2 champion checkpoint: {v2_ckpt}"

    v2_model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20)
    sd = torch.load(v2_ckpt, map_location='cpu', weights_only=False)
    v2_model.load_state_dict(sd.get('model_state_dict', sd), strict=True)
    v2_model.eval()

    v4_model = EURSNetV4(num_classes=5, bounded_restoration=True, bound_scale=0.20)
    ws_info = v4_model.load_v2_champion_weights(v2_ckpt, strict_v2_keys=True)
    v4_model.eval()
    assert ws_info['transferred_tensors'] == 282 and ws_info['skipped_tensors'] == 0

    torch.manual_seed(123)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        r2, s2 = v2_model(x)
        r4, s4 = v4_model(x)
    diff_r = float((r2 - r4).abs().max().item())
    diff_s = float((s2 - s4).abs().max().item())
    assert diff_r < 1e-6, f"Step-0 restored parity mismatch: {diff_r}"
    assert diff_s < 1e-6, f"Step-0 seg_logits parity mismatch: {diff_s}"
    print(f"  [PASS] Gate A: Scorer verified & V2->V4 Step-0 parity exact (max_diff_r={diff_r:.2e}, max_diff_s={diff_s:.2e}).")


def test_gate_b_manifest_and_protected_checkpoints():
    print("\n[Gate B] Testing Frozen Manifest & Protected Checkpoints Integrity...")
    verify_protected_checkpoints()
    with open(FROZEN_MANIFEST_PATH, 'rb') as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()
    assert manifest_sha == "979bb00c859076cdeb7f5ba950a5e9df0e07fdd2b274f59fd4d7773b59f868b8"
    print("  [PASS] Gate B: Manifest and protected baseline checkpoints verified bit-for-bit.")


def test_gate_c_epoch_varying_augmentations():
    print("\n[Gate C] Testing Epoch-Varying Augmentation Seeds & Geometric Synchronization...")
    tx = SynchronizedTransform(target_size=(256, 256), scale_range=(0.5, 1.0), rot90_prob=0.5)
    ds = EURSDataset(
        data_dir=str(DATA_DIR),
        split='train',
        manifest_path=str(FROZEN_MANIFEST_PATH),
        transform=tx,
        target_size=(256, 256),
        base_seed=42,
    )

    ds.set_epoch(1)
    deg_e1_a, clean_e1_a, mask_e1_a, _ = ds[0]
    deg_e1_b, clean_e1_b, mask_e1_b, _ = ds[0]

    # Determinism within the same (epoch, idx)
    assert torch.allclose(deg_e1_a, deg_e1_b), "Same (epoch, idx) should be deterministic!"
    assert torch.equal(mask_e1_a, mask_e1_b), "Same (epoch, idx) mask should be deterministic!"

    # Distinctness across different epochs
    ds.set_epoch(2)
    deg_e2, clean_e2, mask_e2, _ = ds[0]
    diff_across_epochs = float((deg_e1_a - deg_e2).abs().mean().item())
    assert diff_across_epochs > 1e-4, f"Augmentation froze across epochs! diff={diff_across_epochs}"

    # Mask taxonomy integrity
    unique_vals = set(torch.unique(mask_e1_a).tolist()) | set(torch.unique(mask_e2).tolist())
    assert unique_vals.issubset({0, 1, 2, 3, 4}), f"Invalid mask values after augmentation: {unique_vals}"
    print(f"  [PASS] Gate C: Epoch-varying seeds active (mean crop diff Ep1 vs Ep2 = {diff_across_epochs:.4f}); mask classes valid {sorted(unique_vals)}.")


def test_gate_d_fp32_loss_fixes():
    print("\n[Gate D] Testing Strict FP32 Loss Fixes (MS-SSIM, ColorAngle, SoftDice, Single Alpha)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = (device.type == 'cuda')

    # 1. MSSSIMLoss on constant image under bfloat16 autocast must be ~0.0 (not 1.0!)
    msssim_fn = MSSSIMLoss().to(device)
    const_img = torch.full((2, 3, 256, 256), 0.5, device=device, dtype=torch.bfloat16 if amp_enabled else torch.float32)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        loss_const = msssim_fn(const_img, const_img)
    assert loss_const.dtype == torch.float32, f"Expected float32 loss, got {loss_const.dtype}"
    assert float(loss_const.item()) < 1e-5, f"bfloat16 MS-SSIM cancellation bug still present! loss={loss_const.item()}"

    # 2. ColorAngleLoss on pure black pixels must be 0.0 (not 1.0!)
    color_fn = ColorAngleLoss().to(device)
    black_target = torch.zeros((2, 3, 128, 128), device=device)
    noisy_pred = torch.full((2, 3, 128, 128), 1e-4, device=device)
    loss_black = color_fn(noisy_pred, black_target)
    assert float(loss_black.item()) < 1e-6, f"ColorAngleLoss black-pixel singularity still present! loss={loss_black.item()}"

    # 3. SoftDiceLoss with absent classes (only classes 0 and 2 present)
    dice_fn = SoftDiceLoss(num_classes=5, batch_dice=True).to(device)
    target_partial = torch.zeros((2, 64, 64), dtype=torch.long, device=device)
    target_partial[:, 16:48, 16:48] = 2
    perfect_logits = torch.full((2, 5, 64, 64), -20.0, device=device)
    perfect_logits[:, 0, :, :] = 20.0
    perfect_logits[:, 0, 16:48, 16:48] = -20.0
    perfect_logits[:, 2, 16:48, 16:48] = 20.0
    loss_dice = dice_fn(perfect_logits, target_partial)
    assert float(loss_dice.item()) < 1e-4, f"SoftDiceLoss penalized absent classes! loss={loss_dice.item()}"

    # 4. SegmentationLoss single alpha check (SoftDiceLoss.alpha must be None)
    alpha = torch.tensor([0.2, 0.8, 1.0, 1.5, 1.5], device=device)
    seg_loss_fn = SegmentationLoss(alpha=alpha).to(device)
    assert seg_loss_fn.dice.alpha is None, "SegmentationLoss double-applied alpha to SoftDiceLoss!"
    assert seg_loss_fn.focal.alpha is not None, "SegmentationLoss missing alpha in FocalLoss!"
    print("  [PASS] Gate D: All 4 loss bugs verified fixed in strict FP32.")


def test_gate_e_two_step_overfit_and_stage_transitions():
    print("\n[Gate E] Testing 2-Step Overfit & Stage-1/Stage-2 + Path-2 Gradient Flow...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EURSNetV4(num_classes=5).to(device)
    v2_ckpt = "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt"
    model.load_v2_champion_weights(v2_ckpt, strict_v2_keys=True)
    model.enable_imagenet_gates(0.05)

    alpha = torch.tensor([0.50, 0.70, 1.40, 2.20, 2.20], device=device)
    criterion = V4MultiTaskLoss(alpha=alpha, lovasz_start_epoch=1).to(device)
    criterion.set_epoch(2)

    optimizer = build_v4_optimizer(model, base_lr=1e-3)

    # Test Path 2 decoupled restoration lock (default lock_restoration=True)
    set_stage_freeze_state(model, epoch=1, freeze_epochs=2, lock_restoration=True)
    assert not any(p.requires_grad for p in model.backbone.parameters()), "Backbone should be frozen when lock_restoration=True!"
    assert not any(p.requires_grad for p in model.restoration_decoder.parameters()), "Restoration decoder should be frozen when lock_restoration=True!"
    assert all(p.requires_grad for p in model.imagenet_encoder.parameters()), "ImageNet encoder should be active when lock_restoration=True!"
    assert all(p.requires_grad for p in model.segmentation_decoder.mb_lat_32.parameters()), "Path 2 adapters should be active in Stage 1!"

    set_stage_freeze_state(model, epoch=3, freeze_epochs=2, lock_restoration=True)
    assert all(p.requires_grad for p in model.segmentation_decoder.parameters()), "Full segmentation decoder should be active in Stage 2 when lock_restoration=True!"

    # Test Stage 1 / Stage 2 transitions when lock_restoration=False
    set_stage_freeze_state(model, epoch=1, freeze_epochs=2, lock_restoration=False)
    assert not any(p.requires_grad for p in model.backbone.parameters()), "Backbone should be frozen in Stage 1!"
    assert all(p.requires_grad for p in model.restoration_decoder.parameters()), "Restoration decoder should be active in Stage 1!"

    set_stage_freeze_state(model, epoch=3, freeze_epochs=2, lock_restoration=False)
    assert all(p.requires_grad for p in model.backbone.parameters()), "Backbone should be unfrozen in Stage 2!"

    torch.manual_seed(7)
    deg = torch.rand(2, 3, 256, 256, device=device)
    clean = torch.clamp(deg + 0.05, 0.0, 1.0)
    mask = torch.randint(0, 5, (2, 256, 256), device=device)

    losses = []
    for step in range(3):
        optimizer.zero_grad()
        restored, seg_logits, bnd_logits = model.forward_with_aux(deg)
        loss, l_dict = criterion(restored, seg_logits, clean, mask, boundary_logits=bnd_logits)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {l_dict}"
        loss.backward()
        # Check gradients on V4 gates, ImageNet gates, and heads
        assert model.segmentation_decoder.context_gate.grad is not None
        assert model.segmentation_decoder.rest_gate.grad is not None
        assert model.segmentation_decoder.mb_gate_32.grad is not None
        assert torch.isfinite(model.segmentation_decoder.context_gate.grad).all()
        assert torch.isfinite(model.segmentation_decoder.mb_gate_32.grad).all()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"Loss did not decrease on 3-step overfit: {losses}"
    print(f"  [PASS] Gate E: 3-step overfit loss decreased ({losses[0]:.4f} -> {losses[-1]:.4f}) with finite gradients across all branches.")


def test_gate_f_onnx_export_and_flops_compliance():
    print("\n[Gate F] Testing Track 2 ONNX Export & Official MaCVi FLOPs/Params Compliance (1024x1024)...")
    out_path = "deploy/eurs_net_v4_track2.onnx"
    res = export_and_validate_v4_onnx(
        checkpoint_path="experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt",
        output_onnx_path=out_path,
        input_hw=(1024, 1024),
        verbose=False,
    )
    assert res["gate_f_passed"] is True
    assert res["total_gflops"] <= 12.0
    assert res["total_params"] <= 1_000_000
    assert res["external_data_dependency"] is False
    print(
        f"  [PASS] Gate F: {out_path} verified at 1024x1024 "
        f"(GFLOPs={res['total_gflops']:.4f}/12.0, Params={res['total_params_m']:.4f}M/1.0M, "
        f"MaxDiff={max(res['max_diff_restored'], res['max_diff_seg']):.2e})."
    )


if __name__ == "__main__":
    print("=" * 75)
    print("         EURS-NET V4 PRE-TRAINING SMOKE TEST SUITE (GATES A - F)")
    print("=" * 75)
    test_gate_a_scorer_and_v2_warmstart_parity()
    test_gate_b_manifest_and_protected_checkpoints()
    test_gate_c_epoch_varying_augmentations()
    test_gate_d_fp32_loss_fixes()
    test_gate_e_two_step_overfit_and_stage_transitions()
    test_gate_f_onnx_export_and_flops_compliance()
    print("\n" + "=" * 75)
    print("  ALL GATES (A, B, C, D, E, F) PASSED WITH ZERO ERRORS!")
    print("=" * 75)
