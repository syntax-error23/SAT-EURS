"""
Comprehensive Test Suite for EURS-Net V2 Architecture.
Verifies:
1. Intermediate backbone feature map dimensions.
2. Full network forward pass output shapes and value bounds.
3. Finite gradient flow across 100% of trainable parameters.
4. Parameter budget (Hard limit: 1.0M, Target: <= 0.60M).
5. FLOP budget (Hard limit: 12.0 GFLOPs at 1024x1024).
6. ONNX export dry-run sanity check.
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EURSNetV2, SharedBackbone, ProgressiveRestorationDecoder, ProgressiveSegmentationDecoder


def test_backbone_shapes():
    print("=" * 60)
    print("Test 1: Backbone Feature Map Dimensions Audit")
    print("=" * 60)
    backbone = SharedBackbone()
    backbone.eval()

    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        stem, f256, f128, f64, f32 = backbone(dummy_input)

    expected_shapes = [
        ("Stem", stem.shape, (1, 16, 512, 512)),
        ("F256", f256.shape, (1, 24, 256, 256)),
        ("F128", f128.shape, (1, 40, 128, 128)),
        ("F64",  f64.shape,  (1, 64, 64, 64)),
        ("F32",  f32.shape,  (1, 96, 32, 32)),
    ]

    for name, actual, expected in expected_shapes:
        assert actual == expected, f"Shape mismatch for {name}: got {actual}, expected {expected}"
        print(f"  [PASS] {name:5s} -> {list(actual)}")
    print("All intermediate backbone dimensions verified successfully!\n")


def test_full_model_forward_and_bounds():
    print("=" * 60)
    print("Test 2: Full EURS-Net V2 Forward Pass & Bounds Audit")
    print("=" * 60)
    model = EURSNetV2(num_classes=5)
    model.eval()

    dummy_input = torch.rand(1, 3, 1024, 1024) # [0, 1] range
    with torch.no_grad():
        restored_rgb, seg_logits = model(dummy_input)

    assert restored_rgb.shape == (1, 3, 1024, 1024), f"Restoration shape error: {restored_rgb.shape}"
    assert seg_logits.shape == (1, 5, 1024, 1024), f"Segmentation shape error: {seg_logits.shape}"

    assert torch.isfinite(restored_rgb).all(), "Restoration output contains non-finite values (NaN / Inf)!"
    assert torch.isfinite(seg_logits).all(), "Segmentation output contains non-finite values (NaN / Inf)!"

    assert restored_rgb.min() >= 0.0 and restored_rgb.max() <= 1.0, (
        f"Restored RGB values outside [0, 1]! min={restored_rgb.min().item():.4f}, max={restored_rgb.max().item():.4f}"
    )

    print(f"  [PASS] Restored RGB Output Shape: {list(restored_rgb.shape)}")
    print(f"  [PASS] Restored RGB Range:        [{restored_rgb.min().item():.4f}, {restored_rgb.max().item():.4f}]")
    print(f"  [PASS] Segmentation Logits Shape: {list(seg_logits.shape)}")
    print(f"  [PASS] Finite values verified across all outputs.")
    print("Forward pass and bounds audit passed!\n")


def test_gradient_flow():
    print("=" * 60)
    print("Test 3: Backward Pass & Finite Gradient Flow Audit")
    print("=" * 60)
    model = EURSNetV2(num_classes=5)
    model.train()

    dummy_input = torch.rand(1, 3, 1024, 1024, requires_grad=True)
    restored_rgb, seg_logits = model(dummy_input)

    # Compute joint dummy loss
    loss = restored_rgb.mean() + seg_logits.mean()
    loss.backward()

    missing_grads = []
    non_finite_grads = []
    total_trainable = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.grad is None:
                missing_grads.append(name)
            elif not torch.isfinite(param.grad).all():
                non_finite_grads.append(name)

    assert len(missing_grads) == 0, f"Missing gradients in parameters: {missing_grads}"
    assert len(non_finite_grads) == 0, f"Non-finite gradients in parameters: {non_finite_grads}"

    print(f"  [PASS] All {total_trainable} trainable parameter tensors received valid, finite gradients.")
    print("Gradient flow audit passed!\n")


def test_parameter_and_flop_budgets():
    print("=" * 60)
    print("Test 4: Parameter & FLOP Budgets Audit")
    print("=" * 60)
    model = EURSNetV2(num_classes=5)
    model.eval()

    fwb_params = sum(p.numel() for p in model.fwb.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    restoration_params = sum(p.numel() for p in model.restoration_decoder.parameters())
    seg_params = sum(p.numel() for p in model.segmentation_decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  - FastWhiteBalance:        {fwb_params:,} parameters")
    print(f"  - SharedBackbone:          {backbone_params:,} parameters ({backbone_params / 1e6:.4f}M)")
    print(f"  - ProgressiveRestoration:  {restoration_params:,} parameters ({restoration_params / 1e6:.4f}M)")
    print(f"  - ProgressiveSegmentation: {seg_params:,} parameters ({seg_params / 1e6:.4f}M)")
    print(f"  --------------------------------------------------")
    print(f"  TOTAL PARAMETERS:          {total_params:,} ({total_params / 1e6:.4f}M)")

    # Assertions
    assert total_params <= 1_000_000, f"Hard limit violated: {total_params} > 1,000,000"
    assert total_params <= 600_000, f"Target budget violated: {total_params} > 600,000"
    print(f"  [PASS] Parameter budget: {total_params:,} <= 600,000 target (and <= 1.0M hard limit).")

    # FLOPs computation via ptflops
    macs, params = get_model_complexity_info(
        model,
        (3, 1024, 1024),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    flops = 2 * macs
    gflops = flops / 1e9

    print(f"  TOTAL COMPUTE (MACS):      {macs / 1e9:.3f} GMACs")
    print(f"  TOTAL COMPUTE (FLOPS):     {gflops:.3f} GFLOPs")

    assert gflops <= 12.0, f"Compute limit violated: {gflops:.2f} GFLOPs > 12.0 GFLOPs"
    print(f"  [PASS] Compute budget: {gflops:.3f} GFLOPs <= 12.0 GFLOPs hard ceiling.\n")


def test_onnx_export_dryrun():
    print("=" * 60)
    print("Test 5: ONNX Export Dry-Run Check")
    print("=" * 60)
    model = EURSNetV2(num_classes=5)
    model.eval()

    dummy_input = torch.randn(1, 3, 1024, 1024)
    onnx_path = os.path.join(os.path.dirname(__file__), 'temp_test_export.onnx')

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['restored_rgb', 'seg_logits'],
            dynamic_axes=None,  # Strict static shape for TensorRT Jetson Track 2
            dynamo=False
        )
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  [PASS] Model successfully exported to ONNX (Opset 18).")
        print(f"  [PASS] Exported file size: {file_size_mb:.2f} MB")
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"  [CLEANUP] Temporary ONNX test artifact removed.\n")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  RUNNING EURS-NET V2 ARCHITECTURE VERIFICATION SUITE")
    print("#" * 60 + "\n")

    test_backbone_shapes()
    test_full_model_forward_and_bounds()
    test_gradient_flow()
    test_parameter_and_flop_budgets()
    test_onnx_export_dryrun()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ARCHITECTURE FULLY VERIFIED.")
    print("=" * 60 + "\n")
