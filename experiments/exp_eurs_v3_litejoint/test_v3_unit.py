"""
Unit tests and gradient flow verification for EURS-Net V3 LiteJoint
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments/exp_eurs_v3_litejoint'))

import torch
import torch.nn as nn
from model_v3 import EURSNetV3LiteJoint
from losses_v3 import MultiTaskLossV3, get_v3_class_weights


def test_model_forward():
    print("[TEST 1] Testing Forward Pass on EURSNetV3LiteJoint...")
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode='bounded', bound_scale=0.20)
    model.eval()

    x = torch.rand(2, 3, 1024, 1024)
    with torch.no_grad():
        restored, seg = model(x)

    assert restored.shape == (2, 3, 1024, 1024), f"Incorrect restored shape: {restored.shape}"
    assert seg.shape == (2, 5, 1024, 1024), f"Incorrect seg shape: {seg.shape}"
    assert restored.min() >= 0.0 and restored.max() <= 1.0, f"Restored values out of [0, 1]: [{restored.min()}, {restored.max()}]"
    assert torch.isfinite(restored).all(), "Non-finite values in restored"
    assert torch.isfinite(seg).all(), "Non-finite values in seg"
    print("  -> PASSED: Forward pass produces correct shapes and valid ranges.")


def test_gradient_flow():
    print("[TEST 2] Testing End-to-End Gradient Flow...")
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode='bounded', bound_scale=0.20)
    model.train()

    weights = get_v3_class_weights()
    criterion = MultiTaskLossV3(num_classes=5, alpha_seg=weights, weight_seg=1.0, weight_rest=0.5)

    x = torch.rand(2, 3, 1024, 1024, requires_grad=False)
    clean = torch.rand(2, 3, 1024, 1024, requires_grad=False)
    mask = torch.randint(0, 5, (2, 1024, 1024), dtype=torch.long)

    restored, seg = model(x)
    loss, loss_dict = criterion(restored, seg, clean, mask)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()

    zero_grad_params = []
    nan_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_params.append(name)
            elif not torch.isfinite(param.grad).all():
                nan_grad_params.append(name)

    assert len(nan_grad_params) == 0, f"NaN gradients in: {nan_grad_params}"
    assert len(zero_grad_params) == 0, f"Missing gradients in: {zero_grad_params}"
    print(f"  -> PASSED: All {len(list(model.parameters()))} parameter tensors received valid, finite gradients!")


def test_tiny_batch_overfit():
    print("[TEST 3] Testing 10-Step Tiny Batch Overfitting...")
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode='bounded', bound_scale=0.20)
    model.train()

    weights = get_v3_class_weights()
    criterion = MultiTaskLossV3(num_classes=5, alpha_seg=weights, weight_seg=1.0, weight_rest=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Fixed synthetic batch
    torch.manual_seed(42)
    x = torch.rand(1, 3, 256, 256)
    clean = torch.rand(1, 3, 256, 256)
    mask = torch.randint(0, 5, (1, 256, 256), dtype=torch.long)

    initial_loss = None
    final_loss = None
    for step in range(10):
        optimizer.zero_grad()
        restored, seg = model(x)
        loss, l_dict = criterion(restored, seg, clean, mask)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    print(f"  -> PASSED: Overfit test converged from {initial_loss:.4f} down to {final_loss:.4f}.")


if __name__ == '__main__':
    test_model_forward()
    test_gradient_flow()
    test_tiny_batch_overfit()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
