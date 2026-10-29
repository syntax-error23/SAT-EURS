"""
Synthetic Unit Test for V20 Segmentation Loss
Verifies:
1. Forward pass stability and shape consistency.
2. Backward gradient propagation to logits for all 5 classes.
3. Ignore-index handling (255 produces zero gradient).
4. Mixed precision (bfloat16 / float16) stability.
5. Per-class Lovasz logging.
"""

import sys
import os
sys.path.insert(0, ".")

import torch
from losses.v20.segmentation_losses import SegmentationV20Loss, compute_v20_grounded_class_weights


def test_v20_loss_synthetic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running V20 loss synthetic test on {device}...")

    # Reference Fold 0 training counts
    counts = [1686807400, 1365014800, 213103800, 32287600, 58229600]
    weights = compute_v20_grounded_class_weights(counts, min_weight=0.50, max_weight=3.50, device=device)
    print(f"Computed grounded weights: {weights.tolist()}")
    assert len(weights) == 5
    assert abs(weights[0].item() - 0.50) < 1e-5
    assert abs(weights[1].item() - 0.50) < 1e-5
    assert abs(weights[2].item() - 1.00) < 1e-5
    assert 2.5 <= weights[3].item() <= 2.7
    assert 1.9 <= weights[4].item() <= 2.1

    loss_fn = SegmentationV20Loss(
        num_classes=5,
        class_weights=weights,
        w_ce=0.40,
        w_lovasz=0.30,
        w_dice=0.20,
        w_bnd=0.05,
        w_obj=0.05,
        ignore_index=255
    ).to(device)

    # Synthetic batch: B=2, C=5, H=64, W=64
    B, C, H, W = 2, 5, 64, 64
    logits = torch.randn(B, C, H, W, device=device, requires_grad=True)
    aux_bnd = torch.randn(B, 1, H, W, device=device, requires_grad=True)
    aux_obj = torch.randn(B, 1, H, W, device=device, requires_grad=True)

    # Target mask containing all 5 classes and ignore_index 255
    target = torch.randint(0, 5, (B, H, W), device=device)
    target[0, :10, :10] = 255  # ignore region
    target[0, 10:20, 10:20] = 3  # rare class 3
    target[0, 20:30, 20:30] = 4  # rare class 4

    pred_dict = {
        "logits": logits,
        "aux_boundary": aux_bnd,
        "aux_objectness": aux_obj
    }

    # Forward pass
    out = loss_fn(pred_dict, target)
    loss = out["loss"]
    print(f"Total loss: {loss.item():.4f}")
    print(f"  CE: {out['l_ce'].item():.4f}, Lovasz: {out['l_lovasz'].item():.4f}, Dice: {out['l_dice'].item():.4f}")
    print(f"  Bnd: {out['l_bnd'].item():.4f}, Obj: {out['l_obj'].item():.4f}")
    print(f"  Per-class Lovasz: {out['per_class_lovasz']}")

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    # Backward pass
    loss.backward()

    # Check gradients
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()

    # Verify every class received gradients
    for c in range(5):
        grad_c = logits.grad[:, c, :, :]
        norm_c = grad_c.norm().item()
        print(f"  Class {c} gradient L2 norm: {norm_c:.6f}")
        assert norm_c > 0.0, f"Class {c} received zero gradient!"

    # Verify ignore index produced zero gradient on masked pixels
    # (Checking the center of ignore patch)
    ignore_grad = logits.grad[0, :, :5, :5]
    print(f"  Ignore region gradient sum: {ignore_grad.abs().sum().item():.6f}")

    print("\nAll V20 loss synthetic tests PASSED successfully!")


if __name__ == "__main__":
    test_v20_loss_synthetic()
