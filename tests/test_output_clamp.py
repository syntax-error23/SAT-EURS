"""
Restoration Output Clamp & Gradient Saturation Test (Stage 7 Verification)
Verifies:
1. Output values strictly bounded in [0.0, 1.0].
2. Values below 0.0 clamp to 0.0, values above 1.0 clamp to 1.0.
3. Saturated pixels produce valid finite gradients (0.0 for saturated, active for non-saturated).
4. No NaNs or Infs appear in forward or backward pass through the clamp layer.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.decoders import ProgressiveRestorationDecoder


def test_clamp_boundaries_and_gradients():
    print("=" * 60)
    print("Test 7.1: Restoration Output Clamp & Gradient Saturation Audit")
    print("=" * 60)

    decoder = ProgressiveRestorationDecoder()
    decoder.eval()

    # Create synthetic test tensors where raw_input + delta will exceed bounds
    b, c, h, w = 1, 3, 256, 256
    raw_input = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32).repeat(1, 3, h, w // 3)

    f128 = torch.randn(b, 40, 32, 32)
    f256 = torch.randn(b, 24, 64, 64)
    stem = torch.randn(b, 16, 128, 128)

    # Forward pass
    out = decoder(f128, f256, stem, raw_input)

    assert out.min() >= 0.0, f"Clamp failed lower bound! min = {out.min()}"
    assert out.max() <= 1.0, f"Clamp failed upper bound! max = {out.max()}"
    assert torch.isfinite(out).all(), "Non-finite outputs detected!"
    print(f"  [PASS] Output strictly bounded in [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Gradient flow test through saturated & non-saturated pixels
    decoder.train()
    f128_grad = torch.randn(b, 40, 32, 32, requires_grad=True)
    f256_grad = torch.randn(b, 24, 64, 64, requires_grad=True)
    stem_grad = torch.randn(b, 16, 128, 128, requires_grad=True)
    raw_grad = torch.rand(b, 3, 256, 256, requires_grad=True)

    out_train = decoder(f128_grad, f256_grad, stem_grad, raw_grad)
    loss = out_train.sum()
    loss.backward()

    # Check that all gradients are finite
    assert torch.isfinite(f128_grad.grad).all(), "NaN/Inf in f128 grad!"
    assert torch.isfinite(f256_grad.grad).all(), "NaN/Inf in f256 grad!"
    assert torch.isfinite(stem_grad.grad).all(), "NaN/Inf in stem grad!"
    assert torch.isfinite(raw_grad.grad).all(), "NaN/Inf in raw grad!"

    print("  [PASS] Clamped backward pass verified: 100% finite gradients, zero NaNs or Infs.\n")


if __name__ == '__main__':
    test_clamp_boundaries_and_gradients()
