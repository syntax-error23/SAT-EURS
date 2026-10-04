"""
Checkpoint Roundtrip Parity Test (Stage 6 Verification)
Verifies that saving and reloading model weights reproduces exact numerical predictions
within tolerance (atol=1e-6).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import EURSNetV2


def test_checkpoint_numerical_parity():
    print("=" * 60)
    print("Test 6.1: Checkpoint Save / Reload Numerical Parity Audit")
    print("=" * 60)

    torch.manual_seed(42)
    model1 = EURSNetV2(num_classes=5).eval()

    dummy_input = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        rest1, seg1 = model1(dummy_input)

    ckpt_path = 'd:/MACVI/audits/temp_checkpoint.pth'
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    try:
        # Save state dict
        torch.save({'model_state_dict': model1.state_dict()}, ckpt_path)

        # Create fresh model instance
        model2 = EURSNetV2(num_classes=5).eval()
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        model2.load_state_dict(ckpt['model_state_dict'])

        with torch.no_grad():
            rest2, seg2 = model2(dummy_input)

        # Numerical prediction parity assertions
        torch.testing.assert_close(rest1, rest2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(seg1, seg2, rtol=1e-5, atol=1e-6)

        max_diff_rest = (rest1 - rest2).abs().max().item()
        max_diff_seg = (seg1 - seg2).abs().max().item()

        print(f"  [PASS] Restoration output numerical parity verified: max diff = {max_diff_rest:.2e} (< 1e-6)")
        print(f"  [PASS] Segmentation logits numerical parity verified: max diff = {max_diff_seg:.2e} (< 1e-6)")
        print("Checkpoint save and reload numerical parity passed!\n")
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


if __name__ == '__main__':
    test_checkpoint_numerical_parity()
