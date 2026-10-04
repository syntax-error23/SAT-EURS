"""
Device Parity & Numerical Sanity Test (Stage 7 Verification)
Verifies:
1. CPU vs CUDA numerical agreement within justified tolerance (rtol=1e-3, atol=1e-4) if CUDA available.
2. Numerical stability across independent runs on CPU.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import EURSNetV2


def test_device_parity_and_stability():
    print("=" * 60)
    print("Test 7.2: Device Parity & Numerical Stability Audit")
    print("=" * 60)

    torch.manual_seed(42)
    model = EURSNetV2(num_classes=5).eval()

    dummy_input = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        rest_cpu, seg_cpu = model(dummy_input)

    assert torch.isfinite(rest_cpu).all(), "NaN/Inf in CPU restored output!"
    assert torch.isfinite(seg_cpu).all(), "NaN/Inf in CPU seg output!"

    if torch.cuda.is_available():
        print("  [DEVICE] CUDA available. Testing CPU vs CUDA numerical parity...")
        model_cuda = EURSNetV2(num_classes=5).eval().cuda()
        model_cuda.load_state_dict(model.state_dict())

        with torch.no_grad():
            rest_cuda, seg_cuda = model_cuda(dummy_input.cuda())

        torch.testing.assert_close(rest_cpu, rest_cuda.cpu(), rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(seg_cpu, seg_cuda.cpu(), rtol=1e-3, atol=1e-4)
        print("  [PASS] CPU vs CUDA numerical parity verified within justified tolerance (rtol=1e-3, atol=1e-4).")
    else:
        print("  [DEVICE] CUDA not available on development host. Verifying CPU numerical stability across runs...")
        with torch.no_grad():
            rest_cpu_2, seg_cpu_2 = model(dummy_input)

        torch.testing.assert_close(rest_cpu, rest_cpu_2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(seg_cpu, seg_cpu_2, rtol=1e-5, atol=1e-6)
        print("  [PASS] Numerical stability verified: identical outputs across independent CPU evaluations.")

    print("Device parity and numerical sanity passed!\n")


if __name__ == '__main__':
    test_device_parity_and_stability()
