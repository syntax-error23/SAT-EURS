"""
Unit Test for SegmentationV7Loss
Stage 3 of SAT-EURS Track 1 V7 Plan
"""

import sys
sys.path.insert(0, ".")
import torch
from losses.segmentation_v7_loss import SegmentationV7Loss

def test_seg_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("TESTING SegmentationV7Loss")
    print(f"  Device: {device}")
    print("=" * 76)

    criterion = SegmentationV7Loss(num_classes=5).to(device)

    # Test batch with all classes present
    b, h, w = 2, 256, 256
    logits = torch.randn(b, 5, h, w, device=device, requires_grad=True)
    targets = torch.randint(0, 5, (b, h, w), device=device)

    total_loss, loss_dict = criterion(logits, targets)
    total_loss.backward()

    print(f"  Loss breakdown: {loss_dict}")
    assert logits.grad is not None and logits.grad.abs().sum() > 0, "Logits did not receive gradients!"
    assert not torch.isnan(total_loss), "Loss is NaN!"
    assert not torch.isinf(total_loss), "Loss is Inf!"
    print("  Loss Stability & Gradient Flow: PASS")

    # Test edge case: only class 0 present in batch
    logits2 = torch.randn(b, 5, h, w, device=device, requires_grad=True)
    targets2 = torch.zeros((b, h, w), dtype=torch.long, device=device)

    total_loss2, loss_dict2 = criterion(logits2, targets2)
    total_loss2.backward()
    print(f"  Single-class edge case loss: {loss_dict2}")
    assert not torch.isnan(total_loss2), "Single-class loss is NaN!"
    print("  Single-Class Edge Case: PASS")

    print("\n" + "=" * 76)
    print("ALL SegmentationV7Loss TESTS PASSED 100%!")
    print("=" * 76)

if __name__ == '__main__':
    test_seg_loss()
