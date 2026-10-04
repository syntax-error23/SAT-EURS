"""
Learning Rate Scheduler Unit Test (Stage 8 Verification)
Verifies:
1. Monotonic linear warmup from initial lr/warmup_epochs to base_lr over warmup epochs.
2. Smooth cosine decay from base_lr down to min_lr.
3. LR never drops below min_lr.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int = 5,
    total_epochs: int = 50,
    min_lr_ratio: float = 1e-3
):
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def test_lr_scheduler_dynamics():
    print("=" * 60)
    print("Test 8.1: Learning Rate Scheduler Dynamics Audit")
    print("=" * 60)

    dummy_param = [torch.nn.Parameter(torch.zeros(1))]
    base_lr = 5e-4
    warmup_epochs = 5
    total_epochs = 50
    min_lr = 1e-6
    min_lr_ratio = min_lr / base_lr

    optimizer = optim.AdamW(dummy_param, lr=base_lr)
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr_ratio=min_lr_ratio
    )

    lrs = []
    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        optimizer.step()
        scheduler.step()

    # 1. Warmup check: strictly monotonic increasing
    for e in range(1, warmup_epochs):
        assert lrs[e] > lrs[e - 1], f"Warmup failed monotonicity at epoch {e}: {lrs[e]} <= {lrs[e-1]}"
    print(f"  [PASS] Warmup phase (epochs 0-{warmup_epochs-1}) is strictly monotonically increasing:")
    print(f"    - Epoch 0 LR: {lrs[0]:.2e}")
    print(f"    - Epoch 4 LR: {lrs[4]:.2e} (Base LR reached: {base_lr:.2e})")

    # 2. Cosine decay check: decreasing towards min_lr
    for e in range(warmup_epochs, total_epochs - 1):
        assert lrs[e + 1] <= lrs[e], f"Cosine decay failed monotonicity at epoch {e}: {lrs[e+1]} > {lrs[e]}"

    assert abs(lrs[-1] - min_lr) < 1e-7, f"Final LR should match min_lr ({min_lr}), got {lrs[-1]}"
    assert min(lrs) >= min_lr - 1e-9, f"LR dropped below min_lr: min={min(lrs)}"

    print(f"  [PASS] Cosine decay phase (epochs {warmup_epochs}-{total_epochs-1}) decays smoothly to min_lr:")
    print(f"    - Epoch 10 LR: {lrs[10]:.2e}")
    print(f"    - Epoch 25 LR: {lrs[25]:.2e}")
    print(f"    - Final LR:    {lrs[-1]:.2e} (Matches min_lr: {min_lr:.2e})")
    print("Learning rate scheduler unit test passed!\n")


if __name__ == '__main__':
    test_lr_scheduler_dynamics()
