"""
Minimal FP32 Debug Training Script for Stage 8 Verification
Verifies:
1. End-to-end dataset -> forward -> loss -> backward -> optimizer.step() loop.
2. Parameter update invariant: parameters_after != parameters_before.
3. 1 complete epoch with validation evaluation.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import EURSNetV2
from data.dataset import EURSDataset
from losses import MultiTaskLoss
from evaluate import evaluate


def run_debug_training(
    data_dir: str = 'd:/MACVI/data/mini_dev',
    manifest_path: str = 'd:/MACVI/splits/manifest.json',
    batch_size: int = 2,
    lr: float = 1e-3,
    target_size: tuple = (256, 256)
):
    print("=" * 60)
    print("Stage 8.1: Minimal FP32 Debug Training Loop & Step Verification")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [DEVICE] Running debug loop on {device.upper()}")

    # 1. Dataset & Loaders
    train_dataset = EURSDataset(
        data_dir=data_dir,
        split='train',
        manifest_path=manifest_path,
        target_size=target_size
    )
    val_dataset = EURSDataset(
        data_dir=data_dir,
        split='val',
        manifest_path=manifest_path,
        target_size=target_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = EURSNetV2(num_classes=5).to(device)
    loss_fn = MultiTaskLoss(weight_seg=1.0, weight_rest=0.5, num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cache parameter tensors before step
    params_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    # 3. Train 1 Epoch
    model.train()
    step = 0
    total_epoch_loss = 0.0

    print(f"\n  Starting Epoch 1 ({len(train_loader)} batches)...")
    for deg, clean, mask, meta in train_loader:
        deg = deg.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        restored_rgb, seg_logits = model(deg)

        # Spatial resize if target_size != 1024
        if restored_rgb.shape[2:] != target_size:
            restored_rgb = torch.nn.functional.interpolate(restored_rgb, size=target_size, mode='bilinear', align_corners=False)
            seg_logits = torch.nn.functional.interpolate(seg_logits, size=target_size, mode='bilinear', align_corners=False)

        loss, l_dict = loss_fn(restored_rgb, seg_logits, clean, mask)
        assert torch.isfinite(loss), f"Non-finite loss detected: {loss.item()}"

        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        step += 1
        print(f"    Batch {step}/{len(train_loader)}: Loss = {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

    # Invariant: Verify optimizer.step() actually updated parameters
    updated_params = 0
    for p, p_before in zip([p for p in model.parameters() if p.requires_grad], params_before):
        if not torch.equal(p, p_before):
            updated_params += 1

    assert updated_params > 0, "FATAL: optimizer.step() failed to update any model parameters!"
    print(f"\n  [PASS] Optimizer Step Verified: {updated_params} parameter tensors updated successfully.")

    # 4. Run Validation
    print("\n  Running validation evaluation at epoch end...")
    val_report = evaluate(model, val_loader, device=device)

    print("Stage 8.1 Minimal FP32 debug training loop passed!\n")
    return val_report


if __name__ == '__main__':
    run_debug_training()
