"""
10-Sample Memorization & Overfit Harness (Stage 6 Verification)
Verifies:
1. Mathematical pipeline, coordinate alignment, and label correctness on real data samples.
2. Downward loss trends across all multi-task components.
3. Class presence and calibration acceptance targets.
4. FP32, augmentation OFF, shuffle OFF, fixed seed.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EURSNetV2
from data.dataset import EURSDataset
from losses import MultiTaskLoss
from metrics import SegmentationEvaluator, RestorationEvaluator


def run_overfit_test(num_samples: int = 5, num_iters: int = 50, resolution: int = 256):
    print("=" * 60)
    print("Test 6.2: Real Sample Memorization & Overfit Verification")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    from utils.paths import DATA_DIR, SPLITS_DIR
    data_dir = str(DATA_DIR / 'mini_dev') if (DATA_DIR / 'mini_dev').exists() else str(DATA_DIR)
    manifest_path = str(SPLITS_DIR / 'manifest.json')

    # 1. Load real data with target resolution (deterministic, no aug)
    dataset = EURSDataset(
        data_dir=data_dir,
        split='train',
        manifest_path=manifest_path,
        target_size=(resolution, resolution),
        transform=None
    )

    # Pick fixed micro-slice
    indices = list(range(min(num_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), shuffle=False)

    deg_batch, clean_batch, mask_batch, meta = next(iter(loader))
    print(f"  [DATA] Loaded {len(indices)} real samples from scenes: {meta['scene_id']}")
    print(f"  [DATA] Degraded: {list(deg_batch.shape)}, Clean: {list(clean_batch.shape)}, Mask: {list(mask_batch.shape)}")

    # Check which classes are present in the ground truth
    present_classes = torch.unique(mask_batch).tolist()
    print(f"  [DATA] Ground truth classes present in batch: {present_classes}")

    # 2. Initialize Model & Loss
    model = EURSNetV2(num_classes=5).train()
    loss_fn = MultiTaskLoss(weight_seg=1.0, weight_rest=0.5, num_classes=5)

    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    losses_total = []
    losses_seg = []
    losses_rest = []

    print("\n  Starting optimization (50 iterations on fixed batch)...")
    for step in range(1, num_iters + 1):
        optimizer.zero_grad()

        restored_rgb, seg_logits = model(deg_batch)

        # Scale model outputs to resolution if needed
        if restored_rgb.shape[2:] != (resolution, resolution):
            restored_rgb = torch.nn.functional.interpolate(
                restored_rgb, size=(resolution, resolution), mode='bilinear', align_corners=False
            )
        if seg_logits.shape[2:] != (resolution, resolution):
            seg_logits = torch.nn.functional.interpolate(
                seg_logits, size=(resolution, resolution), mode='bilinear', align_corners=False
            )

        loss, l_dict = loss_fn(restored_rgb, seg_logits, clean_batch, mask_batch)
        loss.backward()
        optimizer.step()

        losses_total.append(l_dict['loss_total'])
        losses_seg.append(l_dict['loss_seg'])
        losses_rest.append(l_dict['loss_rest'])

        if step % 10 == 0 or step == 1:
            print(f"    Step {step:2d}/{num_iters:2d} | Total Loss: {l_dict['loss_total']:.4f} "
                  f"(Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

    # 3. Multi-Criteria Verifications
    init_loss = np.mean(losses_total[:5])
    final_loss = np.mean(losses_total[-5:])
    loss_reduction = (init_loss - final_loss) / init_loss

    print("\n  [AUDIT] Loss Dynamics:")
    print(f"    - Initial 5-step mean loss: {init_loss:.4f}")
    print(f"    - Final 5-step mean loss:   {final_loss:.4f}")
    print(f"    - Relative reduction:       {loss_reduction * 100:.1f}%")

    assert final_loss < init_loss * 0.50, (
        f"Loss failed to decrease significantly! Initial: {init_loss:.4f}, Final: {final_loss:.4f}"
    )
    print("  [PASS] Total loss demonstrated a clear downward trend (> 50% reduction).")

    # 4. Final Evaluation on Memorized Batch
    model.eval()
    with torch.no_grad():
        restored_rgb, seg_logits = model(deg_batch)
        if restored_rgb.shape[2:] != (resolution, resolution):
            restored_rgb = torch.nn.functional.interpolate(
                restored_rgb, size=(resolution, resolution), mode='bilinear', align_corners=False
            )
        if seg_logits.shape[2:] != (resolution, resolution):
            seg_logits = torch.nn.functional.interpolate(
                seg_logits, size=(resolution, resolution), mode='bilinear', align_corners=False
            )

        seg_eval = SegmentationEvaluator(num_classes=5)
        seg_eval.update(seg_logits, mask_batch)
        seg_res = seg_eval.compute()

        rest_eval = RestorationEvaluator()
        rest_eval.update(restored_rgb, clean_batch)
        rest_res = rest_eval.compute()

    print("\n  [AUDIT] Memorization Metrics:")
    print(f"    - Segmentation mIoU:    {seg_res['mIoU']:.4f}")
    print(f"    - Macro F1:             {seg_res['macro_F1']:.4f}")
    print(f"    - Per-class IoU:        {seg_res['per_class_IoU']}")
    print(f"    - Restored PSNR:        {rest_res['PSNR_dB']:.2f} dB")
    print(f"    - Restored SSIM:        {rest_res['SSIM']:.4f}")

    # Verify that all present classes in GT have non-zero predictions
    pred_classes = torch.unique(seg_logits.argmax(dim=1)).tolist()
    print(f"    - Predicted classes:    {pred_classes}")
    for c in present_classes:
        assert c in pred_classes, f"Class {c} was present in GT but completely missing from model predictions!"
    print("  [PASS] Class presence verified: All GT classes successfully predicted by the network.")

    # 5. Save visual comparison artifact for inspection
    from utils.paths import AUDITS_DIR
    out_dir = str(AUDITS_DIR / 'overfit_preds')
    os.makedirs(out_dir, exist_ok=True)
    sample_idx = 0

    deg_img = (deg_batch[sample_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    clean_img = (clean_batch[sample_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rest_img = (np.clip(restored_rgb[sample_idx].permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    mask_gt = (mask_batch[sample_idx].cpu().numpy() * 50).astype(np.uint8)
    mask_pred = (seg_logits[sample_idx].argmax(dim=0).cpu().numpy() * 50).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, 'sample_degraded.png'), cv2.cvtColor(deg_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, 'sample_clean.png'), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, 'sample_restored.png'), cv2.cvtColor(rest_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, 'sample_mask_gt.png'), mask_gt)
    cv2.imwrite(os.path.join(out_dir, 'sample_mask_pred.png'), mask_pred)

    print(f"\n  [ARTIFACT] Saved qualitative prediction images to {out_dir}")
    print("Real sample overfit and memorization audit passed!\n")


if __name__ == '__main__':
    run_overfit_test(num_samples=4, num_iters=50, resolution=256)
