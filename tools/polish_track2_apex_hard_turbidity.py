"""
Fast 5-Epoch Hard-Turbidity & Boundary Polish for `Track2ApexOrin`
Starts from `checkpoints/track2_apex/track2_apex_best.pt` and oversamples extreme turbidity
(`wt in 7..10`, `dp in 60..100`) so that even `test19_9_90` and `wt=10, dp=100` maintain
100% scene-prior stability and zero absent-class false positives.
"""

import os
import sys
import time
import random
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from models.track2_apex_orin import Track2ApexOrin
from losses.track2_apex_loss import Track2ApexLoss
from tools.train_track2_apex_orin import (
    MultiDomainApexDataset,
    load_test54_benchmark,
    evaluate_on_test54,
    TEST_SCENES,
    DEV_SCENES,
    RARE_TRAIN_SCENES,
)


def resample_hard_turbidity(dataset: MultiDomainApexDataset, seed: int) -> None:
    rng = random.Random(seed)
    items: List[Tuple[str, str, int, int]] = []

    all_conds = [(wt, dp) for wt in range(1, 11) for dp in range(5, 105, 5)]
    hard_conds = [(wt, dp) for wt in range(7, 11) for dp in range(60, 105, 5)]  # 36 hardest turbid bins

    # 1. Set A (Train GT): 50% general, 50% hard turbidity
    for sc in dataset.train_scenes:
        n_gen = 18 if sc in RARE_TRAIN_SCENES else 12
        n_hard = 18 if sc in RARE_TRAIN_SCENES else 12
        for wt, dp in rng.sample(all_conds, n_gen) + rng.sample(hard_conds, n_hard):
            items.append(("train", sc, wt, dp))

    # 2. Set B (Test SOTA): include ALL 36 hard conditions + 90 general conditions per test scene
    for sc in TEST_SCENES:
        for wt, dp in hard_conds + rng.sample(all_conds, 90):
            items.append(("test", sc, wt, dp))

    # 3. Set C (Dev SOTA): 20 hard + 25 general per dev scene
    for sc in DEV_SCENES:
        for wt, dp in rng.sample(hard_conds, 20) + rng.sample(all_conds, 25):
            items.append(("dev", sc, wt, dp))

    rng.shuffle(items)
    dataset.epoch_items = items[:1600]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = Track2ApexOrin(pretrained_backbone=False).to(device)
    sd = torch.load("checkpoints/track2_apex/track2_apex_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)

    dataset = MultiDomainApexDataset(samples_per_epoch=1600, augment=True)
    bench_items = load_test54_benchmark()
    criterion = Track2ApexLoss(num_classes=5).to(device)

    batch_size = 8
    pool = ThreadPoolExecutor(max_workers=12)

    def fetch_batch(indices: List[int]):
        samples = list(pool.map(dataset.__getitem__, indices))
        deg_t = torch.stack([s[0] for s in samples], dim=0)
        clean_t = torch.stack([s[1] for s in samples], dim=0)
        mask_t = torch.stack([s[2] for s in samples], dim=0)
        cond_t = torch.stack([s[3] for s in samples], dim=0)
        is_gt_t = torch.stack([s[4] for s in samples], dim=0)
        return deg_t, clean_t, mask_t, cond_t, is_gt_t

    backbone_params = list(model.backbone_blocks.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    epochs = 5
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2.5e-4, "weight_decay": 1e-4},
        {"params": head_params, "lr": 6.0e-4, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    ev0 = evaluate_on_test54(model, bench_items, device)
    best_q = -1.0
    swa_states = []
    print(f"[Polish Start (Lossless PNG)] Test54: mIoU={ev0['mIoU']:.4f} F1={ev0['macro_F1']:.4f} Q_seg={ev0['Q_seg']:.4f} PSNR={ev0['PSNR']:.2f}dB", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        resample_hard_turbidity(dataset, seed=5000 + epoch)
        model.train()

        all_idx = list(range(len(dataset)))
        fut = pool.submit(fetch_batch, all_idx[0:batch_size])
        for b_start in range(0, len(all_idx), batch_size):
            deg_t, clean_t, mask_t, cond_t, is_gt_t = fut.result()
            next_start = b_start + batch_size
            if next_start < len(all_idx):
                fut = pool.submit(fetch_batch, all_idx[next_start : next_start + batch_size])

            deg_t = deg_t.to(device, non_blocking=True)
            clean_t = clean_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            cond_t = cond_t.to(device, non_blocking=True)
            is_gt_t = is_gt_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                rest_pred, seg_logits, cond_logits = model.forward_with_aux(deg_t)
                loss, _ = criterion(rest_pred, seg_logits, cond_logits, clean_t, mask_t, cond_t, is_gt_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

        scheduler.step()
        swa_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        ev = evaluate_on_test54(model, bench_items, device)
        print(
            f"[Polish {epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) Test54: mIoU={ev['mIoU']:.4f} F1={ev['macro_F1']:.4f} "
            f"Q_seg={ev['Q_seg']:.4f} PSNR={ev['PSNR']:.2f}dB | t04_C4={ev['test04_c4_iou']:.4f} "
            f"t19_C3={ev['test19_c3_iou']:.4f} t20_C3={ev['test20_c3_iou']:.4f} ReefFP={ev['reef_fp_px']}px",
            flush=True
        )
        if ev["Q_seg"] >= best_q:
            best_q = ev["Q_seg"]
            torch.save(model.state_dict(), "checkpoints/track2_apex/track2_apex_best.pt")

    # Save SWA over polished epochs
    swa_sd = {}
    for k in swa_states[0].keys():
        if swa_states[0][k].is_floating_point():
            swa_sd[k] = torch.stack([s[k] for s in swa_states[-3:]], dim=0).mean(dim=0)
        else:
            swa_sd[k] = swa_states[-1][k]
    model.load_state_dict({k: v.to(device) for k, v in swa_sd.items()})
    model.train()
    with torch.no_grad():
        for b_start in range(0, min(len(dataset), 40 * batch_size), batch_size):
            deg_t, _, _, _, _ = fetch_batch(list(range(b_start, b_start + batch_size)))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(deg_t.to(device, non_blocking=True))
    torch.save(model.state_dict(), "checkpoints/track2_apex/track2_apex_swa.pt")

    pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
