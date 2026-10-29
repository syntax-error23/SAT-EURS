"""
Independent Validation Script: Measures pure per-image held-out metrics
for DINOv2-L + SegFormer-B5 ensemble (zero taxonomy gating, zero consensus).
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, ".")
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_segformer_baseline import LeakFreeSegFormerDataset
from tools.train_dinov2_l import DINOv2L14Segmenter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

def _apply_rsuigm_batch(x_raw, wts, depths):
    B, C, H, W = x_raw.shape
    restored = []
    for i in range(B):
        xi = x_raw[i:i+1]
        wt_i = int(wts[i].item())
        d_i = int(depths[i].item())
        mat = rsuigm_table.get((wt_i, d_i), rsuigm_table[(1, 10)])
        feats = build_rsuigm_poly_features_gpu(xi)
        phys = torch.clamp(torch.matmul(feats, mat).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
        restored.append(phys)
    return torch.cat(restored, dim=0)

mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# Checkpoints
model_b5 = smp.DeepLabV3Plus(encoder_name="mit_b5", encoder_weights=None, classes=5, in_channels=3).to(device)
ckpt_b5 = torch.load("checkpoints/track1_high_capacity/exp_segformer_b5/best_model.pt", map_location="cpu", weights_only=False)
model_b5.load_state_dict(ckpt_b5["model_state_dict"])
model_b5.eval()

model_dino = DINOv2L14Segmenter(num_classes=5).to(device)
ckpt_dino = torch.load("checkpoints/track1_high_capacity/exp_dinov2_l/best_model.pt", map_location="cpu", weights_only=False)
model_dino.load_state_dict(ckpt_dino["model_state_dict"])
model_dino.eval()

val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
data_root = "data/cached_1024" if os.path.exists("data/cached_1024") else "data/train"

evaluator = SegmentationEvaluator(num_classes=5)

t0 = time.time()
with torch.no_grad():
    for scene in val_scenes:
        ds = LeakFreeSegFormerDataset(scenes=[scene], data_root=data_root, crop_size=512, is_train=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        for x_raw, y_mask, wts, depths in loader:
            x_raw = x_raw.to(device)
            y_mask = y_mask.to(device)
            x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
            x_norm = (x_rest - mean_im) / std_im

            B, C, H, W = x_norm.shape
            x_b5 = F.interpolate(x_norm, size=(1024, 1024), mode="bilinear", align_corners=False)
            x_dino = F.interpolate(x_norm, size=(1022, 1022), mode="bilinear", align_corners=False)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p_b5 = F.softmax(F.interpolate(model_b5(x_b5), size=(H, W), mode="bilinear", align_corners=False), dim=1)
                p_dino = F.softmax(F.interpolate(model_dino(x_dino), size=(H, W), mode="bilinear", align_corners=False), dim=1)

            # Pure Ensemble: 50% DINOv2-L + 50% SegFormer-B5
            p_ens = 0.5 * p_dino + 0.5 * p_b5
            pred_mask = torch.argmax(p_ens, dim=1)
            evaluator.update(pred_mask, y_mask)

m = evaluator.compute()
t1 = time.time()

print("=" * 76)
print("   MEASURED HELD-OUT SEGMENTATION METRICS (PURE PER-IMAGE ENSEMBLE)")
print("=" * 76)
print(f"Execution Time:        {t1-t0:.2f}s")
print(f"Q_seg:                 {m['Q_seg']:.4f}")
print(f"Official Q_seg:        {m.get('official_Q_seg', m['Q_seg']):.4f}")
print(f"mIoU:                  {m['mIoU']:.4f}")
print(f"Macro F1:              {m['macro_F1']:.4f}")
print(f"Coverage Penalty:      {m['coverage_penalty']:.4f}")
print(f"Soft Coverage Penalty: {m.get('soft_coverage_penalty', 1.0):.4f}")
print(f"Per-class IoU:         {m['per_class_IoU']}")
print(f"Per-class F1:          {m['per_class_F1']}")
print("=" * 76)

with open("checkpoints/track1_high_capacity/held_out_recomputed_metrics.json", "w") as fp:
    json.dump(m, fp, indent=2)
