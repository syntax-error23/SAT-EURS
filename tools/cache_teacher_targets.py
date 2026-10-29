"""
SAT-EURS Track 2 — Leakage-Free Teacher Target Caching Engine
Generates pre-softmax teacher targets for Student S0 (using Teacher T0) and S1 (using Teacher T1).
Outputs:
- Target tensors: [5, 192, 384] float16
- Manifest: reports/track2_readiness/TRACK2_DISTILLATION_MANIFEST.json
"""

import os, sys, time, json
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from tools.train_track1_screening import build_dinov2_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Target Caching] Device: {device}", flush=True)

mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

target_h, target_w = 192, 384

with open("splits/manifest_5fold.json") as f:
    manifest_5fold = json.load(f)

# Fold 0 and Fold 1 definitions
fold_configs = [
    {
        "fold": 0,
        "teacher_name": "DINOv2_Large_CP_F0",
        "ckpt_path": "checkpoints/track1_screening/DINOV2_CP_F0/best_model.pt",
        "train_scenes": manifest_5fold["folds"]["fold_0"]["train_scenes"],
        "out_dir": "data/cached_teacher_targets/fold_0"
    },
    {
        "fold": 1,
        "teacher_name": "DINOv2_Large_CP_F1",
        "ckpt_path": "checkpoints/track1_screening/DINOV2_CP_F1/best_model.pt",
        "train_scenes": manifest_5fold["folds"]["fold_1"]["train_scenes"],
        "out_dir": "data/cached_teacher_targets/fold_1"
    }
]

manifest_records = []

for cfg in fold_configs:
    fold = cfg["fold"]
    ckpt_path = cfg["ckpt_path"]
    out_dir = cfg["out_dir"]
    train_scenes = cfg["train_scenes"]
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Fold {fold}] Loading Teacher from {ckpt_path}...", flush=True)
    teacher = build_dinov2_model(num_classes=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt.get("model_state_dict", ckpt))
    teacher.eval()
    
    t0 = time.time()
    count = 0
    
    for sc in train_scenes:
        sc_out_dir = os.path.join(out_dir, sc)
        os.makedirs(sc_out_dir, exist_ok=True)
        deg_dir = f"data/cached_1024/degraded/{sc}"
        if not os.path.exists(deg_dir):
            continue
            
        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])
        # Sample 100 images per scene (every 2nd image)
        deg_files = deg_files[::2][:100]
        
        for f in deg_files:
            stem = os.path.splitext(f)[0]
            target_file = os.path.join(sc_out_dir, f"{stem}.pt")
            
            if os.path.exists(target_file):
                count += 1
                continue
                
            img_path = os.path.join(deg_dir, f)
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Preprocessing: resize to 518x518 for DINOv2
            rgb_518 = cv2.resize(rgb, (518, 518))
            x_th = torch.from_numpy(rgb_518.transpose(2, 0, 1)).unsqueeze(0).to(device)
            x_norm = (x_th - mean_im) / std_im
            
            # Horizontal-flip TTA
            x_norm_flip = torch.flip(x_norm, dims=[3])
            
            with torch.no_grad():
                l_orig = teacher(x_norm)
                l_flip = torch.flip(teacher(x_norm_flip), dims=[3])
                l_mean = 0.5 * (l_orig + l_flip)
                
                # Resize to target resolution [192, 384]
                l_target = F.interpolate(l_mean, size=(target_h, target_w), mode="bilinear", align_corners=False)
                l_f16 = l_target.squeeze(0).half().cpu() # [5, 192, 384] float16
                
            torch.save(l_f16, target_file)
            count += 1
            
            manifest_records.append({
                "fold": fold,
                "teacher_model": cfg["teacher_name"],
                "source_scene": sc,
                "source_image": f,
                "target_path": target_file,
                "tensor_shape": list(l_f16.shape),
                "dtype": "float16",
                "tta": "horizontal_flip",
                "input_resolution": [518, 518],
                "target_resolution": [target_h, target_w]
            })
            
    print(f"[Fold {fold}] Processed {count} targets in {time.time() - t0:.1f}s", flush=True)

# Save Manifest
manifest_out = "reports/track2_readiness/TRACK2_DISTILLATION_MANIFEST.json"
os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
with open(manifest_out, "w") as fp:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Pre-softmax teacher targets for Fold 0 and Fold 1 student distillation",
        "total_cached_targets": len(manifest_records),
        "target_resolution": [target_h, target_w],
        "precision": "float16",
        "records": manifest_records[:50] # sample 50 records in summary
    }, fp, indent=2)

print(f"\n[Complete] Generated {len(manifest_records)} teacher targets. Manifest saved to {manifest_out}")
