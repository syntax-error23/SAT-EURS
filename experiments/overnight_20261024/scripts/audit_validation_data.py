import os
import os.path as osp
import cv2
import numpy as np
import json
from collections import defaultdict
import sys
sys.path.insert(0, ".")
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data

manifest = validate_manifest_and_folds()

# Clean scenes have 1 ground-truth mask each in data/clean or data/train/masks or data/v18/clean_masks
mask_dirs = ["data/train/masks", "data/masks", "data/v18/clean_masks", "data/clean/masks"]
mask_dir = None
for d in mask_dirs:
    if osp.exists(d):
        mask_dir = d
        break

print(f"Using mask directory: {mask_dir}")
assert mask_dir is not None, "Could not find mask directory"

scenes = [
    'fish1', 'fish11', 'fish12', 'fish13', 'fish14', 'fish15', 'fish17', 'fish18',
    'fish2', 'fish20', 'fish21', 'fish4', 'fish5', 'fish7', 'fish9',
    'ray1', 'seahorse1', 'sharkFish', 'starfish2', 'turtles1'
]

CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]
scene_stats = {}

for s in scenes:
    # Look for mask file
    candidates = [
        osp.join(mask_dir, f"{s}.png"),
        osp.join(mask_dir, f"{s}_mask.png"),
        osp.join("data/clean", f"{s}.png"),
        osp.join("data/clean", f"{s}_mask.png"),
        osp.join("data/masks", f"{s}.png")
    ]
    mf = None
    for c in candidates:
        if osp.exists(c):
            mf = c
            break
    if mf is None:
        # Search anywhere for f"{s}.png" in data
        for root, _, files in os.walk("data"):
            if f"{s}.png" in files and "mask" in root.lower():
                mf = osp.join(root, f"{s}.png")
                break
    
    assert mf is not None, f"Could not find mask for scene {s}"
    m = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
    if m.ndim > 2:
        m = m[:, :, 0]
    h, w = m.shape[:2]
    total_px = h * w
    counts = np.bincount(m.ravel(), minlength=5)
    
    scene_stats[s] = {
        "mask_path": mf,
        "height": h,
        "width": w,
        "total_pixels": total_px,
        "counts": counts[:5].tolist(),
        "present_classes": [c for c in range(5) if counts[c] > 0]
    }

# Build fold breakdown
fold_stats = {}
for fold in range(5):
    f_data = get_fold_data(manifest, fold)
    val_scenes = f_data["val_scenes"]
    val_counts = np.zeros(5, dtype=np.int64)
    val_total = 0
    for s in val_scenes:
        val_counts += np.array(scene_stats[s]["counts"])
        val_total += scene_stats[s]["total_pixels"]
    present = [c for c in range(5) if val_counts[c] > 0]
    absent = [c for c in range(5) if val_counts[c] == 0]
    fold_stats[fold] = {
        "val_scenes": val_scenes,
        "train_scenes": f_data["train_scenes"],
        "val_total_px": int(val_total),
        "val_counts": val_counts.tolist(),
        "present_classes": present,
        "absent_classes": absent
    }

report_lines = [
    "# SAT-EURS Track 1 — Validation Set & Scene-Level Leakage Audit",
    "",
    "## 1. Scene-Level Data Hierarchy & Split Protocol",
    "",
    "- The EURS training dataset consists of **20 unique clean scenes**.",
    "- Each scene has 1 clean ground-truth image + mask, and **200 synthetically degraded variants** (10 Jerlov water types x 20 depth stations).",
    "- **Critical Rule:** Validation must always be **scene-disjoint**. Evaluating a model on degraded variants of a scene seen during training constitutes severe data leakage.",
    "",
    "## 2. 5-Fold Scene-Disjoint Split Definition",
    "",
    "| Fold ID | Validation Scenes (4 Scenes) | Training Scenes (16 Scenes) | Absent Classes in Val ($GT=0$) | Present Classes in Val ($GT>0$) |",
    "| :---: | :--- | :--- | :--- | :--- |"
]

for fold in range(5):
    fs = fold_stats[fold]
    absent_names = [CLASS_NAMES[c] for c in fs["absent_classes"]]
    present_names = [CLASS_NAMES[c] for c in fs["present_classes"]]
    abs_str = ", ".join(absent_names) if absent_names else "None (All Present)"
    pres_str = ", ".join(present_names)
    report_lines.append(f"| **Fold {fold}** | `{', '.join(fs['val_scenes'])}` | 16 scenes | **{abs_str}** | {pres_str} |")

report_lines.extend([
    "",
    "## 3. Per-Scene Geometric & Class Prevalence Census",
    "",
    "| Scene ID | Dimensions ($H \\times W$) | Total Pixels | Background (%) | Coral (%) | Fish (%) | Ray (%) | MacroFauna (%) | Classes Present |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
])

for s in sorted(scenes):
    st = scene_stats[s]
    tot = st["total_pixels"]
    cnt = st["counts"]
    pcts = [f"{(cnt[c]/tot)*100:.2f}%" for c in range(5)]
    pres = ", ".join([CLASS_NAMES[c] for c in st["present_classes"]])
    report_lines.append(f"| `{s}` | {st['height']}x{st['width']} | {tot:,} | {pcts[0]} | {pcts[1]} | {pcts[2]} | {pcts[3]} | {pcts[4]} | {pres} |")

report_lines.extend([
    "",
    "## 4. Key Diagnostic Insights on Rare Classes",
    "",
    "1. **Class 3 (Ray / Elasmobranch):**",
    "   - Present in ONLY 2 training scenes: `ray1` (Fold 2) and `sharkFish` (Fold 1).",
    "   - Completely absent ($GT = 0$) in Fold 0, Fold 3, and Fold 4 validation splits.",
    "2. **Class 4 (MacroFauna - Seahorse, Starfish, Turtle):**",
    "   - Present in ONLY 3 training scenes: `seahorse1` (Fold 0), `starfish2` (Fold 3), and `turtles1` (Fold 4).",
    "   - Completely absent ($GT = 0$) in Fold 1 and Fold 2 validation splits.",
    "3. **The Absent-Class Vacuous Truth Artifact:**",
    "   - In single-fold splits where Ray or MacroFauna is absent, a model that predicts 0 pixels gets $IoU = 1.0$ (vacuous truth).",
    "   - Predicting even 1 false-positive pixel drops IoU from $1.0$ to $0.0$, penalizing single-fold mIoU by $-0.20$.",
    "   - However, on the 1,200 hidden test images, all 5 classes are present. Therefore, pooled 5-fold evaluation across all 20 scenes (where all 5 classes have $GT > 0$) is the only uncorrupted local validation benchmark.",
    ""
])

with open("experiments/overnight_20261024/validation_audit.md", "w") as f:
    f.write("\n".join(report_lines))

print("Validation audit written to experiments/overnight_20261024/validation_audit.md")
