"""
Phase 1 & 1B Audit Script:
1. Exact Pixel & Connected Component Instance Counts per Scene and per Fold.
2. Enumeration of all 10 possible fold pairs for S_folds and C_folds.
3. Deterministic selection of S_folds and C_folds based on positive support for all 5 classes and rare-class balance.
"""

import os
import os.path as osp
import json
import itertools
import cv2
import numpy as np

def run_audit():
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    masks_dir = "data/cached_1024/masks"
    scenes = sorted([osp.splitext(f)[0] for f in os.listdir(masks_dir) if f.endswith(".png")])

    scene_stats = {}
    for sc in scenes:
        m = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
        counts = np.bincount(m.flatten(), minlength=5)
        instances = [0] * 5
        for c in range(1, 5):
            bin_m = (m == c).astype(np.uint8)
            if bin_m.sum() > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
                valid = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 10]
                instances[c] = len(valid)
        scene_stats[sc] = {
            "pixels": counts.tolist(),
            "instances": instances
        }

    print("=" * 80)
    print("SCENE LEVEL STATS (Classes 0: Background, 1: Coral, 2: Fish, 3: Elasmobranch, 4: Other)")
    print("=" * 80)
    for sc in scenes:
        p_str = ", ".join(f"C{c}:{scene_stats[sc]['pixels'][c]:,}" for c in range(5))
        i_str = ", ".join(f"C{c}:{scene_stats[sc]['instances'][c]}" for c in range(1, 5))
        print(f"{sc:14s} | Pixels: [{p_str}] | Instances: [{i_str}]")

    print("\n" + "=" * 80)
    print("HELD-OUT VALIDATION SCENE STATS PER FOLD")
    print("=" * 80)
    fold_val_stats = {}
    for f_idx in range(5):
        val_scs = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        px = np.zeros(5, dtype=np.int64)
        inst = np.zeros(5, dtype=np.int64)
        for sc in val_scs:
            px += np.array(scene_stats[sc]["pixels"])
            inst += np.array(scene_stats[sc]["instances"])
        fold_val_stats[f_idx] = {
            "pixels": px.tolist(),
            "instances": inst.tolist(),
            "scenes": val_scs
        }
        p_str = ", ".join(f"C{c}:{px[c]:,}" for c in range(5))
        i_str = ", ".join(f"C{c}:{inst[c]}" for c in range(1, 5))
        print(f"Fold {f_idx} ({val_scs}):\n  Pixels: [{p_str}]\n  Instances: [{i_str}]")

    print("\n" + "=" * 80)
    print("ENUMERATION OF ALL 10 POSSIBLE FOLD PAIRS (S_folds = 2 folds, C_folds = 3 folds)")
    print("=" * 80)
    all_folds = set(range(5))
    eligible_pairs = []

    for pair in itertools.combinations(range(5), 2):
        s_set = set(pair)
        c_set = all_folds - s_set

        s_px = np.sum([fold_val_stats[f]["pixels"] for f in s_set], axis=0)
        c_px = np.sum([fold_val_stats[f]["pixels"] for f in c_set], axis=0)

        s_inst = np.sum([fold_val_stats[f]["instances"] for f in s_set], axis=0)
        c_inst = np.sum([fold_val_stats[f]["instances"] for f in c_set], axis=0)

        s_pos = bool(np.all(s_px > 0))
        c_pos = bool(np.all(c_px > 0))

        status = "ELIGIBLE" if (s_pos and c_pos) else "INELIGIBLE"
        print(f"\nCandidate Pair S_folds={pair} vs C_folds={sorted(list(c_set))}:")
        print(f"  Status: {status} (S_has_all_5={s_pos}, C_has_all_5={c_pos})")
        print(f"  S_folds (val px): {[int(x) for x in s_px]}, inst: {[int(x) for x in s_inst]}")
        print(f"  C_folds (val px): {[int(x) for x in c_px]}, inst: {[int(x) for x in c_inst]}")

        if s_pos and c_pos:
            # Deterministic balance score on rare classes 3 and 4
            ratio_3 = float(min(s_px[3], c_px[3]) / max(s_px[3], c_px[3]))
            ratio_4 = float(min(s_px[4], c_px[4]) / max(s_px[4], c_px[4]))
            inst_ratio_3 = float(min(s_inst[3], c_inst[3]) / max(s_inst[3], c_inst[3]))
            inst_ratio_4 = float(min(s_inst[4], c_inst[4]) / max(s_inst[4], c_inst[4]))
            balance_score = ratio_3 * ratio_4
            eligible_pairs.append({
                "pair": pair,
                "c_folds": sorted(list(c_set)),
                "balance_score": balance_score,
                "pixel_ratio_3": ratio_3,
                "pixel_ratio_4": ratio_4,
                "inst_ratio_3": inst_ratio_3,
                "inst_ratio_4": inst_ratio_4,
                "s_pixels": s_px.tolist(),
                "c_pixels": c_px.tolist(),
                "s_instances": s_inst.tolist(),
                "c_instances": c_inst.tolist()
            })

    print("\n" + "=" * 80)
    print("RANKING OF ELIGIBLE PAIRS BY RARE-CLASS BALANCE SCORE:")
    print("=" * 80)
    ranked = sorted(eligible_pairs, key=lambda x: -x["balance_score"])
    for idx, ep in enumerate(ranked):
        print(f"Rank {idx+1}: S_folds={ep['pair']} | C_folds={ep['c_folds']} | Balance Score = {ep['balance_score']:.6f} (R3={ep['pixel_ratio_3']:.4f}, R4={ep['pixel_ratio_4']:.4f}, InstR3={ep['inst_ratio_3']:.4f}, InstR4={ep['inst_ratio_4']:.4f})")

    selected = ranked[0]
    print("\n" + "*" * 80)
    print(f"DETERMINISTICALLY SELECTED S_folds: {selected['pair']}")
    print(f"DETERMINISTICALLY SELECTED C_folds: {selected['c_folds']}")
    print(f"Selection Reason: Achieves strictly positive ground-truth support for all 5 classes in both groups, and maximizes the rare-class pixel balance score ({selected['balance_score']:.6f}).")
    print("*" * 80)

    # Save machine-readable manifest
    os.makedirs("reports/track1_research/v15", exist_ok=True)
    out_json = "reports/track1_research/v15/phase1b_fold_split_selection.json"
    with open(out_json, "w") as fp:
        json.dump({
            "selected_S_folds": list(selected["pair"]),
            "selected_C_folds": selected["c_folds"],
            "eligible_pairs_count": len(ranked),
            "ranked_pairs": ranked,
            "scene_stats": scene_stats,
            "fold_val_stats": fold_val_stats
        }, fp, indent=2)
    print(f"Saved Phase 1B selection to {out_json}")

if __name__ == "__main__":
    run_audit()
