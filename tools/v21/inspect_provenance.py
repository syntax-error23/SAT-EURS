import torch
import os

print("--- Checkpoint & Table Provenance Audit ---")
for f in range(5):
    tb_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{f}.pt"
    if os.path.exists(tb_path):
        tb = torch.load(tb_path, map_location="cpu")
        if isinstance(tb, dict):
            print(f"Fold {f} Table B (dict): keys={list(tb.keys())}")
            for k, v in list(tb.items())[:2]:
                if hasattr(v, "shape"):
                    print(f"   key {k}: shape {v.shape}, dtype {v.dtype}")
                else:
                    print(f"   key {k}: {type(v)}")
        else:
            print(f"Fold {f} Table B: shape={tb.shape}, dtype={tb.dtype}")
    else:
        print(f"Fold {f} Table B: NOT FOUND at {tb_path}")

    rf_path = f"checkpoints/v18/fold_{f}/refiner_best.pt"
    if os.path.exists(rf_path):
        rf = torch.load(rf_path, map_location="cpu")
        prov = rf.get("provenance", {})
        print(f"Fold {f} Refiner: arch={prov.get('arch_name')}, fold={prov.get('fold')}")
        print(f"   train_scenes ({len(prov.get('train_scenes', []))}): {prov.get('train_scenes')}")
        print(f"   val_scenes ({len(prov.get('val_scenes', []))}): {prov.get('val_scenes')}")
        print(f"   metrics: {prov.get('metrics')}")
    else:
        print(f"Fold {f} Refiner: NOT FOUND at {rf_path}")
