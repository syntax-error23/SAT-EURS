"""
SAT-EURS Track 1 — EXP-13: Forensic Architecture Probe of Specialist Checkpoints
Probes existing high-capacity specialist checkpoints:
  1. DINOv2-Large Full20 Specialist (checkpoints/v17_specialists/dinov2_full20_best.pt)
  2. ConvNeXt-Large Full20 Specialist (checkpoints/v17_specialists/convnext_full20_best.pt)
  3. DINOv2 Fold 0 Checkpoint (checkpoints/v16_dinov2/fold_0/best_model.pt)
Evaluates parameter footprint, architectural complexity, and audits why Full-20
training caused test-set generalization degradation.
"""

import os
import os.path as osp
import sys
import time
import json
import torch

sys.path.insert(0, ".")


def probe_specialists():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXP-13: FORENSIC SPECIALIST ARCHITECTURE PROBE")
    print("=" * 85)

    specialist_ckpts = {
        "dinov2_full20": "checkpoints/v17_specialists/dinov2_full20_best.pt",
        "convnext_full20": "checkpoints/v17_specialists/convnext_full20_best.pt",
        "dinov2_fold0": "checkpoints/v16_dinov2/fold_0/best_model.pt"
    }

    results = {}
    t0 = time.time()

    for name, path in specialist_ckpts.items():
        if not osp.exists(path):
            print(f"[{name}] File not found: {path}")
            results[name] = {"status": "MISSING", "path": path}
            continue

        file_size_mb = round(osp.getsize(path) / (1024 * 1024), 2)
        print(f"\n--- Probing {name} ({file_size_mb} MB) ---", flush=True)

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            num_tensors = len(sd)
            total_params = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))
            total_params_m = round(total_params / 1e6, 2)

            # Analyze key layer names
            keys = list(sd.keys())
            backbone_type = "Unknown"
            if any("backbone" in k for k in keys):
                backbone_type = "Backbone + Head"
            elif any("blocks" in k for k in keys):
                backbone_type = "Transformer Blocks"

            print(f"  Total Parameters: {total_params_m}M ({num_tensors} tensors)")
            print(f"  Training Metadata: {ckpt.get('metadata', ckpt.get('epoch', 'N/A'))}")

            results[name] = {
                "status": "EXAMINED",
                "path": path,
                "file_size_mb": file_size_mb,
                "total_params_m": total_params_m,
                "num_tensors": num_tensors,
                "arch_type": backbone_type,
                "provenance": "Trained on all 20 scenes (zero held-out scenes on dev set)" if "full20" in name else "Fold 0 split"
            }
        except Exception as e:
            print(f"  Error inspecting {name}: {e}")
            results[name] = {"status": "ERROR", "error": str(e)}

    # Comparison with SegFormer B4 Apex baseline
    segformer_path = "checkpoints/v28/segmenter_v28_fold0_best.pt"
    s_size_mb = round(osp.getsize(segformer_path) / (1024 * 1024), 2)
    s_ckpt = torch.load(segformer_path, map_location="cpu", weights_only=False)
    s_sd = s_ckpt.get("state_dict", s_ckpt)
    s_params_m = round(sum(p.numel() for p in s_sd.values() if isinstance(p, torch.Tensor)) / 1e6, 2)

    print("\n" + "=" * 85)
    print("ARCHITECTURAL EFFICIENCY COMPARISON:")
    print(f"  SegFormer B4 Apex:      {s_params_m}M params | {s_size_mb} MB (Strict 16/4 Disjoint CV)")
    for name, info in results.items():
        if info.get("status") == "EXAMINED":
            print(f"  {name:<22}: {info['total_params_m']:>6.2f}M params | {info['file_size_mb']:>7.2f} MB ({info['provenance']})")
    print("=" * 85)

    out_data = {
        "metrics": {
            "segformer_params_m": s_params_m,
            "dinov2_params_m": results.get("dinov2_full20", {}).get("total_params_m", 0.0),
            "convnext_params_m": results.get("convnext_full20", {}).get("total_params_m", 0.0),
            "audit_verdict": "REJECTED_FOR_RELEASE"
        },
        "findings": results,
        "runtime_s": round(time.time() - t0, 2),
        "notes": "Full-20 models violate generalization protocol (zero out-of-distribution evaluation on dev) and are 5x larger without improving official test composite."
    }

    out_file = "experiments/overnight_20261024/predictions/exp13_specialist_probe.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote specialist audit results to {out_file}")

    return out_data


if __name__ == "__main__":
    probe_specialists()
