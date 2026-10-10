"""
Isolated ONNX Exporter & Parity Validator for Pilot Epoch 2 Checkpoint
Target: MaCVi WACV 2027 Track 2 (Jetson AGX Orin)
Contract:
  Opset: 18 (Static dimensions)
  Input:  'input'          [1, 3, 1024, 1024] float32 [0.0, 1.0]
  Output: 'restored_image' [1, 3, 1024, 1024] float32 [0.0, 1.0]
  Output: 'seg_logits'     [1, 5, 1024, 1024] float32
"""

import os
import sys
import json
import hashlib
import time

sys.path.insert(0, os.path.abspath('.'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
import onnx
import onnxruntime as ort

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single
from deploy.check_flops import profile_complexity


def compute_file_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    print("=" * 80)
    print("    ISOLATED ONNX EXPORT & NUMERICAL PARITY VALIDATION (PILOT EPOCH 2)")
    print("=" * 80)

    ckpt_path = "experiments/exp_pilot_restricted_restoration/pilot_restricted_ep2.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    out_dir = "deploy/exports/pilot_epoch_2"
    os.makedirs(out_dir, exist_ok=True)
    out_onnx_path = os.path.join(out_dir, "eurs_net_v2_pilot_ep2.onnx")

    # 1. Instantiate PyTorch Model
    model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Epoch 2 PyTorch model with {total_params:,} parameters.")
    assert total_params <= 1_000_000, f"Budget exceeded: {total_params} > 1,000,000"

    # Profile complexity
    _, gflops = profile_complexity(resolution=1024)
    print(f"Computational complexity: {gflops:.3f} GFLOPs (Budget: <= 12.0 GFLOPs)")
    assert gflops <= 12.0, f"Complexity budget exceeded: {gflops} > 12.0"

    # 2. Export to ONNX (Static Contract, Opset 18)
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["restored_image", "seg_logits"]

    print(f"Exporting to isolated path: {out_onnx_path} (Opset 18)...")
    torch.onnx.export(
        model,
        dummy_input,
        out_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # Strictly static for Jetson AGX Orin TensorRT engine
        dynamo=False
    )
    print("Export complete.")

    # 3. ONNX Model Verification
    onnx_model = onnx.load(out_onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_sha256 = compute_file_sha256(out_onnx_path)
    onnx_size_bytes = os.path.getsize(out_onnx_path)
    print(f"ONNX Model Checked. Size: {onnx_size_bytes:,} bytes ({onnx_size_bytes/(1024*1024):.2f} MB)")
    print(f"ONNX File SHA-256: {onnx_sha256}")

    # 4. Initialize ONNX Runtime Session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(out_onnx_path, session_options, providers=['CPUExecutionProvider'])

    # Verify input/output signatures
    in_meta = session.get_inputs()[0]
    out_metas = {o.name: o for o in session.get_outputs()}
    assert in_meta.name == "input" and in_meta.shape == [1, 3, 1024, 1024]
    assert "restored_image" in out_metas and out_metas["restored_image"].shape == [1, 3, 1024, 1024]
    assert "seg_logits" in out_metas and out_metas["seg_logits"].shape == [1, 5, 1024, 1024]
    print("Inference signatures verified: Static input and dual outputs strictly conform.")

    # 5. Numerical Agreement Across Representative Real Images (All 4 Scenes)
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)
    
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    # Select 5 representative images from each of the 4 scenes (total 20 images)
    sample_indices = {}
    for s in scenes:
        sample_indices[s] = [i for i, smp in enumerate(val_ds.samples) if smp['scene_id'] == s][:5]

    scene_parity_results = {}
    all_max_diffs_rest = []
    all_max_diffs_seg = []

    print("\n--- EVALUATING NUMERICAL PARITY (PYTORCH vs ONNX RUNTIME) ---")
    for s in scenes:
        print(f"Evaluating scene: {s} (5 representative images)...")
        scene_diffs_rest = []
        scene_diffs_seg = []
        mask_match_rates = []
        pt_psnrs = []
        ort_psnrs = []
        pt_clips = []
        ort_clips = []

        for idx in sample_indices[s]:
            deg, clean, _, _ = val_ds[idx]
            deg_t = deg.unsqueeze(0)
            clean_np = clean.permute(1, 2, 0).numpy()

            # PyTorch forward
            with torch.no_grad():
                pt_rest, pt_seg = model(deg_t)
            pt_rest_np = pt_rest.squeeze(0).permute(1, 2, 0).numpy()
            pt_seg_np = pt_seg.squeeze(0).numpy()
            pt_mask = np.argmax(pt_seg_np, axis=0)

            # ONNX Runtime forward
            ort_outs = session.run(["restored_image", "seg_logits"], {"input": deg_t.numpy()})
            ort_rest_np = ort_outs[0].squeeze(0).transpose(1, 2, 0)
            ort_seg_np = ort_outs[1].squeeze(0)
            ort_mask = np.argmax(ort_seg_np, axis=0)

            # Differences
            d_rest = float(np.max(np.abs(pt_rest_np - ort_rest_np)))
            d_seg = float(np.max(np.abs(pt_seg_np - ort_seg_np)))
            mask_match = float(np.mean(pt_mask == ort_mask)) * 100.0

            scene_diffs_rest.append(d_rest)
            scene_diffs_seg.append(d_seg)
            mask_match_rates.append(mask_match)

            p_pt, _ = compute_psnr_and_norm(pt_rest_np, clean_np)
            p_ort, _ = compute_psnr_and_norm(ort_rest_np, clean_np)
            c_pt = float(np.mean((pt_rest_np == 0.0) | (pt_rest_np == 1.0))) * 100.0
            c_ort = float(np.mean((ort_rest_np == 0.0) | (ort_rest_np == 1.0))) * 100.0

            pt_psnrs.append(p_pt)
            ort_psnrs.append(p_ort)
            pt_clips.append(c_pt)
            ort_clips.append(c_ort)

        scene_parity_results[s] = {
            'max_diff_restored': float(np.max(scene_diffs_rest)),
            'mean_diff_restored': float(np.mean(scene_diffs_rest)),
            'max_diff_seg_logits': float(np.max(scene_diffs_seg)),
            'mean_diff_seg_logits': float(np.mean(scene_diffs_seg)),
            'mask_exact_match_pct': float(np.mean(mask_match_rates)),
            'pt_psnr_mean': float(np.mean(pt_psnrs)),
            'ort_psnr_mean': float(np.mean(ort_psnrs)),
            'psnr_discrepancy': float(abs(np.mean(pt_psnrs) - np.mean(ort_psnrs))),
            'pt_clip_pct_mean': float(np.mean(pt_clips)),
            'ort_clip_pct_mean': float(np.mean(ort_clips)),
            'clip_discrepancy': float(abs(np.mean(pt_clips) - np.mean(ort_clips))),
            'parity_passed': bool(np.max(scene_diffs_rest) < 1e-4 and np.max(scene_diffs_seg) < 1e-4)
        }

        all_max_diffs_rest.append(np.max(scene_diffs_rest))
        all_max_diffs_seg.append(np.max(scene_diffs_seg))
        print(f"  [{s:10s}] Max Delta Restored: {np.max(scene_diffs_rest):.2e} | Max Delta Seg: {np.max(scene_diffs_seg):.2e} | Mask Match: {np.mean(mask_match_rates):.4f}%")

    global_max_diff_rest = float(np.max(all_max_diffs_rest))
    global_max_diff_seg = float(np.max(all_max_diffs_seg))
    all_parity_passed = bool(global_max_diff_rest < 1e-4 and global_max_diff_seg < 1e-4)

    print("\n" + "=" * 80)
    print("                    PARITY AUDIT SUMMARY")
    print("=" * 80)
    print(f"Global Max Restored RGB Delta:   {global_max_diff_rest:.2e} (Threshold: < 1.00e-04) -> {'PASS' if global_max_diff_rest < 1e-4 else 'FAIL'}")
    print(f"Global Max Seg Logits Delta:     {global_max_diff_seg:.2e} (Threshold: < 1.00e-04) -> {'PASS' if global_max_diff_seg < 1e-4 else 'FAIL'}")
    print(f"Overall Numerical Parity Status: {'PASS' if all_parity_passed else 'FAIL'}")
    print("=" * 80)

    # 6. Save Parity Report
    report = {
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': compute_file_sha256(ckpt_path),
        'onnx_output_path': out_onnx_path,
        'onnx_sha256': onnx_sha256,
        'onnx_size_bytes': onnx_size_bytes,
        'opset_version': 18,
        'total_params': total_params,
        'gflops': gflops,
        'input_signature': {'name': 'input', 'shape': [1, 3, 1024, 1024], 'dtype': 'float32'},
        'output_signatures': {
            'restored_image': {'shape': [1, 3, 1024, 1024], 'dtype': 'float32'},
            'seg_logits': {'shape': [1, 5, 1024, 1024], 'dtype': 'float32'}
        },
        'tolerance_threshold': 1e-4,
        'global_max_diff_restored': global_max_diff_rest,
        'global_max_diff_seg_logits': global_max_diff_seg,
        'parity_passed': all_parity_passed,
        'per_scene_parity': scene_parity_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    report_path = "audits/final_independent_pilot_audit/onnx_export_parity_ep2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Parity report written to: {report_path}")

if __name__ == '__main__':
    main()
