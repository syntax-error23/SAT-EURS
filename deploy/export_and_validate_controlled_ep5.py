"""
Deployment Readiness & Static ONNX Export for Selected Epoch 5 Champion Checkpoint
Target: MaCVi @ WACV 2027 Track 2 (Embedded Jetson AGX Orin)
Checkpoint: experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt

Contract Enforced:
  Opset: 18 (Static dimensions)
  Input:  'input'          [1, 3, 1024, 1024] float32 in [0.0, 1.0]
  Output: 'restored_image' [1, 3, 1024, 1024] float32 in [0.0, 1.0]
  Output: 'seg_logits'     [1, 5, 1024, 1024] float32 (raw logits, 5 classes)
  Single self-contained file with embedded weights (no external .onnx.data)
  Parameters <= 1,000,000
  GFLOPs <= 12.0
"""

import os
import sys
import json
import hashlib
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath('.'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import onnx
import onnxruntime as ort

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import (
    RestorationEvaluator,
    compute_psnr_and_norm,
    compute_ssim_single,
    compute_sharpness_ratio
)
from metrics.nr_metric import NonReferenceQualityEvaluator
from metrics.composite_score import compute_composite_score
from deploy.check_flops import profile_complexity


def compute_file_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_single_image_metrics(args: Tuple[np.ndarray, np.ndarray, str]) -> Dict[str, Any]:
    r_np, c_np, sid = args
    p_val, p_norm = compute_psnr_and_norm(r_np, c_np)
    s_val = compute_ssim_single(r_np, c_np)
    sh_val = compute_sharpness_ratio(r_np, c_np)
    
    cl_val = float(np.mean((r_np == 0.0) | (r_np == 1.0))) * 100.0
    cl_0 = float(np.mean(r_np == 0.0)) * 100.0
    cl_1 = float(np.mean(r_np == 1.0)) * 100.0
    cl_r = float(np.mean((r_np[:, :, 0] == 0.0) | (r_np[:, :, 0] == 1.0))) * 100.0
    cl_g = float(np.mean((r_np[:, :, 1] == 0.0) | (r_np[:, :, 1] == 1.0))) * 100.0
    cl_b = float(np.mean((r_np[:, :, 2] == 0.0) | (r_np[:, :, 2] == 1.0))) * 100.0
    
    return {
        'sid': sid,
        'psnr': p_val,
        'psnr_norm': p_norm,
        'ssim': s_val,
        'sharpness': sh_val,
        'clip_pct': cl_val,
        'clip_0': cl_0,
        'clip_1': cl_1,
        'clip_r': cl_r,
        'clip_g': cl_g,
        'clip_b': cl_b
    }


def evaluate_full_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    class_names = ['Background', 'Coral', 'Fish', 'Shark', 'Other']

    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()
    nr_eval = NonReferenceQualityEvaluator()

    tasks = []
    with torch.no_grad():
        for deg, clean, mask, meta in loader:
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            restored, seg_logits = model(deg)

            seg_eval.update(seg_logits, mask)
            nr_eval.update(restored)

            b_size = deg.shape[0]
            rest_cpu = restored.permute(0, 2, 3, 1).cpu().numpy()
            clean_cpu = clean.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(b_size):
                tasks.append((rest_cpu[i], clean_cpu[i], meta['scene_id'][i]))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compute_single_image_metrics, tasks))

    scene_data = {s: {'psnr': [], 'ssim': [], 'sharp': [], 'clip_pct': [], 'clip_r': [], 'clip_g': [], 'clip_b': []} for s in scenes}
    all_clip_pcts = []
    all_clip_0 = []
    all_clip_1 = []
    all_clip_r = []
    all_clip_g = []
    all_clip_b = []

    for r in results:
        sid = r['sid']
        rest_eval.psnr_list.append(r['psnr'])
        rest_eval.psnr_norm_list.append(r['psnr_norm'])
        rest_eval.ssim_list.append(r['ssim'])
        rest_eval.sharpness_list.append(r['sharpness'])

        all_clip_pcts.append(r['clip_pct'])
        all_clip_0.append(r['clip_0'])
        all_clip_1.append(r['clip_1'])
        all_clip_r.append(r['clip_r'])
        all_clip_g.append(r['clip_g'])
        all_clip_b.append(r['clip_b'])
        
        if sid in scene_data:
            scene_data[sid]['psnr'].append(r['psnr'])
            scene_data[sid]['ssim'].append(r['ssim'])
            scene_data[sid]['sharp'].append(r['sharpness'])
            scene_data[sid]['clip_pct'].append(r['clip_pct'])
            scene_data[sid]['clip_r'].append(r['clip_r'])
            scene_data[sid]['clip_g'].append(r['clip_g'])
            scene_data[sid]['clip_b'].append(r['clip_b'])

    seg_res = seg_eval.compute()
    rest_res = rest_eval.compute()
    nr_res = nr_eval.compute()

    comp_surrogate = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], nr_res['Q_NR'])
    comp_standard = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], 0.50)

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            'iou': float(seg_res['per_class_IoU'][idx]),
            'f1': float(seg_res['per_class_F1'][idx])
        }

    per_scene = {}
    for s in scenes:
        count = len(scene_data[s]['psnr'])
        per_scene[s] = {
            'count': count,
            'psnr_mean': float(np.mean(scene_data[s]['psnr'])),
            'ssim_mean': float(np.mean(scene_data[s]['ssim'])),
            'sharp_mean': float(np.mean(scene_data[s]['sharp'])),
            'clip_pct_mean': float(np.mean(scene_data[s]['clip_pct'])),
            'clip_pct_max': float(np.max(scene_data[s]['clip_pct'])),
            'clip_r_mean': float(np.mean(scene_data[s]['clip_r'])),
            'clip_g_mean': float(np.mean(scene_data[s]['clip_g'])),
            'clip_b_mean': float(np.mean(scene_data[s]['clip_b'])),
            'clip_guardrail_pass': bool(np.mean(scene_data[s]['clip_pct']) < 5.0)
        }

    aggregate_clip_pct = float(np.mean(all_clip_pcts))

    return {
        'total_images': len(all_clip_pcts),
        'mIoU': seg_res['mIoU'],
        'macro_F1': seg_res['macro_F1'],
        'shark_iou': per_class['Shark']['iou'],
        'per_class': per_class,
        'coverage_penalty': seg_res['coverage_penalty'],
        'Q_seg': seg_res['Q_seg'],
        'PSNR_dB': rest_res['PSNR_dB'],
        'PSNR_norm': rest_res['PSNR_norm'],
        'SSIM': rest_res['SSIM'],
        'sharpness_ratio': rest_res['sharpness_ratio'],
        'Q_rest': rest_res['Q_rest'],
        'clipping_aggregate_pct': round(aggregate_clip_pct, 4),
        'clipping_low_0_pct': round(float(np.mean(all_clip_0)), 4),
        'clipping_high_1_pct': round(float(np.mean(all_clip_1)), 4),
        'clipping_channels_pct': {
            'R': round(float(np.mean(all_clip_r)), 4),
            'G': round(float(np.mean(all_clip_g)), 4),
            'B': round(float(np.mean(all_clip_b)), 4)
        },
        'clipping_pct_max': round(float(np.max(all_clip_pcts)), 4),
        'clipping_guardrail_pass': bool(aggregate_clip_pct < 5.0),
        'per_scene': per_scene,
        'Q_NR_surrogate': nr_res['Q_NR'],
        'composite_surrogate': comp_surrogate['Composite_Score'],
        'composite_standard_0_50': comp_standard['Composite_Score']
    }


def main():
    print("=" * 85)
    print("    EURS-NET V2 DEPLOYMENT READINESS & STATIC ONNX EXPORT AUDIT")
    print("    Candidate Checkpoint: Epoch 5 (best_controlled_checkpoint.pt)")
    print("    Timestamp: " + time.strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 85)

    ckpt_path = "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt"
    ep5_path = "experiments/exp_controlled_restoration_v2/checkpoints/controlled_ep5.pt"
    base_path = "checkpoints/run2_50ep/best_model.pt"

    assert os.path.exists(ckpt_path), f"Candidate checkpoint not found: {ckpt_path}"
    assert os.path.exists(ep5_path), f"Epoch 5 checkpoint not found: {ep5_path}"
    assert os.path.exists(base_path), f"Authoritative baseline not found: {base_path}"

    # Gate 1: Checkpoint Provenance & SHA-256 Hash Verification
    print("\n--- GATE 1: CHECKPOINT PROVENANCE & BITWISE INVARIANCE ---")
    cand_sha256 = compute_file_sha256(ckpt_path)
    cand_size = os.path.getsize(ckpt_path)
    expected_cand_sha = "444675aa4b6c25f686b8f0c30c9456b269b1c56d41c3d15d2bb4dd2e4229bf18"
    assert cand_sha256 == expected_cand_sha, f"SHA-256 mismatch! Expected {expected_cand_sha}, got {cand_sha256}"
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  SHA-256:    {cand_sha256} [VERIFIED EXACT]")
    print(f"  Size:       {cand_size:,} bytes")

    cand_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ep5_ckpt = torch.load(ep5_path, map_location='cpu', weights_only=False)
    base_ckpt = torch.load(base_path, map_location='cpu', weights_only=False)

    cand_sd = cand_ckpt['model_state_dict']
    ep5_sd = ep5_ckpt['model_state_dict']
    base_sd = base_ckpt['model_state_dict']

    assert cand_ckpt.get('epoch') == 5, f"Saved epoch is {cand_ckpt.get('epoch')}, expected 5!"
    assert all(torch.equal(cand_sd[k], ep5_sd[k]) for k in cand_sd), "Candidate weights not identical to Epoch 5!"
    print(f"  Epoch Tag:  Epoch {cand_ckpt.get('epoch')} [CONFIRMED]")
    print(f"  Equality:   100% bitwise identical to controlled_ep5.pt")

    # Frozen parameter invariance check
    frozen_prefixes = ('backbone.', 'segmentation_decoder.', 'seg_head.', 'fwb.')
    frozen_keys = [k for k in base_sd if k.startswith(frozen_prefixes)]
    assert all(torch.equal(cand_sd[k], base_sd[k]) for k in frozen_keys), "Frozen parameter drift detected!"
    print(f"  Invariance: All {len(frozen_keys)} frozen tensors bitwise identical to authoritative baseline.")

    # All weights finite check
    assert all(torch.isfinite(v).all().item() for v in cand_sd.values()), "Non-finite weights found in checkpoint!"
    print(f"  Finiteness: 100% of weights are finite float32 values.")

    # Gate 2: Complexity & Parameter Budget Verification
    print("\n--- GATE 2: COMPUTATIONAL COMPLEXITY & PARAMETER BUDGET ---")
    model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20)
    model.load_state_dict(cand_sd)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter Count:        {total_params:,} (Limit: <= 1,000,000) -> {'PASS' if total_params <= 1_000_000 else 'FAIL'}")
    assert total_params <= 1_000_000, f"Parameter budget exceeded: {total_params} > 1,000,000"

    _, gflops = profile_complexity(resolution=1024)
    print(f"  Static GFLOPs at 1024:  {gflops:.3f} GFLOPs (Limit: <= 12.0 GFLOPs) -> {'PASS' if gflops <= 12.0 else 'FAIL'}")
    assert gflops <= 12.0, f"GFLOPs budget exceeded: {gflops} > 12.0"

    # Gate 3: Static ONNX Export (Track 2 Contract)
    print("\n--- GATE 3: ISOLATED STATIC ONNX EXPORT ---")
    out_dir = "deploy/exports/controlled_epoch_5"
    os.makedirs(out_dir, exist_ok=True)
    out_onnx_path = os.path.join(out_dir, "eurs_net_v2_controlled_ep5.onnx")

    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["restored_image", "seg_logits"]

    print(f"  Target ONNX Path: {out_onnx_path}")
    print(f"  Opset Version:    18 (Static Jetson TensorRT target)")
    print(f"  Input Shape:      [1, 3, 1024, 1024] float32")
    print(f"  Output Names:     {output_names}")

    torch.onnx.export(
        model,
        dummy_input,
        out_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # Strictly static
        dynamo=False
    )
    print("  ONNX export executed successfully.")

    # Validate ONNX model with onnx.checker
    onnx_model = onnx.load(out_onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_size_bytes = os.path.getsize(out_onnx_path)
    onnx_sha256 = compute_file_sha256(out_onnx_path)
    print(f"  ONNX Validation: Model passed onnx.checker.check_model().")
    print(f"  ONNX File Size:  {onnx_size_bytes:,} bytes ({onnx_size_bytes / (1024 * 1024):.2f} MB)")
    print(f"  ONNX SHA-256:    {onnx_sha256}")

    # Check for external data dependency (must be self-contained)
    external_data = [f for f in os.listdir(out_dir) if f.endswith('.onnx.data')]
    assert len(external_data) == 0, f"Found external ONNX data files: {external_data}"
    print(f"  Self-Contained:  Confirmed (zero external .onnx.data files).")

    # Gate 4: ONNX Runtime I/O Signature Verification
    print("\n--- GATE 4: ONNX RUNTIME I/O CONTRACT VERIFICATION ---")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(out_onnx_path, session_options, providers=['CPUExecutionProvider'])

    in_meta = session.get_inputs()[0]
    out_metas = {o.name: o for o in session.get_outputs()}

    assert in_meta.name == "input", f"Input name mismatch: {in_meta.name} != 'input'"
    assert in_meta.shape == [1, 3, 1024, 1024], f"Input shape mismatch: {in_meta.shape}"
    assert in_meta.type == "tensor(float)", f"Input type mismatch: {in_meta.type}"

    assert "restored_image" in out_metas, "Missing output 'restored_image'"
    assert out_metas["restored_image"].shape == [1, 3, 1024, 1024], f"Output shape mismatch: {out_metas['restored_image'].shape}"
    assert out_metas["restored_image"].type == "tensor(float)"

    assert "seg_logits" in out_metas, "Missing output 'seg_logits'"
    assert out_metas["seg_logits"].shape == [1, 5, 1024, 1024], f"Output shape mismatch: {out_metas['seg_logits'].shape}"
    assert out_metas["seg_logits"].type == "tensor(float)"

    print("  Input 'input':           [1, 3, 1024, 1024] float32 -> [VERIFIED]")
    print("  Output 'restored_image': [1, 3, 1024, 1024] float32 -> [VERIFIED]")
    print("  Output 'seg_logits':     [1, 5, 1024, 1024] float32 -> [VERIFIED]")

    # Gate 5: PyTorch vs ONNX Numerical Parity Tests
    print("\n--- GATE 5: NUMERICAL PARITY (PYTORCH vs ONNX RUNTIME) ---")
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)
    
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    sample_indices = {}
    for s in scenes:
        sample_indices[s] = [i for i, smp in enumerate(val_ds.samples) if smp['scene_id'] == s][:5]

    scene_parity_results = {}
    all_max_diffs_rest = []
    all_max_diffs_seg = []

    for s in scenes:
        scene_diffs_rest = []
        scene_diffs_seg = []
        mask_match_rates = []
        pt_psnrs = []
        ort_psnrs = []
        rest_ranges_ok = True

        for idx in sample_indices[s]:
            deg, clean, _, _ = val_ds[idx]
            deg_t = deg.unsqueeze(0)
            clean_np = clean.permute(1, 2, 0).numpy()

            # PyTorch inference
            with torch.no_grad():
                pt_rest, pt_seg = model(deg_t)
            pt_rest_np = pt_rest.squeeze(0).permute(1, 2, 0).numpy()
            pt_seg_np = pt_seg.squeeze(0).numpy()
            pt_mask = np.argmax(pt_seg_np, axis=0)

            # ONNX Runtime inference
            ort_outs = session.run(["restored_image", "seg_logits"], {"input": deg_t.numpy()})
            ort_rest_np = ort_outs[0].squeeze(0).transpose(1, 2, 0)
            ort_seg_np = ort_outs[1].squeeze(0)
            ort_mask = np.argmax(ort_seg_np, axis=0)

            # Check finiteness and range [0, 1]
            assert np.isfinite(ort_rest_np).all(), "Non-finite values in ONNX restored_image!"
            assert np.isfinite(ort_seg_np).all(), "Non-finite values in ONNX seg_logits!"
            assert np.min(ort_rest_np) >= 0.0 and np.max(ort_rest_np) <= 1.0, f"Restored range out of [0, 1]: [{np.min(ort_rest_np)}, {np.max(ort_rest_np)}]"

            # Absolute differences
            d_rest = float(np.max(np.abs(pt_rest_np - ort_rest_np)))
            d_seg = float(np.max(np.abs(pt_seg_np - ort_seg_np)))
            mask_match = float(np.mean(pt_mask == ort_mask)) * 100.0

            scene_diffs_rest.append(d_rest)
            scene_diffs_seg.append(d_seg)
            mask_match_rates.append(mask_match)

            p_pt, _ = compute_psnr_and_norm(pt_rest_np, clean_np)
            p_ort, _ = compute_psnr_and_norm(ort_rest_np, clean_np)
            pt_psnrs.append(p_pt)
            ort_psnrs.append(p_ort)

        scene_parity_results[s] = {
            'max_diff_restored': float(np.max(scene_diffs_rest)),
            'mean_diff_restored': float(np.mean(scene_diffs_rest)),
            'max_diff_seg_logits': float(np.max(scene_diffs_seg)),
            'mean_diff_seg_logits': float(np.mean(scene_diffs_seg)),
            'mask_exact_match_pct': float(np.mean(mask_match_rates)),
            'pt_psnr_mean': float(np.mean(pt_psnrs)),
            'ort_psnr_mean': float(np.mean(ort_psnrs)),
            'psnr_discrepancy': float(abs(np.mean(pt_psnrs) - np.mean(ort_psnrs))),
            'parity_passed': bool(np.max(scene_diffs_rest) < 1e-4 and np.max(scene_diffs_seg) < 1e-4)
        }

        all_max_diffs_rest.append(np.max(scene_diffs_rest))
        all_max_diffs_seg.append(np.max(scene_diffs_seg))
        print(f"  [{s:10s}] Max Delta Restored: {np.max(scene_diffs_rest):.2e} | Max Delta Seg: {np.max(scene_diffs_seg):.2e} | Mask Match: {np.mean(mask_match_rates):.4f}%")

    global_max_diff_rest = float(np.max(all_max_diffs_rest))
    global_max_diff_seg = float(np.max(all_max_diffs_seg))
    all_parity_passed = bool(global_max_diff_rest < 1e-4 and global_max_diff_seg < 1e-4)

    print(f"\n  Global Max Restored Delta: {global_max_diff_rest:.2e} (< 1e-4 threshold) -> {'PASS' if global_max_diff_rest < 1e-4 else 'FAIL'}")
    print(f"  Global Max Seg Delta:      {global_max_diff_seg:.2e} (< 1e-4 threshold) -> {'PASS' if global_max_diff_seg < 1e-4 else 'FAIL'}")
    print(f"  Numerical Parity Status:   {'ALL PASS' if all_parity_passed else 'FAIL'}")
    assert all_parity_passed, "Numerical parity check failed!"

    # Gate 6: Full 800-Image Validation Re-Run on Candidate Model
    print("\n--- GATE 6: FULL 800-IMAGE VALIDATION RE-VERIFICATION ---")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    t0 = time.time()
    val_metrics = evaluate_full_dataset(model, val_loader, device)
    val_time = round(time.time() - t0, 2)

    print(f"  Validation Time:         {val_time}s across {val_metrics['total_images']} images")
    print(f"  PSNR:                    {val_metrics['PSNR_dB']:.2f} dB (Norm: {val_metrics['PSNR_norm']:.4f})")
    print(f"  SSIM:                    {val_metrics['SSIM']:.4f}")
    print(f"  Sharpness Ratio:         {val_metrics['sharpness_ratio']:.4f}")
    print(f"  Q_rest:                  {val_metrics['Q_rest']:.4f}")
    print(f"  Aggregate Clipping:      {val_metrics['clipping_aggregate_pct']:.4f}% (Low: {val_metrics['clipping_low_0_pct']:.2f}%, High: {val_metrics['clipping_high_1_pct']:.2f}%)")
    print(f"  Scene fish12 Clipping:   {val_metrics['per_scene']['fish12']['clip_pct_mean']:.4f}% (Max: {val_metrics['per_scene']['fish12']['clip_pct_max']:.2f}%)")
    print(f"  Segmentation mIoU:       {val_metrics['mIoU']:.4f} | Macro F1: {val_metrics['macro_F1']:.4f}")
    print(f"  Shark IoU (>= 0.1472):   {val_metrics['shark_iou']:.4f} -> {'PASS' if val_metrics['shark_iou'] >= 0.1472 else 'FAIL'}")
    print(f"  Standard Composite:      {val_metrics['composite_standard_0_50']:.4f}")
    print(f"  Surrogate Composite:     {val_metrics['composite_surrogate']:.4f}")

    # Verify replication vs saved Epoch 5 report
    assert abs(val_metrics['PSNR_dB'] - 18.16) < 0.05, f"PSNR discrepancy: {val_metrics['PSNR_dB']} vs 18.16"
    assert abs(val_metrics['SSIM'] - 0.6762) < 0.005, f"SSIM discrepancy: {val_metrics['SSIM']} vs 0.6762"
    assert abs(val_metrics['mIoU'] - 0.2683) < 0.0001, f"mIoU discrepancy: {val_metrics['mIoU']} vs 0.2683"
    assert abs(val_metrics['shark_iou'] - 0.1472) < 0.0001, f"Shark IoU discrepancy: {val_metrics['shark_iou']} vs 0.1472"
    assert abs(val_metrics['composite_standard_0_50'] - 0.2297) < 0.0005, f"Composite discrepancy: {val_metrics['composite_standard_0_50']} vs 0.2297"
    print("  Replication Integrity:   100% MATCH against audited Epoch 5 training logs!")

    # Gate 7: Generate Complete JSON Audit Report
    os.makedirs("audits/deployment_readiness", exist_ok=True)
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': {
            'path': ckpt_path,
            'sha256': cand_sha256,
            'size_bytes': cand_size,
            'epoch': cand_ckpt.get('epoch')
        },
        'onnx_export': {
            'path': out_onnx_path,
            'sha256': onnx_sha256,
            'size_bytes': onnx_size_bytes,
            'opset_version': 18,
            'input_contract': {'name': in_meta.name, 'shape': in_meta.shape, 'dtype': in_meta.type, 'range': [0.0, 1.0]},
            'output_contracts': {
                'restored_image': {'shape': out_metas['restored_image'].shape, 'dtype': out_metas['restored_image'].type, 'range': [0.0, 1.0]},
                'seg_logits': {'shape': out_metas['seg_logits'].shape, 'dtype': out_metas['seg_logits'].type, 'type': 'raw_logits'}
            },
            'self_contained_single_file': True
        },
        'complexity_budget': {
            'parameters': total_params,
            'param_budget': 1_000_000,
            'param_pass': bool(total_params <= 1_000_000),
            'gflops_1024': gflops,
            'gflops_budget': 12.0,
            'gflops_pass': bool(gflops <= 12.0)
        },
        'numerical_parity': {
            'tolerance_threshold': 1e-4,
            'global_max_diff_restored': global_max_diff_rest,
            'global_max_diff_seg_logits': global_max_diff_seg,
            'parity_passed': all_parity_passed,
            'per_scene_parity': scene_parity_results
        },
        're_verified_validation_metrics': val_metrics,
        'unresolved_blockers_and_disclaimers': {
            'official_Q_NR': 'PENDING_OFFICIAL_SERVER_EVALUATION (UIQM/UCIQE/CCF not fabricated locally)',
            'real_jetson_hardware_latency': 'PENDING_PHYSICAL_DEVICE_BENCHMARKING (Local machine is Windows RTX 5090; challenge hardware is Jetson AGX Orin 30W)',
            'track1_resizing_script': 'NOT_APPLICABLE_TO_TRACK2 (Track 2 is static ONNX evaluation; Track 1 uses paired PNG masks and image folder)',
            'submission_upload_status': 'STOPPED_AWAITING_USER_APPROVAL (No upload initiated)'
        },
        'deployment_verdict': 'GO (TRACK 2 ONNX DEPLOYMENT CANDIDATE READY)'
    }

    report_json_path = "audits/deployment_readiness/onnx_export_parity_controlled_ep5.json"
    with open(report_json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\nAudit Report Written To: {report_json_path}")
    print("=" * 85)
    print("RESULT: ALL GATES PASSED! MODEL DEPLOYMENT READY FOR TRACK 2.")
    print("=" * 85)


if __name__ == '__main__':
    main()
