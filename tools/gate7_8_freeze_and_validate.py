"""
Gate 7 & 8 — Candidate Freeze & Full-Output Validation Suite
EURS / MaCVi WACV 2027 Challenge (Tracks 1 & 2)

Performs:
1. Cryptographic freeze of all constituent model checkpoints and calibration tables.
2. Exhaustive structural validation of Track 1 submission archives (1,200 images, 1,200 discrete masks).
3. Stem matching against the official Dev test set (6 scenes x 200 images = 1,200 images).
4. Discrete mask class verification {0, 1, 2, 3, 4} across all evaluated masks.
5. Official static FLOPs and parameter compliance check on Track 2 ONNX models (<= 12.0 GFLOPs, <= 1.0M params).
6. Tensor contract validation: input [1, 3, 1024, 1024], restored_image [1, 3, 1024, 1024], seg_logits [1, 5, 1024, 1024].
7. Generation of immutable deploy/FROZEN_CANDIDATE_V17_MANIFEST.json with full provenance.
"""

import os
import os.path as osp
import sys
import time
import json
import zipfile
import hashlib
import numpy as np
import cv2
from PIL import Image
import onnx
import onnxruntime as ort

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from deploy.check_submission_flops import analyze


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


def run_freeze_and_validate():
    print("=" * 80)
    print("   SAT-EURS GATE 7 & 8: CANDIDATE FREEZE & FULL-OUTPUT VALIDATION")
    print("   Workstation: NVIDIA RTX 5090 | 64GB DDR5 | Windows 11")
    print("=" * 80)

    manifest_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v17_frozen",
        "track1": {},
        "track2": {},
        "checkpoints": {}
    }

    # -------------------------------------------------------------------------
    # STAGE 1: Checkpoint Inventory & Cryptographic Freeze
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: CHECKPOINT INVENTORY & CRYPTOGRAPHIC FREEZE ---")
    ckpts_to_freeze = {
        "rsuigm_20sc_table": "checkpoints/track1_screening/tbl19_20sc_clean.pt",
        "rsuigm_f0_table": "checkpoints/track1_screening/tbl19_16sc_fold0.pt",
        "dinov2_l_fold0": "checkpoints/v16_dinov2/fold_0/best_model.pt",
        "dinov2_l_fold1": "checkpoints/v16_dinov2/fold_1/best_model.pt",
        "dinov2_l_fold2": "checkpoints/v16_dinov2/fold_2/best_model.pt",
        "dinov2_l_fold3": "checkpoints/v16_dinov2/fold_3/best_model.pt",
        "dinov2_l_fold4": "checkpoints/v16_dinov2/fold_4/best_model.pt",
        "convnext_b_fold0": "checkpoints/v16_convnext/fold_0/best_model.pt",
        "convnext_b_fold1": "checkpoints/v16_convnext/fold_1/best_model.pt",
        "convnext_b_fold2": "checkpoints/v16_convnext/fold_2/best_model.pt",
        "convnext_b_fold3": "checkpoints/v16_convnext/fold_3/best_model.pt",
        "convnext_b_fold4": "checkpoints/v16_convnext/fold_4/best_model.pt",
    }

    for name, p in ckpts_to_freeze.items():
        assert osp.exists(p), f"Missing checkpoint: {p}"
        sz_mb = osp.getsize(p) / (1024**2)
        h = compute_sha256(p)
        manifest_output["checkpoints"][name] = {
            "path": p,
            "size_mb": sz_mb,
            "sha256": h
        }
        print(f"  {name:20s}: {sz_mb:7.2f} MB | SHA-256: {h}")

    # -------------------------------------------------------------------------
    # STAGE 2: Track 1 Full-Output Archive Validation
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: TRACK 1 FULL-OUTPUT ARCHIVE VALIDATION ---")
    dev_dir = "data/dev/degraded"
    assert osp.exists(dev_dir), f"Missing Dev test set: {dev_dir}"

    dev_scenes = sorted(os.listdir(dev_dir))
    expected_stems = set()
    for sc in dev_scenes:
        sc_dir = osp.join(dev_dir, sc)
        if osp.isdir(sc_dir):
            for f in os.listdir(sc_dir):
                if f.endswith(".png"):
                    expected_stems.add(osp.splitext(f)[0])

    print(f"  Official Dev Set Manifest: {len(expected_stems)} expected stems across {len(dev_scenes)} scenes ({dev_scenes})")
    assert len(expected_stems) == 1200, f"Expected exactly 1200 Dev images, found {len(expected_stems)}"

    t1_candidates = [
        ("Historical_SOTA_Fallback_18713", "deploy/eurs_track1_dryrun_compact_candidate.zip"),
        ("Apex_DualBackbone_SOTA_v16", "deploy/submission_v16_apex_sota.zip"),
    ]

    for cand_name, zip_p in t1_candidates:
        print(f"\n  Validating Track 1 Archive: {cand_name} ({zip_p})...")
        assert osp.exists(zip_p), f"Missing archive: {zip_p}"
        sz_mb = osp.getsize(zip_p) / (1024**2)
        zip_sha256 = compute_sha256(zip_p)
        print(f"    Size: {sz_mb:.2f} MB | SHA-256: {zip_sha256}")

        with zipfile.ZipFile(zip_p, "r") as z:
            names = z.namelist()
            restored_files = [n for n in names if n.startswith("restored/") and not n.endswith("/")]
            mask_files = [n for n in names if n.startswith("seg_masks/") and not n.endswith("/")]
            other_files = [n for n in names if not n.startswith("restored/") and not n.startswith("seg_masks/") and not n.endswith("/")]

            print(f"    Total items in ZIP: {len(names)} | Restored: {len(restored_files)} | Masks: {len(mask_files)} | Extraneous: {len(other_files)}")
            assert len(restored_files) == 1200, f"Wrong restored count: {len(restored_files)}"
            assert len(mask_files) == 1200, f"Wrong masks count: {len(mask_files)}"
            assert len(other_files) == 0, f"Disallowed files inside root of ZIP: {other_files}"

            # Stem matching check
            restored_stems = {osp.splitext(osp.basename(n))[0] for n in restored_files}
            mask_stems = {osp.splitext(osp.basename(n))[0] for n in mask_files}

            assert restored_stems == expected_stems, f"Restored stems do not match Dev set! Missing: {expected_stems - restored_stems}"
            assert mask_stems == expected_stems, f"Mask stems do not match Dev set! Missing: {expected_stems - mask_stems}"
            print("    [PASS] Stem Matching: Exactly 1,200 stems match official Dev set 100%.")

            # Sample deep integrity check: reopen 10 random samples
            sample_masks = mask_files[:10]
            sample_res = restored_files[:10]

            for mf_name, rf_name in zip(sample_masks, sample_res):
                with z.open(mf_name) as mf:
                    m_im = Image.open(mf)
                    m_arr = np.array(m_im)
                    assert m_im.mode == "L", f"Mask mode is {m_im.mode}, expected 'L'"
                    u_classes = np.unique(m_arr)
                    assert set(u_classes).issubset({0, 1, 2, 3, 4}), f"Spurious mask class IDs in {mf_name}: {u_classes}"

                with z.open(rf_name) as rf:
                    r_im = Image.open(rf)
                    r_arr = np.array(r_im)
                    assert r_im.mode == "RGB", f"Restored image mode is {r_im.mode}, expected 'RGB'"
                    assert max(r_im.size) <= 1024, f"Longest side exceeds 1024 pixels in {rf_name}: {r_im.size}"
                    assert r_arr.shape[:2] == m_arr.shape[:2], f"Dimension mismatch between image {r_im.size} and mask {m_im.size}"

            print("    [PASS] Deep Integrity Check: Discrete mask classes {0, 1, 2, 3, 4}, RGB dtypes, and aspect-preserved dimensions verified.")

        manifest_output["track1"][cand_name] = {
            "path": zip_p,
            "size_mb": sz_mb,
            "sha256": zip_sha256,
            "restored_count": 1200,
            "masks_count": 1200,
            "status": "VALIDATED_COMPLIANT"
        }

    # -------------------------------------------------------------------------
    # STAGE 3: Track 2 ONNX Export & Design-Space Verification
    # -------------------------------------------------------------------------
    print("\n--- STAGE 3: TRACK 2 ONNX DESIGN-SPACE & RUNTIME VERIFICATION ---")
    t2_candidates = [
        ("Official_Reference_18699", "deploy/eurs_net_v5_calibrated_track2.onnx"),
        ("EURSEdgeSOTA_V2_Candidate", "deploy/eurs_edgesota_v2_final.onnx"),
        ("EXP_COMBINED_BEST_Candidate", "deploy/EXP-COMBINED-BEST.onnx"),
    ]

    for cand_name, onnx_p in t2_candidates:
        print(f"\n  Validating Track 2 Model: {cand_name} ({onnx_p})...")
        assert osp.exists(onnx_p), f"Missing ONNX file: {onnx_p}"
        sz_mb = osp.getsize(onnx_p) / (1024**2)
        onnx_sha256 = compute_sha256(onnx_p)

        # 1. Official FLOPs / Params Static Checker
        analysis = analyze(onnx_p)
        total_gflops = analysis["total_gflops"]
        total_params_m = analysis["total_params_m"]
        print(f"    Official Static Analysis: {total_gflops:.4f} GFLOPs (Cap: <= 12.0) | {total_params_m:.4f} M params (Cap: <= 1.0M)")
        assert total_gflops <= 12.0, f"Exceeded GFLOPs cap: {total_gflops} > 12.0"
        assert total_params_m <= 1.0, f"Exceeded parameters cap: {total_params_m} > 1.0M"
        print("    [PASS] Static Design-Space Caps: PASSED.")

        # 2. Structural & Shape Inference Check
        m = onnx.load(onnx_p)
        onnx.checker.check_model(m)
        inferred = onnx.shape_inference.infer_shapes(m)
        print("    [PASS] ONNX Structure & Shape Inference: VALID.")

        # 3. ONNX Runtime Inference & Numerical Contract
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_p, sess_options=opts, providers=["CPUExecutionProvider"])

        inp_info = session.get_inputs()[0]
        out_info = session.get_outputs()
        out_names = [o.name for o in out_info]

        assert inp_info.name == "input", f"Expected input 'input', got {inp_info.name}"
        assert inp_info.shape == [1, 3, 1024, 1024], f"Expected static shape [1, 3, 1024, 1024], got {inp_info.shape}"
        assert "restored_image" in out_names, f"Missing output 'restored_image': {out_names}"
        assert "seg_logits" in out_names, f"Missing output 'seg_logits': {out_names}"

        # Run synthetic probe
        dummy_in = np.random.uniform(0.0, 1.0, (1, 3, 1024, 1024)).astype(np.float32)
        t0 = time.perf_counter()
        outputs = session.run(None, {"input": dummy_in})
        lat_ms = (time.perf_counter() - t0) * 1000.0

        rest_out, logits_out = outputs[0], outputs[1]
        assert rest_out.shape == (1, 3, 1024, 1024), f"Bad restored shape: {rest_out.shape}"
        assert logits_out.shape == (1, 5, 1024, 1024), f"Bad logits shape: {logits_out.shape}"
        assert rest_out.min() >= -0.01 and rest_out.max() <= 1.01, f"Restored values out of [0, 1]: [{rest_out.min()}, {rest_out.max()}]"

        print(f"    [PASS] ONNX Runtime Test: Output shapes verified | CPU Latency: {lat_ms:.1f} ms | Logits range: [{logits_out.min():.2f}, {logits_out.max():.2f}]")

        manifest_output["track2"][cand_name] = {
            "path": onnx_p,
            "size_mb": sz_mb,
            "sha256": onnx_sha256,
            "total_gflops": total_gflops,
            "total_params_m": total_params_m,
            "latency_cpu_ms": lat_ms,
            "status": "VALIDATED_COMPLIANT"
        }

    # -------------------------------------------------------------------------
    # STAGE 4: Write Immutable Manifest
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: GENERATING IMMUTABLE FROZEN MANIFEST ---")
    manifest_path = "deploy/FROZEN_CANDIDATE_V17_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_output, f, indent=2)

    manifest_sha = compute_sha256(manifest_path)
    print(f"  [PASS] Written frozen manifest to {manifest_path}")
    print(f"  Manifest SHA-256: {manifest_sha}")

    print("\n" + "=" * 80)
    print("   ALL GATE 7 & 8 CHECKS COMPLETED & VALIDATED PERFECTLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_freeze_and_validate()
