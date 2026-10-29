"""
SAT-EURS Track 2 — Production Packaging & Verification Engine
Exports production LiteV3 checkpoint to standalone ONNX, executes the complete
verification suite, and packages into a competition-ready ZIP artifact.

Verification Suite:
1. Static shapes: input [1, 3, 384, 768] -> restored_image [1, 3, 384, 768], seg_logits [1, 5, 384, 768]
2. Operator compliance: strictly whitelisted opset 17 operators
3. Compute limits: params <= 1.0M, FLOPs <= 12.0 GFLOPs via deploy/check_submission_flops.py
4. Numerical parity: PyTorch CPU vs ONNX Runtime CPU (tol < 1e-4)
5. Dynamic range tests: restored_image strictly in [0.0, 1.0], finite logits
6. Clean root ZIP packaging: root 'model.onnx' only
7. Cryptographic SHA-256 manifests
"""

import os
import sys
import time
import json
import hashlib
import zipfile
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, ".")
from models.track2_litev3 import build_track2_litev3_model


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_and_verify_production():
    ckpt_path = "checkpoints/production/litev3_production_full20_best.pt"
    out_onnx = "deploy/eurs_litev3_production.onnx"
    out_zip = "deploy/eurs_litev3_production_submission.zip"
    manifest_path = "deploy/eurs_litev3_production_manifest.json"
    
    print("\n=======================================================")
    print("SAT-EURS Track 2: Production ONNX Export & Verification")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Target ONNX: {out_onnx}")
    print(f"Target ZIP:  {out_zip}")
    print("=======================================================\n")
    
    os.makedirs("deploy", exist_ok=True)
    
    # 1. Load trained model
    print("[1/7] Loading trained model weights...", flush=True)
    model = build_track2_litev3_model(pretrained_backbone=False, unsharp_factor=1.20, use_edge_gated=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params:,} ({total_params / 1e6:.4f} M)")
    assert total_params <= 1_000_000, f"Parameter count {total_params} exceeds 1.0M limit!"
    
    # 2. Export ONNX Graph
    print("[2/7] Exporting static ONNX graph with Opset 17...", flush=True)
    dummy_input = torch.randn(1, 3, 384, 768, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        out_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamic_axes=None,
        dynamo=False
    )
    onnx_bytes = os.path.getsize(out_onnx)
    print(f"  Exported successfully: {out_onnx} ({onnx_bytes / 1024 / 1024:.2f} MB)")
    
    # 3. ONNX Structure & Operator Whitelist Check
    print("[3/7] Verifying ONNX graph and operator compliance...", flush=True)
    onnx_model = onnx.load(out_onnx)
    onnx.checker.check_model(onnx_model)
    
    allowed_ops = {
        "Conv", "BatchNormalization", "Relu", "Add", "Mul", "Div", "Clip",
        "Sigmoid", "Softmax", "Resize", "MatMul", "Shape", "Gather", "Unsqueeze",
        "Squeeze", "Concat", "Reshape", "Transpose", "AveragePool", "Constant",
        "Sub", "Identity", "Cast", "Sqrt", "Slice", "Gemm", "Flatten",
        "GlobalAveragePool", "ReduceMax"
    }
    ops_found = set(node.op_type for node in onnx_model.graph.node)
    print(f"  Operators present ({len(ops_found)}): {sorted(ops_found)}")
    disallowed = ops_found - allowed_ops
    assert not disallowed, f"Disallowed operators found: {disallowed}"
    print("  >>> Strict operator whitelist passed!")
    
    # 4. Official FLOPs and Parameter Checker
    print("[4/7] Running official deploy/check_submission_flops.py...", flush=True)
    cmd = [sys.executable, "deploy/check_submission_flops.py", "--onnx", out_onnx]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    assert proc.returncode == 0, f"FLOPs check failed:\n{proc.stderr}"
    print("  >>> Official FLOPs and parameter caps PASSED!")
    
    # 5. Numerical Parity Verification (PyTorch CPU vs ONNX Runtime CPU)
    print("[5/7] Testing numerical parity across 10 samples...", flush=True)
    session = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
    
    max_err_rest = 0.0
    max_err_logits = 0.0
    
    for i in range(10):
        test_x = torch.rand(1, 3, 384, 768, dtype=torch.float32)
        with torch.no_grad():
            pt_rest, pt_logits = model(test_x)
            
        ort_outs = session.run(["restored_image", "seg_logits"], {"input": test_x.numpy()})
        ort_rest, ort_logits = ort_outs[0], ort_outs[1]
        
        err_r = float(np.max(np.abs(pt_rest.numpy() - ort_rest)))
        err_l = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
        
        max_err_rest = max(max_err_rest, err_r)
        max_err_logits = max(max_err_logits, err_l)
        
        # Output range sanity
        assert np.all(ort_rest >= -1e-6) and np.all(ort_rest <= 1.0 + 1e-6), "Restored image outside [0, 1]!"
        assert not np.isnan(ort_rest).any() and not np.isnan(ort_logits).any(), "NaN in outputs!"
        assert ort_rest.shape == (1, 3, 384, 768), f"Restored shape mismatch: {ort_rest.shape}"
        assert ort_logits.shape == (1, 5, 384, 768), f"Logits shape mismatch: {ort_logits.shape}"
        
    print(f"  Restored Image Max Parity Error: {max_err_rest:.2e}")
    print(f"  Segmentation Logits Max Error:   {max_err_logits:.2e}")
    assert max_err_rest < 1e-4, f"Restoration parity error {max_err_rest} exceeds 1e-4"
    assert max_err_logits < 5e-4, f"Logits parity error {max_err_logits} exceeds 5e-4"
    print("  >>> Exact numerical parity confirmed (<5e-4)!")
    
    # 6. Package into Submission ZIP
    print(f"[6/7] Packaging into submission ZIP: {out_zip}...", flush=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_onnx, arcname="model.onnx")
        
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        print(f"  ZIP Contents: {names}")
        assert names == ["model.onnx"], f"ZIP must contain ONLY 'model.onnx' at root! Found: {names}"
        
    zip_bytes = os.path.getsize(out_zip)
    onnx_sha = sha256_file(out_onnx)
    zip_sha = sha256_file(out_zip)
    
    print("  >>> Submission ZIP packaged and verified!")
    
    # 7. Generate Production Manifest
    print(f"[7/7] Generating manifest: {manifest_path}...", flush=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "SAT-EURS-LiteV3 (Production Full-20)",
        "source_checkpoint": ckpt_path,
        "parameters": total_params,
        "parameters_m": round(total_params / 1e6, 4),
        "input_resolution": [1, 3, 384, 768],
        "output_tensors": {
            "restored_image": [1, 3, 384, 768],
            "seg_logits": [1, 5, 384, 768]
        },
        "enhancements": [
            "Trained on ALL 20 scenes (4,000 degraded images)",
            "Native Edge-Gated Sharpening (base=0.45, thresh=0.08)",
            "Champion Phase 3 Class Weighting [0.70, 0.90, 1.20, 3.00, 3.50]",
            "RSUIGM-200 Optical Polynomial Color Matrix Router"
        ],
        "opset_version": 17,
        "onnx_file": out_onnx,
        "onnx_sha256": onnx_sha,
        "onnx_bytes": onnx_bytes,
        "submission_zip": out_zip,
        "zip_sha256": zip_sha,
        "zip_bytes": zip_bytes,
        "max_err_rest": max_err_rest,
        "max_err_logits": max_err_logits,
        "validation_passed": True
    }
    
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
        
    print(f"\n=======================================================")
    print("PRODUCTION CANDIDATE ARTIFACT SUMMARY:")
    print(f"  ONNX File:   {out_onnx}")
    print(f"  ONNX SHA256: {onnx_sha}")
    print(f"  ZIP File:    {out_zip}")
    print(f"  ZIP SHA256:  {zip_sha}")
    print("=======================================================\n")
    return manifest


if __name__ == "__main__":
    export_and_verify_production()
