"""
Export, Verify, and Package Track2SOTAOrin Production ONNX Submission
Target: MaCVi @ WACV 2027 Challenge — Track 2 (Embedded Edge Deployment on Jetson AGX Orin)

Performs:
  1. PyTorch -> Static ONNX Graph Export (Opset 17, [1, 3, 1024, 1024])
  2. Static FLOPs & Parameter Count Analysis (Strict <= 1.0M params, <= 12.0 GFLOPs)
  3. Numerical Parity Verification with ONNX Runtime CPU
  4. Packaging of clean submission archive (strictly 'model.onnx' at root)
  5. SHA-256 hash generation and manifest logging
"""

import os
import sys
import time
import zipfile
import hashlib
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, ".")

from models.track2_sota_orin import Track2SOTAOrin, build_track2_sota_model
from deploy.check_submission_flops import analyze


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_and_package_submission(
    ckpt_path: str = "checkpoints/track2_sota_orin_best.pt",
    out_onnx: str = "deploy/track2_sota_orin.onnx",
    out_zip: str = "deploy/track2_sota_orin_submission.zip"
):
    print("\n" + "=" * 80)
    print("      EXPORTING & VERIFYING TRACK 2 SOTA ORIN PRODUCTION PACKAGE")
    print("=" * 80 + "\n", flush=True)

    os.makedirs(os.path.dirname(out_onnx), exist_ok=True)
    device = torch.device("cpu")

    # 1. Load PyTorch model
    print(f"[1/5] Loading trained checkpoint from {ckpt_path}...", flush=True)
    model = build_track2_sota_model(pretrained_backbone=False)
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found!"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(st, strict=True)
    model.eval()

    # Parameter count check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  PyTorch Model Parameters: {total_params:,} ({total_params / 1e6:.4f} M)")
    assert total_params <= 1_000_000, f"Parameter count {total_params} exceeds 1.0M limit!"

    # 2. Export ONNX Graph
    print("[2/5] Exporting static ONNX graph with Opset 17...", flush=True)
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
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
    onnx_sha = compute_sha256(out_onnx)
    print(f"  Exported successfully: {out_onnx} ({onnx_bytes / 1024 / 1024:.2f} MB)")
    print(f"  ONNX SHA-256: {onnx_sha}")

    # 3. ONNX Structure & FLOPs Check
    print("[3/5] Verifying official FLOPs & parameter caps...", flush=True)
    onnx_model = onnx.load(out_onnx)
    onnx.checker.check_model(onnx_model)

    res = analyze(out_onnx)
    total_gflops = res["total_gflops"]
    total_params_m = res["total_params_m"]
    print(f"  Official GFLOPs:    {total_gflops:.4f} GFLOPs (Cap: <= 12.0000 GFLOPs)")
    print(f"  Official Params:    {total_params_m:.4f} M (Cap: <= 1.0000 M)")

    assert total_gflops <= 12.0000, f"GFLOPs {total_gflops:.4f} exceeds 12.0 cap!"
    assert total_params_m <= 1.0000, f"Parameters {total_params_m:.4f}M exceeds 1.0M cap!"
    print("  --> ALL OFFICIAL COMPLEXITY CAPS PASSED!")

    # 4. Numerical Parity Test with ONNX Runtime CPU
    print("[4/5] Verifying numerical parity between PyTorch and ONNX Runtime...", flush=True)
    session = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
    test_x = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

    with torch.no_grad():
        pt_rest, pt_logits = model(torch.from_numpy(test_x))
    ort_outs = session.run(None, {"input": test_x})
    ort_rest, ort_logits = ort_outs[0], ort_outs[1]

    rest_diff = np.max(np.abs(pt_rest.numpy() - ort_rest))
    logits_diff = np.max(np.abs(pt_logits.numpy() - ort_logits))
    print(f"  Restored Image Max Error: {rest_diff:.2e}")
    print(f"  Logits Max Error:         {logits_diff:.2e}")
    assert rest_diff < 1e-4, f"Restoration parity error {rest_diff} exceeds tolerance!"
    assert logits_diff < 5e-4, f"Logits parity error {logits_diff} exceeds tolerance!"
    print("  --> NUMERICAL PARITY CONFIRMED!")

    # 5. Clean Submission Packaging
    print(f"[5/5] Packaging clean submission archive to {out_zip}...", flush=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_onnx, arcname="model.onnx")

    zip_bytes = os.path.getsize(out_zip)
    zip_sha = compute_sha256(out_zip)
    print(f"  Submission ZIP Created: {out_zip} ({zip_bytes / 1024 / 1024:.2f} MB)")
    print(f"  ZIP SHA-256: {zip_sha}")

    # Verify single model.onnx inside ZIP
    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()
        print(f"  Archive Contents: {names}")
        assert names == ["model.onnx"], f"ZIP structure invalid: {names}"

    print("\n" + "=" * 80)
    print("SUBMISSION ARTIFACT READY FOR UPLOAD:")
    print(f"  ZIP Path:     {out_zip}")
    print(f"  ZIP SHA-256:  {zip_sha}")
    print(f"  Model SHA:    {onnx_sha}")
    print(f"  GFLOPs:       {total_gflops:.4f} / 12.0000")
    print(f"  Parameters:   {total_params_m:.4f}M / 1.0000M")
    print("=" * 80 + "\n")
    return out_zip, zip_sha, onnx_sha


if __name__ == "__main__":
    export_and_package_submission()
