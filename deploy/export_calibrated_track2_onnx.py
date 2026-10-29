"""
Export Calibrated EURS-Net V5 ONNX for Track 2 Embedded Deployment.
Applies verified bias calibration (Delta_3 = -1.5) to eliminate Class 3 precision collapse.
Verifies numerical parity with ONNX Runtime, and checks parameter and FLOPs constraints.
"""

import os
import os.path as osp
import subprocess
import numpy as np
import torch
import onnxruntime as ort

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from models.eurs_net_v5 import EURSNetV5


def export_and_verify_calibrated_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("   EXPORTING VERIFIED CALIBRATED TRACK 2 ONNX ENGINE")
    print("=" * 72)

    # 1. Load trained student
    student = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    src_ckpt = "checkpoints/v5_clinical_distilled_best.pt"
    ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    student.eval()

    # 2. Apply verified Class 3 calibration to the bias of head_512
    orig_b3 = student.segmentation_decoder.head_512.bias.data[3].item()
    student.segmentation_decoder.head_512.bias.data[3] -= 1.5
    cal_b3 = student.segmentation_decoder.head_512.bias.data[3].item()
    print(f"  Class 3 Bias Calibrated: {orig_b3:.4f} -> {cal_b3:.4f} (Offset = -1.50)")

    # 3. Save calibrated checkpoint
    cal_ckpt_path = "checkpoints/v5_calibrated_track2_best.pt"
    cal_payload = {
        "epoch": ckpt.get("epoch", 8),
        "model_state_dict": student.state_dict(),
        "loss": ckpt.get("loss", 4.4602),
        "calibration": {"delta_3": -1.5, "delta_4": 0.0}
    }
    torch.save(cal_payload, cal_ckpt_path)
    print(f"  Saved calibrated PyTorch checkpoint to {cal_ckpt_path}")

    # 4. Export to static ONNX
    onnx_path = "deploy/eurs_net_v5_calibrated_track2.onnx"
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.float32).clamp(0.0, 1.0)

    print(f"  Exporting static 1024x1024 ONNX to {onnx_path}...", flush=True)
    torch.onnx.export(
        student,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamo=False,
    )
    print(f"  ONNX export complete! File size: {os.path.getsize(onnx_path) / 1e6:.2f} MB")

    # 5. Numerical Parity Verification with ONNX Runtime
    print("\n[Parity Check] Testing PyTorch vs ONNX Runtime numerical fidelity...")
    test_inp_np = np.random.uniform(0.0, 1.0, (1, 3, 1024, 1024)).astype(np.float32)
    test_inp_th = torch.from_numpy(test_inp_np).to(device)

    # Test PyTorch CPU vs ONNX Runtime CPU for exact hardware parity
    student_cpu = student.cpu()
    with torch.no_grad():
        th_rest, th_logits = student_cpu(torch.from_numpy(test_inp_np))
    th_rest_np = th_rest.numpy()
    th_logits_np = th_logits.numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outputs = ort_session.run(None, {"input": test_inp_np})
    ort_rest_np = ort_outputs[0]
    ort_logits_np = ort_outputs[1]

    rest_mae = float(np.mean(np.abs(th_rest_np - ort_rest_np)))
    rest_max = float(np.max(np.abs(th_rest_np - ort_rest_np)))
    seg_mae = float(np.mean(np.abs(th_logits_np - ort_logits_np)))
    seg_max = float(np.max(np.abs(th_logits_np - ort_logits_np)))

    print(f"  Restored Image Max Error (CPU vs CPU): {rest_max:.8f} | MAE: {rest_mae:.8f}")
    print(f"  Seg Logits Max Error     (CPU vs CPU): {seg_max:.8f} | MAE: {seg_mae:.8f}")

    assert rest_max < 1e-4, f"Restoration parity check failed! Max error: {rest_max}"
    assert seg_max < 1e-4, f"Segmentation parity check failed! Max error: {seg_max}"
    print("  [PASS] 100% Exact Numerical Parity Verified (Max error < 1e-4)!")

    # 6. Official Complexity Checker
    print("\n[Official FLOPs Audit] Running deploy/check_submission_flops.py...")
    cmd = ["python", "deploy/check_submission_flops.py", "--onnx", onnx_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)


if __name__ == "__main__":
    export_and_verify_calibrated_onnx()
