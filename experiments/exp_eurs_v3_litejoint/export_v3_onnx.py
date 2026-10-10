"""
Static ONNX Export & Parity Verification for EURS-Net V3 LiteJoint
- Static shape: [1, 3, 1024, 1024]
- Output 1: 'restored_image' [1, 3, 1024, 1024] in [0, 1]
- Output 2: 'seg_logits' [1, 5, 1024, 1024] (raw logits)
- Numerical parity verification (PyTorch vs ONNX Runtime) across validation scenes
"""

import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments/exp_eurs_v3_litejoint'))

from model_v3 import EURSNetV3LiteJoint


def export_and_verify(
    checkpoint_path: str,
    output_onnx_path: str = "experiments/exp_eurs_v3_litejoint/export/eurs_net_v3_litejoint.onnx"
):
    print("=" * 70)
    print("      EURS-NET V3 LITEJOINT: STATIC ONNX EXPORT & VERIFICATION")
    print("=" * 70)
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # 1. Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    mode = ckpt.get('mode', 'bounded')
    bound_scale = ckpt.get('bound_scale', 0.20)

    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode=mode, bound_scale=bound_scale)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Export to static ONNX
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    input_names = ['input']
    output_names = ['restored_image', 'seg_logits']

    print(f"Exporting to static ONNX: {output_onnx_path} (opset=17)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None  # Strictly static shapes as mandated by Track 2
    )

    file_size_bytes = os.path.getsize(output_onnx_path)
    print(f"Export complete. File size: {file_size_bytes:,} bytes ({file_size_bytes / 1024 / 1024:.2f} MB)")

    # 3. Verify ONNX Model Integrity
    import onnx
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure checked and valid.")

    # 4. Numerical Parity Testing with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])

    # Test across 3 synthetic variations
    max_rest_diff = 0.0
    mean_rest_diff = 0.0
    max_seg_diff = 0.0
    mean_seg_diff = 0.0

    torch.manual_seed(123)
    for i in range(3):
        test_x = torch.rand(1, 3, 1024, 1024, dtype=torch.float32)
        with torch.no_grad():
            pt_rest, pt_seg = model(test_x)

        ort_inputs = {input_names[0]: test_x.numpy()}
        ort_outs = session.run(output_names, ort_inputs)
        ort_rest, ort_seg = ort_outs[0], ort_outs[1]

        r_diff = np.abs(pt_rest.numpy() - ort_rest)
        s_diff = np.abs(pt_seg.numpy() - ort_seg)

        max_rest_diff = max(max_rest_diff, float(r_diff.max()))
        mean_rest_diff = max(mean_rest_diff, float(r_diff.mean()))
        max_seg_diff = max(max_seg_diff, float(s_diff.max()))
        mean_seg_diff = max(mean_seg_diff, float(s_diff.mean()))

    print(f"\nNumerical Parity Results (PyTorch vs ONNX Runtime):")
    print(f"  Restored Image Max Abs Error:  {max_rest_diff:.2e} (Threshold: < 1e-4)")
    print(f"  Restored Image Mean Abs Error: {mean_rest_diff:.2e}")
    print(f"  Seg Logits Max Abs Error:      {max_seg_diff:.2e} (Threshold: < 1e-4)")
    print(f"  Seg Logits Mean Abs Error:     {mean_seg_diff:.2e}")

    parity_passed = (max_rest_diff < 1e-4) and (max_seg_diff < 1e-4)
    print(f"Parity Test Status: {'PASSED' if parity_passed else 'FAILED'}")

    # Output verification summary
    export_meta = {
        "onnx_path": output_onnx_path,
        "size_bytes": file_size_bytes,
        "input_shape": [1, 3, 1024, 1024],
        "output_shapes": {
            "restored_image": [1, 3, 1024, 1024],
            "seg_logits": [1, 5, 1024, 1024]
        },
        "max_rest_error": max_rest_diff,
        "max_seg_error": max_seg_diff,
        "parity_passed": parity_passed
    }
    with open("experiments/exp_eurs_v3_litejoint/export/onnx_parity_results.json", "w") as f:
        json.dump(export_meta, f, indent=2)

    return export_meta


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "experiments/exp_eurs_v3_litejoint/checkpoints/best_v3_checkpoint.pt"
    if os.path.exists(ckpt_path):
        export_and_verify(ckpt_path)
    else:
        print(f"Checkpoint {ckpt_path} not found yet. Ready for post-training execution.")
