"""
EURS-Net V2 ONNX Exporter & Validator (Track 2 Jetson Orin Target)
Opset: 18 (Frozen Authoritative Specification)
Static Contract:
  Input:  'input'          [1, 3, 1024, 1024] float32
  Output: 'restored_image' [1, 3, 1024, 1024] float32
  Output: 'seg_logits'     [1, 5, 1024, 1024] float32
Constraints:
  Params <= 1,000,000
  FLOPs  <= 12.0 GFLOPs
  Numerical Parity: max |PyTorch - ONNX Runtime| < 1e-4
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v2 import EURSNetV2


def export_and_validate_onnx(
    checkpoint_path: str = None,
    output_onnx_path: str = "deploy/eurs_net_v2_track2.onnx",
    opset_version: int = 18,
    tolerance: float = 1e-4,
    verbose: bool = True
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    # 1. Instantiate model
    model = EURSNetV2(num_classes=5)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"[ONNX Export] Loaded checkpoint weights from: {checkpoint_path}")
    else:
        if verbose:
            print("[ONNX Export] Initialized EURSNetV2 with default weights.")

    model.eval()

    # 2. Audit Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 1_000_000, f"Parameter ceiling exceeded: {total_params} > 1,000,000!"

    # 3. Export to ONNX Opset 18
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    input_names = ["input"]
    output_names = ["restored_image", "seg_logits"]

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None  # Strictly static axes for Track 2 Jetson Orin
    )

    if verbose:
        print(f"[ONNX Export] Saved ONNX model to: {output_onnx_path} (Opset {opset_version})")

    # 4. Structural Schema Validation via onnx
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    if verbose:
        print("[ONNX Export] onnx.checker passed successfully.")

    # 5. ONNX Runtime Execution & Numerical Parity Verification
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(output_onnx_path, session_options, providers=['CPUExecutionProvider'])

    # Verify input metadata
    inputs = session.get_inputs()
    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert inputs[0].name == "input", f"Expected input name 'input', got '{inputs[0].name}'"
    assert inputs[0].shape == [1, 3, 1024, 1024], f"Expected input shape [1, 3, 1024, 1024], got {inputs[0].shape}"

    # Verify output metadata
    outputs = session.get_outputs()
    output_map = {out.name: out for out in outputs}
    assert "restored_image" in output_map, "Missing output 'restored_image'!"
    assert "seg_logits" in output_map, "Missing output 'seg_logits'!"
    assert output_map["restored_image"].shape == [1, 3, 1024, 1024], f"Bad restored_image shape: {output_map['restored_image'].shape}"
    assert output_map["seg_logits"].shape == [1, 5, 1024, 1024], f"Bad seg_logits shape: {output_map['seg_logits'].shape}"

    # Run PyTorch forward
    with torch.no_grad():
        pt_restored, pt_seg = model(dummy_input)

    # Run ONNX Runtime forward
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outs = session.run(["restored_image", "seg_logits"], ort_inputs)
    ort_restored, ort_seg = ort_outs[0], ort_outs[1]

    # Calculate absolute differences
    diff_restored = np.max(np.abs(pt_restored.numpy() - ort_restored))
    diff_seg = np.max(np.abs(pt_seg.numpy() - ort_seg))

    if verbose:
        print(f"[ONNX Export] Parity Delta Restored: {diff_restored:.2e} (tolerance: {tolerance})")
        print(f"[ONNX Export] Parity Delta Seg Logits: {diff_seg:.2e} (tolerance: {tolerance})")

    assert diff_restored < tolerance, f"Restored output parity failed: delta={diff_restored} >= {tolerance}"
    assert diff_seg < tolerance, f"Seg logits output parity failed: delta={diff_seg} >= {tolerance}"

    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)

    audit_result = {
        "output_path": output_onnx_path,
        "opset_version": opset_version,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "file_size_mb": round(file_size_mb, 4),
        "input_name": "input",
        "input_shape": [1, 3, 1024, 1024],
        "output_restored_name": "restored_image",
        "output_restored_shape": [1, 3, 1024, 1024],
        "output_seg_name": "seg_logits",
        "output_seg_shape": [1, 5, 1024, 1024],
        "max_diff_restored": float(diff_restored),
        "max_diff_seg": float(diff_seg),
        "numerical_parity_passed": True
    }

    if verbose:
        print(f"[ONNX Export] Complete! File Size: {file_size_mb:.2f} MB, Total Params: {total_params:,}")
        print("=" * 60)

    return audit_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EURSNetV2 to Track 2 ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--output", type=str, default="deploy/eurs_net_v2_track2.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (frozen at 18)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max allowed diff between PyTorch and ONNX Runtime")
    args = parser.parse_args()

    export_and_validate_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        opset_version=args.opset,
        tolerance=args.tolerance,
        verbose=True
    )
