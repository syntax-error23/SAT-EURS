"""
Static ONNX Export & Verification for EURS-Net V2.
Target Platform: NVIDIA Jetson AGX Orin (Track 2 @ MaCVi WACV 2027)

Guarantees:
- Static input tensor: [1, 3, H, W] (strictly no dynamic batch/spatial axes for TensorRT engine building).
- Dual outputs:
    * restored_image: [1, 3, H, W] in [0.0, 1.0]
    * seg_logits:     [1, 5, H, W] (raw logits)
- Locked to ONNX Opset 18 (default) / 17 for seamless TensorRT compilation.
- Self-contained single-file ONNX binary (save_as_external_data=False).
- Rigorous numerical verification against PyTorch reference outputs: max diff < 1e-4.
"""

import os
import sys
import inspect
import argparse
from typing import Optional, Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v2 import EURSNetV2


def export_and_validate_onnx(
    checkpoint_path: Optional[str] = None,
    output_onnx_path: str = "deploy/eurs_net_v2_track2.onnx",
    opset_version: int = 18,
    tolerance: float = 1e-4,
    resolution: int = 1024,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exports EURS-Net V2 to static ONNX and validates schema + numerical parity.
    Called by tests/test_onnx_contract.py and deploy pipelines.
    """
    if resolution % 32 != 0:
        raise ValueError(f"Resolution must be divisible by 32, got {resolution}")

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

    # 3. Export to ONNX
    dummy_input = torch.randn(1, 3, resolution, resolution, dtype=torch.float32)

    input_names = ["input"]
    output_names = ["restored_image", "seg_logits"]

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": None,  # Strictly static axes for Track 2 Jetson Orin
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        **export_kwargs
    )

    if verbose:
        print(f"[ONNX Export] Saved ONNX model to: {output_onnx_path} (Opset {opset_version})")

    # 4. Structural Schema Validation via onnx
    import onnx
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    if verbose:
        print("[ONNX Export] onnx.checker passed successfully.")

    # 5. ONNX Runtime Execution & Numerical Parity Verification
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(output_onnx_path, session_options, providers=['CPUExecutionProvider'])

    # Verify input metadata
    inputs = session.get_inputs()
    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert inputs[0].name == "input", f"Expected input name 'input', got '{inputs[0].name}'"
    assert inputs[0].shape == [1, 3, resolution, resolution], f"Expected input shape [1, 3, {resolution}, {resolution}], got {inputs[0].shape}"

    # Verify output metadata
    outputs = session.get_outputs()
    output_map = {out.name: out for out in outputs}
    assert "restored_image" in output_map, "Missing output 'restored_image'!"
    assert "seg_logits" in output_map, "Missing output 'seg_logits'!"
    assert output_map["restored_image"].shape == [1, 3, resolution, resolution], f"Bad restored_image shape: {output_map['restored_image'].shape}"
    assert output_map["seg_logits"].shape == [1, 5, resolution, resolution], f"Bad seg_logits shape: {output_map['seg_logits'].shape}"

    # Run PyTorch forward
    with torch.no_grad():
        pt_restored, pt_seg = model(dummy_input)

    # Run ONNX Runtime forward
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outs = session.run(["restored_image", "seg_logits"], ort_inputs)
    ort_restored, ort_seg = ort_outs[0], ort_outs[1]

    # Calculate absolute differences
    diff_restored = float(np.max(np.abs(pt_restored.numpy() - ort_restored)))
    diff_seg = float(np.max(np.abs(pt_seg.numpy() - ort_seg)))

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
        "input_shape": [1, 3, resolution, resolution],
        "output_restored_name": "restored_image",
        "output_restored_shape": [1, 3, resolution, resolution],
        "output_seg_name": "seg_logits",
        "output_seg_shape": [1, 5, resolution, resolution],
        "max_diff_restored": diff_restored,
        "max_diff_seg": diff_seg,
        "numerical_parity_passed": True
    }

    if verbose:
        print(f"[ONNX Export] Complete! File Size: {file_size_mb:.2f} MB, Total Params: {total_params:,}")
        print("=" * 60)

    return audit_result


def export_onnx(
    output_path: str = "deploy/eurs_net_v2.onnx",
    resolution: int = 1024,
    checkpoint_path: Optional[str] = None,
    opset_version: int = 18,
    verify: bool = True
) -> str:
    """
    Exports EURS-Net V2 to static ONNX format with optional verification.
    CLI wrapper for Shubham's deployment workflow.
    """
    if verify:
        export_and_validate_onnx(
            checkpoint_path=checkpoint_path,
            output_onnx_path=output_path,
            opset_version=opset_version,
            tolerance=1e-4,
            resolution=resolution,
            verbose=True
        )
    else:
        export_kwargs = {
            "export_params": True,
            "opset_version": opset_version,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["restored_image", "seg_logits"],
            "dynamic_axes": None,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        model = EURSNetV2(num_classes=5)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict)
        model.eval()

        dummy = torch.randn(1, 3, resolution, resolution, dtype=torch.float32)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.onnx.export(model, dummy, output_path, **export_kwargs)

    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export EURS-Net V2 to Track 2 ONNX for Jetson Orin / TensorRT")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth or .pt")
    parser.add_argument("--output", type=str, default="deploy/eurs_net_v2_track2.onnx", help="Output .onnx path")
    parser.add_argument("--resolution", type=int, default=1024, help="Input spatial resolution (e.g. 1024 or 768)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (17 or 18)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max allowed diff between PyTorch and ONNX Runtime")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical verification")
    args = parser.parse_args()

    if args.no_verify:
        export_onnx(
            output_path=args.output,
            resolution=args.resolution,
            checkpoint_path=args.checkpoint,
            opset_version=args.opset,
            verify=False
        )
    else:
        export_and_validate_onnx(
            checkpoint_path=args.checkpoint,
            output_onnx_path=args.output,
            opset_version=args.opset,
            tolerance=args.tolerance,
            resolution=args.resolution,
            verbose=True
        )


if __name__ == "__main__":
    main()
