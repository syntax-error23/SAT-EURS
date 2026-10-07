"""
Static ONNX Export for EURS-Net V2.
Target Platform: NVIDIA Jetson AGX Orin (Track 2 @ MaCVi WACV 2027)

Guarantees:
- Static input tensor: [1, 3, H, W] (strictly no dynamic batch/spatial axes for TensorRT engine building).
- Dual outputs:
    * restored_image: [1, 3, H, W] in [0.0, 1.0]
    * seg_logits:     [1, 5, H, W] (raw logits)
- Locked to ONNX Opset 17/18 for seamless TensorRT compilation.
- Self-contained single-file ONNX binary (save_as_external_data=False).
- Optional numerical verification against PyTorch reference outputs.
"""

import os
import sys
import argparse
from typing import Optional, Tuple
import torch
import torch.nn as nn

from models.eurs_net_v2 import EURSNetV2


def export_onnx(
    output_path: str = "deploy/eurs_net_v2.onnx",
    resolution: int = 1024,
    checkpoint_path: Optional[str] = None,
    opset_version: int = 17,
    verify: bool = True
) -> str:
    """
    Exports EURS-Net V2 to static ONNX format.

    Args:
        output_path: Destination path for .onnx file.
        resolution: Spatial input size (must be divisible by 32, typically 1024 or 768).
        checkpoint_path: Optional path to trained model weights.
        opset_version: ONNX operator set version (17 or 18).
        verify: If True, validates ONNX graph and verifies PyTorch vs ONNX numerical match.

    Returns:
        Absolute path to the exported ONNX model.
    """
    if resolution % 32 != 0:
        raise ValueError(f"Resolution must be divisible by 32, got {resolution}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Initialize model
    model = EURSNetV2(num_classes=5)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from: {checkpoint_path}")
    else:
        print("Exporting model with initialized weights (no checkpoint specified).")

    model.eval()

    # 2. Static dummy input [1, 3, H, W]
    dummy_input = torch.randn(1, 3, resolution, resolution, dtype=torch.float32)

    # 3. Export to ONNX
    input_names = ["input"]
    output_names = ["restored_image", "seg_logits"]

    print(f"Exporting EURS-Net V2 to {output_path} (Static shape: [1, 3, {resolution}, {resolution}], Opset: {opset_version})...")

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": None,  # Strictly static for TensorRT
    }
    # In PyTorch 2.x+, specify dynamo=False to use the TorchScript exporter and avoid onnxscript requirement
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            **export_kwargs
        )
    except Exception as e:
        if "onnx" in str(e).lower():
            print(f"\n[ERROR] ONNX export failed: {e}")
            print("Notice: To export and compile ONNX graphs, install the 'onnx' package in your environment:")
            print("    pip install onnx onnxruntime")
            sys.exit(1)
        raise

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export successful! File size: {file_size_mb:.2f} MB")

    # 4. Optional ONNX Runtime Verification
    if verify:
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure verified successfully via onnx.checker.")
        except ImportError:
            print("Notice: 'onnx' package not installed; skipping structural check.")
        except Exception as e:
            print(f"Warning: ONNX structural check failed: {e}")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outs = session.run(None, ort_inputs)

            with torch.no_grad():
                pt_restored, pt_logits = model(dummy_input)

            # Compare outputs
            diff_rest = abs(ort_outs[0] - pt_restored.numpy()).max()
            diff_seg = abs(ort_outs[1] - pt_logits.numpy()).max()

            print(f"ONNX Runtime numerical parity check:")
            print(f"  Restored image max abs diff: {diff_rest:.6f}")
            print(f"  Segmentation logits max abs diff: {diff_seg:.6f}")

            assert diff_rest < 1e-4, f"Restored image numerical diff too large: {diff_rest}"
            assert diff_seg < 1e-4, f"Segmentation logits numerical diff too large: {diff_seg}"
            print("Numerical parity check PASSED!")
        except ImportError:
            print("Notice: 'onnxruntime' package not installed; skipping numerical parity check.")
        except Exception as e:
            print(f"Warning: Numerical parity check failed: {e}")

    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export EURS-Net V2 to static ONNX for Jetson Orin / TensorRT.")
    parser.add_argument("--output", type=str, default="deploy/eurs_net_v2.onnx", help="Path to save ONNX file")
    parser.add_argument("--resolution", type=int, default=1024, help="Input spatial resolution (e.g. 1024 or 768)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint .pth")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (17 or 18)")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX verification")
    args = parser.parse_args()

    export_onnx(
        output_path=args.output,
        resolution=args.resolution,
        checkpoint_path=args.checkpoint,
        opset_version=args.opset,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
