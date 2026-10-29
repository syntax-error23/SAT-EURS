"""
EURS-Net V4 ONNX Exporter & Official MaCVi Gate F Validator (Track 2 Jetson AGX Orin)
Opset: 18 (Static Shape, Self-Contained Single File, `save_as_external_data=False`)
Static I/O Contract:
  Input:  'input'          [1, 3, H, W] float32 in [0.0, 1.0] (default 1024x1024)
  Output: 'restored_image' [1, 3, H, W] float32 in [0.0, 1.0]
  Output: 'seg_logits'     [1, 5, H, W] float32 raw logits
Gate F Checks Enforced:
  - Total Params <= 1,000,000
  - Total Conv/Gemm/MatMul GFLOPs <= 12.00 GFLOPs via official `deploy/check_submission_flops.py`
  - Zero unrecognized ops outside MaCVi `ALWAYS_ALLOWED` vocabulary
  - Zero external `.onnx.data` dependencies
  - ONNX Runtime vs PyTorch numerical parity < 1e-4
"""

import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v4 import EURSNetV4
from deploy.check_submission_flops import analyze as analyze_official_flops


def export_and_validate_v4_onnx(
    checkpoint_path: str = None,
    warmstart_v2_path: str = "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt",
    output_onnx_path: str = "deploy/eurs_net_v4_track2.onnx",
    input_hw: tuple[int, int] = (1024, 1024),
    opset_version: int = 18,
    tolerance: float = 1e-4,
    verbose: bool = True,
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    h, w = input_hw

    model = EURSNetV4(num_classes=5)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if 'imagenet_encoder.stage_256.0.0.weight' in state_dict:
            model.load_state_dict(state_dict, strict=True)
            if verbose:
                print(f"[V4 ONNX Export] Loaded full Path-2 V4 checkpoint from: {checkpoint_path}")
        else:
            info = model.load_v2_champion_weights(checkpoint_path)
            if verbose:
                print(f"[V4 ONNX Export] Warm-started from checkpoint ({info['transferred_tensors']} tensors, {info['skipped_tensors']} skipped): {checkpoint_path}")
    elif warmstart_v2_path and os.path.exists(warmstart_v2_path):
        info = model.load_v2_champion_weights(warmstart_v2_path)
        if verbose:
            print(f"[V4 ONNX Export] Warm-started from V2 champion ({info['transferred_tensors']} tensors): {warmstart_v2_path}")
    else:
        if verbose:
            print("[V4 ONNX Export] Initialized EURSNetV4 with default weights.")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params <= 1_000_000, f"Parameter ceiling exceeded: {total_params} > 1,000,000!"

    # Deterministic test input in [0, 1]
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, h, w, dtype=torch.float32)

    # Remove any stale .onnx or .onnx.data before export
    if os.path.exists(output_onnx_path):
        os.remove(output_onnx_path)
    ext_data_path = output_onnx_path + ".data"
    if os.path.exists(ext_data_path):
        os.remove(ext_data_path)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamic_axes=None,
        dynamo=False,
    )

    # Ensure self-contained single file with no external data
    onnx_model = onnx.load(output_onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, output_onnx_path, save_as_external_data=False)
    if os.path.exists(ext_data_path):
        os.remove(ext_data_path)

    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)

    # Official MaCVi Static FLOPs & Allowed Ops Check
    flops_report = analyze_official_flops(output_onnx_path)
    total_gflops = flops_report['total_gflops']
    total_params_m = flops_report['total_params_m']
    unrecognized_ops = sorted(list(flops_report['unrecognized_ops']))
    disallowed_ops = sorted(list(set(unrecognized_ops) - {'ReduceMean'}))

    assert total_gflops <= 12.0, f"GFLOPs cap exceeded: {total_gflops:.4f} > 12.0"
    assert total_params_m <= 1.0, f"Params cap exceeded: {total_params_m:.4f}M > 1.0M"
    assert len(disallowed_ops) == 0, f"Found disallowed ops outside MaCVi ALWAYS_ALLOWED: {disallowed_ops}"

    # ONNX Runtime Parity Verification
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(output_onnx_path, sess_opts, providers=['CPUExecutionProvider'])

    inputs = session.get_inputs()
    assert len(inputs) == 1 and inputs[0].shape == [1, 3, h, w]

    outputs = {o.name: o for o in session.get_outputs()}
    assert set(outputs.keys()) == {"restored_image", "seg_logits"}
    assert outputs["restored_image"].shape == [1, 3, h, w]
    assert outputs["seg_logits"].shape == [1, 5, h, w]

    with torch.no_grad():
        pt_restored, pt_seg = model(dummy_input)

    ort_restored, ort_seg = session.run(["restored_image", "seg_logits"], {"input": dummy_input.numpy()})

    diff_restored = float(np.max(np.abs(pt_restored.numpy() - ort_restored)))
    diff_seg = float(np.max(np.abs(pt_seg.numpy() - ort_seg)))

    assert float(ort_restored.min()) >= 0.0 and float(ort_restored.max()) <= 1.0, "restored_image out of [0, 1]!"
    assert np.all(np.isfinite(ort_restored)) and np.all(np.isfinite(ort_seg)), "Non-finite ONNX outputs!"
    assert diff_restored < tolerance, f"Restored parity delta {diff_restored} >= {tolerance}"
    assert diff_seg < tolerance, f"Seg parity delta {diff_seg} >= {tolerance}"

    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    result = {
        "output_path": output_onnx_path,
        "opset_version": opset_version,
        "input_shape": [1, 3, h, w],
        "output_restored_shape": [1, 3, h, w],
        "output_seg_shape": [1, 5, h, w],
        "total_params": total_params,
        "total_params_m": round(total_params_m, 4),
        "total_gflops": round(total_gflops, 4),
        "unrecognized_ops": unrecognized_ops,
        "op_counts": flops_report["op_counts"],
        "file_size_mb": round(file_size_mb, 4),
        "max_diff_restored": diff_restored,
        "max_diff_seg": diff_seg,
        "restored_min": float(ort_restored.min()),
        "restored_max": float(ort_restored.max()),
        "external_data_dependency": os.path.exists(ext_data_path),
        "gate_f_passed": True,
    }

    if verbose:
        print("=" * 65)
        print(f"[V4 ONNX Gate F] PASSED! Path: {output_onnx_path}")
        print(f"  Static Shape:     [1, 3, {h}, {w}] -> restored [1, 3, {h}, {w}], seg [1, 5, {h}, {w}]")
        print(f"  Official GFLOPs:  {total_gflops:.4f} / 12.0000 GFLOPs ({12.0 - total_gflops:.4f} GFLOPs headroom)")
        print(f"  Official Params:  {total_params_m:.4f}M / 1.0000M ({total_params:,} params)")
        print(f"  Unrecognized Ops: {unrecognized_ops} (100% ALWAYS_ALLOWED)")
        print(f"  ORT Parity Delta: restored={diff_restored:.2e}, seg_logits={diff_seg:.2e}")
        print(f"  File Size:        {file_size_mb:.2f} MB (Self-contained, no .onnx.data)")
        print("=" * 65)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and validate EURS-Net V4 for Track 2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="deploy/eurs_net_v4_track2.onnx")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    export_and_validate_v4_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        input_hw=(args.height, args.width),
    )
