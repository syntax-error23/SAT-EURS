"""
ONNX Track 2 Contract & Numerical Parity Unit Test (Phase 2A Pre-Flight)
Verifies:
1. Frozen Opset 18 export
2. Input name 'input' [1, 3, 1024, 1024]
3. Output name 'restored_image' [1, 3, 1024, 1024]
4. Output name 'seg_logits' [1, 5, 1024, 1024]
5. Parameter ceiling <= 1,000,000
6. Computational ceiling <= 12.0 GFLOPs
7. Numerical parity between PyTorch and ONNX Runtime: max |PT - ORT| < 1e-4
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tempfile
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ptflops import get_model_complexity_info

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v2 import EURSNetV2
from deploy.export_onnx import export_and_validate_onnx


def test_onnx_export_contract():
    print("=" * 60)
    print("Test 2A.1: Track 2 ONNX Hardware Contract & Numerical Parity")
    print("=" * 60)

    model = EURSNetV2(num_classes=5)
    model.eval()

    # 1. Parameter Ceiling Check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [AUDIT] Total Parameters: {total_params:,} (limit <= 1,000,000)")
    assert total_params <= 1_000_000, f"Parameter count exceeded: {total_params} > 1,000,000"

    # 2. FLOPs Ceiling Check
    macs, params = get_model_complexity_info(
        model,
        (3, 1024, 1024),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    flops = 2 * macs
    gflops = flops / 1e9
    print(f"  [AUDIT] Total FLOPs at 1024x1024: {gflops:.4f} GFLOPs (limit <= 12.0 GFLOPs)")
    assert gflops <= 12.0, f"FLOPs ceiling exceeded: {gflops:.4f} > 12.0 GFLOPs"

    # 3. Export to Temp ONNX and Validate Contract
    with tempfile.TemporaryDirectory() as tmpdir:
        test_onnx_path = os.path.join(tmpdir, "test_eurs_track2.onnx")
        result = export_and_validate_onnx(
            checkpoint_path=None,
            output_onnx_path=test_onnx_path,
            opset_version=18,
            tolerance=1e-4,
            verbose=False
        )

        assert result["opset_version"] == 18, f"Expected opset 18, got {result['opset_version']}"
        assert result["input_name"] == "input"
        assert result["input_shape"] == [1, 3, 1024, 1024]
        assert result["output_restored_name"] == "restored_image"
        assert result["output_restored_shape"] == [1, 3, 1024, 1024]
        assert result["output_seg_name"] == "seg_logits"
        assert result["output_seg_shape"] == [1, 5, 1024, 1024]
        assert result["max_diff_restored"] < 1e-4, f"Restored delta {result['max_diff_restored']} >= 1e-4"
        assert result["max_diff_seg"] < 1e-4, f"Seg logits delta {result['max_diff_seg']} >= 1e-4"
        assert result["numerical_parity_passed"] is True

        print("  [PASS] Frozen Opset 18 verified.")
        print(f"  [PASS] Static Input Contract: '{result['input_name']}' {result['input_shape']}")
        print(f"  [PASS] Static Output Contracts: '{result['output_restored_name']}' {result['output_restored_shape']}, '{result['output_seg_name']}' {result['output_seg_shape']}")
        print(f"  [PASS] ONNX Runtime Numerical Parity: Restored Delta = {result['max_diff_restored']:.2e}, Seg Delta = {result['max_diff_seg']:.2e} (< 1e-4)")

    print("=" * 60)
    print("WORK PACKAGE 2A.1 ONNX CONTRACT VERIFICATION: 100% PASSED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_onnx_export_contract()
