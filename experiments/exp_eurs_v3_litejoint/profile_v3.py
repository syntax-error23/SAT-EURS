"""
Profile Parameters & FLOPs for EURSNetV3LiteJoint
Target: <= 1,000,000 parameters and <= 12.0 GFLOPs at 1024x1024
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))

from experiments.exp_eurs_v3_litejoint.model_v3 import EURSNetV3LiteJoint


def profile_v3():
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode='bounded', bound_scale=0.20)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    rest_params = sum(p.numel() for p in model.restoration_decoder.parameters())
    seg_params = sum(p.numel() for p in model.segmentation_decoder.parameters())

    print("=" * 70)
    print("           EURS-NET V3 LITEJOINT COMPLEXITY PROFILING")
    print("=" * 70)
    print(f"Total Parameters:        {total_params:,} (Limit: <= 1,000,000)")
    print(f"  - Shared Encoder:      {encoder_params:,} ({encoder_params/total_params*100:.2f}%)")
    print(f"  - Restoration Decoder: {rest_params:,} ({rest_params/total_params*100:.2f}%)")
    print(f"  - Segmentation Decoder:{seg_params:,} ({seg_params/total_params*100:.2f}%)")
    print(f"  - FWB Pre-Net:         {sum(p.numel() for p in model.fwb.parameters()):,}")

    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        restored, seg = model(x)

    print(f"\nForward Pass Verification:")
    print(f"  Input:    {list(x.shape)}")
    print(f"  Restored: {list(restored.shape)} | Range: [{restored.min():.3f}, {restored.max():.3f}]")
    print(f"  Seg:      {list(seg.shape)}")

    # Measure FLOPs
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            model(x)

    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None and evt.flops > 0)
    gflops = float(total_flops) / 1e9
    print(f"\nFLOP Profiling at 1024x1024:")
    print(f"  Measured: {gflops:.3f} GFLOPs (Limit: <= 12.0 GFLOPs)")
    print(f"  Parameter Headroom: {1_000_000 - total_params:,} parameters")
    print(f"  FLOP Headroom:      {12.0 - gflops:.3f} GFLOPs")
    print("=" * 70)

    assert total_params <= 1_000_000, f"PARAM BUDGET EXCEEDED: {total_params} > 1,000,000"
    assert gflops <= 12.0, f"FLOP BUDGET EXCEEDED: {gflops} > 12.0"
    print("RESULT: FULLY COMPLIANT WITH TRACK 2 BUDGET CAP!\n")

    import json
    out_dict = {
        "architecture": "EURSNetV3LiteJoint",
        "total_parameters": total_params,
        "param_limit": 1000000,
        "param_headroom": 1000000 - total_params,
        "encoder_params": encoder_params,
        "restoration_params": rest_params,
        "segmentation_params": seg_params,
        "fwb_params": 6,
        "gflops_1024x1024": gflops,
        "flop_limit_gflops": 12.0,
        "flop_headroom_gflops": 12.0 - gflops,
        "compliant": True
    }
    with open("experiments/exp_eurs_v3_litejoint/profile_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print("Saved profile results to experiments/exp_eurs_v3_litejoint/profile_results.json")
    return total_params, gflops


if __name__ == '__main__':
    profile_v3()
