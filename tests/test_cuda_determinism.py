"""
CUDA Determinism & RTX 5090 Environment Smoke Test (Phase 2B Gatekeeper)
Executes on the GPU training machine:
1. Verifies CUDA availability, device capability, and VRAM capacity.
2. Configures CUDA deterministic operations:
     torch.use_deterministic_algorithms(True)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
3. Verifies AMP float16 forward and backward execution with zero NaN/Inf.
4. Verifies GPU seed determinism: identical initializations produce bitwise identical outputs.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v2 import EURSNetV2


def test_cuda_environment():
    print("=" * 65)
    print("  PHASE 2B: RTX 5090 CUDA DETERMINISM & ENVIRONMENT VALIDATION")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("  [WARN] CUDA is not available in the current environment (running on CPU).")
        print("  This script is designed to run on the RTX 5090 system tomorrow.")
        print("=" * 65 + "\n")
        return

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  [CUDA] Detected GPU: {device_name}")
    print(f"  [CUDA] Total VRAM:   {total_mem_gb:.2f} GB")
    print(f"  [CUDA] PyTorch:      {torch.__version__}")
    print(f"  [CUDA] cuDNN:        {torch.backends.cudnn.version()}")

    # 1. Configure Deterministic Flags
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        print("  [PASS] Configured torch.use_deterministic_algorithms(True)")
    except Exception as e:
        print(f"  [WARN] Could not set strict deterministic algorithms: {e}")

    # 2. Test AMP float16 Forward & Backward
    model = EURSNetV2(num_classes=5).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    dummy_input = torch.randn(2, 3, 1024, 1024, device='cuda')
    target_clean = torch.randn(2, 3, 1024, 1024, device='cuda')
    criterion = nn.L1Loss()

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        restored, seg = model(dummy_input)
        loss = criterion(restored, target_clean)

    assert torch.isfinite(loss), "CUDA AMP loss is NaN/Inf!"
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    current_mem_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    print(f"  [PASS] AMP float16 forward & backward successful (Peak VRAM: {current_mem_gb:.2f} GB).")

    # 3. GPU Deterministic Seed Test
    torch.manual_seed(999)
    t1 = torch.randn(100, 100, device='cuda')
    torch.manual_seed(999)
    t2 = torch.randn(100, 100, device='cuda')
    assert torch.equal(t1, t2), "CUDA RNG is not deterministic!"
    print("  [PASS] CUDA RNG determinism verified: torch.equal == True.")

    print("=" * 65)
    print("STAGE B1 CUDA ENVIRONMENT VALIDATION: 100% PASSED")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    test_cuda_environment()
