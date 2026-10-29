"""
Phase 3 — NAFNet Batch Size & Memory Benchmark Suite
MaCVi @ WACV 2027 Challenge — SAT-EURS Track 1
Workstation: NVIDIA RTX 5090 (32GB VRAM) | Windows 11
"""

import sys
sys.path.insert(0, ".")

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nafnet import build_nafnet_restorer

def run_batch_benchmark():
    print("=" * 80)
    print("PHASE 3: NAFNET BATCH SIZE & THROUGHPUT BENCHMARK (RTX 5090)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    free_init, total_mem = torch.cuda.mem_get_info(0)
    print(f"Initial Free VRAM: {free_init / (1024**3):.2f} GiB / {total_mem / (1024**3):.2f} GiB")

    crop_size = 512
    candidates = [2, 4, 8]
    results = []

    for b in candidates:
        print(f"\n--- Testing Physical Batch Size B = {b} (Crop: {crop_size}x{crop_size}, BF16) ---")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = build_nafnet_restorer(width=32).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

        # Warmup steps
        x_warm = torch.randn(b, 3, crop_size, crop_size, device=device)
        y_warm = torch.randn(b, 3, crop_size, crop_size, device=device)
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x_warm)
                loss = F.l1_loss(pred, y_warm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Benchmark 15 measured forward/backward steps
        step_times = []
        torch.cuda.synchronize()
        for step in range(15):
            x = torch.randn(b, 3, crop_size, crop_size, device=device)
            y = torch.randn(b, 3, crop_size, crop_size, device=device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
                loss = F.l1_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        avg_step_ms = (sum(step_times) / len(step_times)) * 1000.0
        fps = (b / (avg_step_ms / 1000.0))
        peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
        peak_res_gib = torch.cuda.max_memory_reserved() / (1024**3)
        headroom_gib = (total_mem / (1024**3)) - peak_res_gib

        print(f"  Step Time:       {avg_step_ms:.2f} ms")
        print(f"  Throughput:      {fps:.2f} samples/sec")
        print(f"  Peak Allocated:  {peak_alloc_gib:.2f} GiB")
        print(f"  Peak Reserved:   {peak_res_gib:.2f} GiB")
        print(f"  Free Headroom:   {headroom_gib:.2f} GiB")

        del model, optimizer, x, y, pred, loss
        torch.cuda.empty_cache()

        results.append({
            "batch_size": b,
            "step_ms": round(avg_step_ms, 2),
            "throughput_fps": round(fps, 2),
            "peak_vram_gib": round(peak_res_gib, 2),
            "headroom_gib": round(headroom_gib, 2),
            "status": "PASS_SAFE" if headroom_gib >= 10.0 else "WARNING"
        })

    print("\n" + "=" * 80)
    print("BATCH BENCHMARK SUMMARY (RTX 5090)")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'Step Time (ms)':<16} | {'Throughput (FPS)':<18} | {'Peak VRAM (GiB)':<16} | {'Headroom (GiB)':<16} | {'Status':<10}")
    print("-" * 96)
    for r in results:
        print(f"{r['batch_size']:<12} | {r['step_ms']:<16.2f} | {r['throughput_fps']:<18.2f} | {r['peak_vram_gib']:<16.2f} | {r['headroom_gib']:<16.2f} | {r['status']:<10}")
    print("=" * 80)

if __name__ == "__main__":
    run_batch_benchmark()
