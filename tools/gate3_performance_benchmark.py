"""
Gate 3 — Performance Profiling & Hardware Benchmark Suite
EURS / MaCVi WACV 2027 Challenge (Tracks 1 & 2)
Workstation: NVIDIA RTX 5090 32GB | 64GB DDR5 | Intel Core Ultra 9 285K | Windows 11

Benchmarks:
1. DataLoader worker scaling (workers = 2, 4, 6, 8, pin_memory = True/False)
2. Physical batch size grid (B = 1, 2, 4, 8) with peak allocated/reserved VRAM
3. Precision comparison (FP32 baseline, TF32, AMP FP16, AMP BF16)
4. Memory layout & optimizer optimizations (cuDNN benchmark, channels-last, fused AdamW)
5. Fine-grained bottleneck breakdown (Data loading vs Host Transfer vs Forward vs Backward vs Optimizer)
"""

import os
import os.path as osp
import sys
import time
import json
import shutil
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from tools.train_track1_screening import ControlledTrack1Dataset, build_dinov2_model


def benchmark_gate3():
    print("=" * 80)
    print("   SAT-EURS GATE 3: PERFORMANCE PROFILING & HARDWARE BENCHMARK")
    print("   Workstation: NVIDIA RTX 5090 | 64GB DDR5 RAM | Windows 11")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA required for Gate 3 benchmarking!"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA Runtime: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
    total_vram = torch.cuda.get_device_properties(0).total_memory

    results = {}

    # -------------------------------------------------------------------------
    # PART 1: DataLoader Worker & Pinned Memory Profiling
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PART 1: DATALOADER WORKER & HOST-TO-DEVICE TRANSFER PROFILING")
    print("-" * 70)

    train_scenes = [
        "fish13", "fish14", "fish15", "fish17", "fish18", "fish2", "fish20",
        "fish21", "fish4", "fish5", "fish7", "fish9", "ray1", "sharkFish",
        "starfish2", "turtles1"
    ]
    ds = ControlledTrack1Dataset(
        scenes=train_scenes,
        data_root="data/cached_1024",
        crop_size=518,
        is_train=True,
        use_rsuigm=False
    )
    print(f"Dataset total items: {len(ds)} across {len(train_scenes)} scenes")

    worker_candidates = [2, 4, 6, 8]
    pin_options = [True] # Benchmark pinned memory scaling

    dataloader_results = {}
    batches_to_profile = 25
    bench_batch_size = 4

    for num_w in worker_candidates:
        for pin_m in pin_options:
            cfg_name = f"workers_{num_w}_pin_{pin_m}"
            loader = DataLoader(
                ds,
                batch_size=bench_batch_size,
                shuffle=True,
                num_workers=num_w,
                pin_memory=pin_m,
                persistent_workers=(num_w > 0)
            )

            # Warm-up 2 batches
            iterator = iter(loader)
            _ = next(iterator)
            _ = next(iterator)

            # Profile timed iteration
            proc = psutil.Process()
            ram_before_mb = proc.memory_info().rss / (1024**2)

            t0 = time.perf_counter()
            t_transfer_total = 0.0
            samples_loaded = 0

            for i in range(batches_to_profile):
                t_fetch_start = time.perf_counter()
                batch = next(iterator)
                t_fetch_end = time.perf_counter()

                x_cpu, y_cpu = batch[0], batch[1]
                t_xfer_start = time.perf_counter()
                x_gpu = x_cpu.to(device, non_blocking=pin_m)
                y_gpu = y_cpu.to(device, non_blocking=pin_m)
                torch.cuda.synchronize()
                t_xfer_end = time.perf_counter()

                t_transfer_total += (t_xfer_end - t_xfer_start)
                samples_loaded += x_cpu.shape[0]

            elapsed_total = time.perf_counter() - t0
            ram_after_mb = proc.memory_info().rss / (1024**2)
            fps = samples_loaded / elapsed_total
            transfer_ms_per_batch = (t_transfer_total / batches_to_profile) * 1000.0

            print(f"  {cfg_name}: {fps:.1f} samples/sec | Fetch+Transfer: {elapsed_total*1000/batches_to_profile:.1f} ms/batch | Host->GPU: {transfer_ms_per_batch:.2f} ms/batch | Process RAM: {ram_after_mb:.1f} MB (Delta: +{ram_after_mb - ram_before_mb:.1f} MB)")

            dataloader_results[cfg_name] = {
                "num_workers": num_w,
                "pin_memory": pin_m,
                "throughput_samples_per_sec": fps,
                "ms_per_batch": elapsed_total * 1000 / batches_to_profile,
                "transfer_ms_per_batch": transfer_ms_per_batch,
                "process_ram_mb": ram_after_mb
            }

    results["dataloader_benchmarks"] = dataloader_results

    # -------------------------------------------------------------------------
    # PART 2: Physical Batch Size Grid on RTX 5090 (DINOv2-Large Foundation)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PART 2: PHYSICAL BATCH SIZE GRID (DINOv2-Large ~308M parameters)")
    print("-" * 70)

    model = build_dinov2_model(num_classes=5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_candidates = [1, 2, 4, 8]
    resolutions = [518] # Standard training patch size

    batch_results = {}
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for b in batch_candidates:
        cfg_name = f"B_{b}_res_518"
        print(f"\n  Testing Physical Batch Size {b} (Resolution 518x518)...")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Synthetic batch on device
            x_test = torch.rand(b, 3, 518, 518, device=device)
            y_test = torch.randint(0, 5, (b, 518, 518), device=device)
            x_norm = (x_test - mean_im) / std_im

            # Warm-up 2 steps
            for _ in range(2):
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_norm)
                    loss = loss_fn(logits.float(), y_test)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            steps = 10

            for _ in range(steps):
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_norm)
                    loss = loss_fn(logits.float(), y_test)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            step_time_ms = (elapsed / steps) * 1000.0
            throughput = (steps * b) / elapsed
            peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
            peak_res_gib = torch.cuda.max_memory_reserved() / (1024**3)
            free_vram_gib = (total_vram - torch.cuda.max_memory_reserved()) / (1024**3)

            print(f"    [SUCCESS] Step Time: {step_time_ms:.1f} ms | Throughput: {throughput:.2f} samples/sec")
            print(f"    Peak VRAM: Allocated = {peak_alloc_gib:.2f} GiB | Reserved = {peak_res_gib:.2f} GiB | Free Reserve = {free_vram_gib:.2f} GiB")

            batch_results[cfg_name] = {
                "batch_size": b,
                "step_time_ms": step_time_ms,
                "throughput_samples_per_sec": throughput,
                "peak_allocated_gib": peak_alloc_gib,
                "peak_reserved_gib": peak_res_gib,
                "free_reserve_gib": free_vram_gib,
                "status": "PASSED"
            }

        except torch.cuda.OutOfMemoryError as e:
            print(f"    [OOM] Batch size {b} exceeded VRAM limits!")
            torch.cuda.empty_cache()
            batch_results[cfg_name] = {"batch_size": b, "status": "OOM"}
            break

    results["batch_size_benchmarks"] = batch_results

    # -------------------------------------------------------------------------
    # PART 3: Precision & Acceleration Benchmarking (FP32 vs TF32 vs FP16 vs BF16)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PART 3: PRECISION & HARDWARE ACCELERATION MODES (Fixed Batch Size 4)")
    print("-" * 70)

    fixed_b = 4
    x_test = torch.rand(fixed_b, 3, 518, 518, device=device)
    y_test = torch.randint(0, 5, (fixed_b, 518, 518), device=device)
    x_norm = (x_test - mean_im) / std_im

    precision_configs = [
        ("FP32_Baseline", False, False, None),
        ("TF32_Matmul", True, True, None),
        ("AMP_FP16", True, True, torch.float16),
        ("AMP_BF16", True, True, torch.bfloat16),
    ]

    precision_results = {}
    baseline_time = None

    for name, tf32_matmul, tf32_cudnn, amp_dtype in precision_configs:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()
        # Warmup
        for _ in range(2):
            if amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x_norm)
                    loss = loss_fn(logits.float(), y_test)
            else:
                logits = model(x_norm)
                loss = loss_fn(logits, y_test)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = 10
        for _ in range(steps):
            optimizer.zero_grad()
            if amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x_norm)
                    loss = loss_fn(logits.float(), y_test)
            else:
                logits = model(x_norm)
                loss = loss_fn(logits, y_test)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_ms = (elapsed / steps) * 1000.0
        throughput = (steps * fixed_b) / elapsed
        peak_gib = torch.cuda.max_memory_reserved() / (1024**3)

        if baseline_time is None:
            baseline_time = step_ms
            speedup = 1.00
        else:
            speedup = baseline_time / step_ms

        print(f"  {name:15s}: {step_ms:.1f} ms/step | {throughput:.2f} samples/sec | Speedup: {speedup:.2f}x | Peak VRAM: {peak_gib:.2f} GiB | Loss: {loss.item():.4f}")

        precision_results[name] = {
            "step_ms": step_ms,
            "throughput_fps": throughput,
            "speedup": speedup,
            "peak_vram_gib": peak_gib,
            "loss_finite": bool(torch.isfinite(loss).all().item())
        }

    results["precision_benchmarks"] = precision_results

    # -------------------------------------------------------------------------
    # PART 4: Kernel & Memory Optimizations (cuDNN Benchmark, Channels-Last, Fused AdamW)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PART 4: KERNEL & MEMORY LAYOUT OPTIMIZATIONS (BF16, Batch Size 4)")
    print("-" * 70)

    # Enable TF32 & BF16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. cuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_norm)
            loss = loss_fn(logits.float(), y_test)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    cudnn_bench_ms = ((time.perf_counter() - t0) / 10) * 1000.0
    print(f"  cuDNN Benchmark Enabled: {cudnn_bench_ms:.1f} ms/step (Speedup vs unbenchmarked BF16: {precision_results['AMP_BF16']['step_ms'] / cudnn_bench_ms:.2f}x)")

    # 2. Fused AdamW
    fused_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fused_opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_norm)
            loss = loss_fn(logits.float(), y_test)
        loss.backward()
        fused_opt.step()
    torch.cuda.synchronize()
    fused_opt_ms = ((time.perf_counter() - t0) / 10) * 1000.0
    print(f"  Fused AdamW Optimizer:   {fused_opt_ms:.1f} ms/step (Speedup vs unbenchmarked BF16: {precision_results['AMP_BF16']['step_ms'] / fused_opt_ms:.2f}x)")

    results["optimizations"] = {
        "cudnn_benchmark_ms": cudnn_bench_ms,
        "fused_adamw_ms": fused_opt_ms
    }

    # -------------------------------------------------------------------------
    # PART 5: Fine-Grained Step Time Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PART 5: STEP TIME BREAKDOWN & BOTTLENECK ATTRIBUTION")
    print("-" * 70)

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    iterator = iter(loader)

    # Synchronized timing
    t_data_list, t_h2d_list, t_fwd_list, t_bwd_list, t_opt_list = [], [], [], [], []

    for _ in range(15):
        torch.cuda.synchronize()
        t_s0 = time.perf_counter()
        batch = next(iterator)
        torch.cuda.synchronize()
        t_s1 = time.perf_counter()

        x_cpu, y_cpu = batch[0], batch[1]
        x_g = x_cpu.to(device, non_blocking=True)
        y_g = y_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_s2 = time.perf_counter()

        x_n = (x_g - mean_im) / std_im
        fused_opt.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x_n)
            l = loss_fn(out.float(), y_g)
        torch.cuda.synchronize()
        t_s3 = time.perf_counter()

        l.backward()
        torch.cuda.synchronize()
        t_s4 = time.perf_counter()

        fused_opt.step()
        torch.cuda.synchronize()
        t_s5 = time.perf_counter()

        t_data_list.append((t_s1 - t_s0) * 1000)
        t_h2d_list.append((t_s2 - t_s1) * 1000)
        t_fwd_list.append((t_s3 - t_s2) * 1000)
        t_bwd_list.append((t_s4 - t_s3) * 1000)
        t_opt_list.append((t_s5 - t_s4) * 1000)

    # Discard first 2 steps as warmup
    m_data = np.mean(t_data_list[2:])
    m_h2d = np.mean(t_h2d_list[2:])
    m_fwd = np.mean(t_fwd_list[2:])
    m_bwd = np.mean(t_bwd_list[2:])
    m_opt = np.mean(t_opt_list[2:])
    m_total = m_data + m_h2d + m_fwd + m_bwd + m_opt

    print(f"  Step Time Breakdown (Batch Size = 4, Resolution = 518x518, Mean over 13 steps):")
    print(f"    1. CPU Data Fetch & Preprocess: {m_data:6.2f} ms ({m_data / m_total * 100:4.1f}%)")
    print(f"    2. Host-to-Device Transfer:     {m_h2d:6.2f} ms ({m_h2d / m_total * 100:4.1f}%)")
    print(f"    3. Model Forward Pass:          {m_fwd:6.2f} ms ({m_fwd / m_total * 100:4.1f}%)")
    print(f"    4. Loss & Backward Pass:        {m_bwd:6.2f} ms ({m_bwd / m_total * 100:4.1f}%)")
    print(f"    5. Optimizer Step (Fused):      {m_opt:6.2f} ms ({m_opt / m_total * 100:4.1f}%)")
    print(f"    -------------------------------------------------------")
    print(f"    Total Iteration Step Time:      {m_total:6.2f} ms (100.0%) | {4 * 1000.0 / m_total:5.1f} samples/sec")

    results["breakdown"] = {
        "data_fetch_ms": float(m_data),
        "h2d_transfer_ms": float(m_h2d),
        "fwd_ms": float(m_fwd),
        "bwd_ms": float(m_bwd),
        "opt_ms": float(m_opt),
        "total_ms": float(m_total),
        "throughput_fps": float(4 * 1000.0 / m_total)
    }

    # Save results
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/gate3_performance_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("   GATE 3 BENCHMARK COMPLETE: RESULTS RECORDED IN tmp/gate3_performance_results.json")
    print("=" * 80)


if __name__ == '__main__':
    benchmark_gate3()
