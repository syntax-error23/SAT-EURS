# SAT-EURS GPU Memory & Adaptive Resource Management Policy
**Target Hardware:** NVIDIA GeForce RTX 5090 32 GB VRAM | 64 GB DDR5 RAM | Intel Core Ultra 9 285K (24 cores)  
**Operating System:** Windows 11 64-bit | PyTorch 2.11.0+cu128 | CUDA 12.8 / Driver 591.55  
**Version:** 1.0.0 (Gate 1 Baseline Locked)  

---

## 1. Operating Budget & Guardrails

The goal of this policy is to maximize useful throughput, training speed, and validation rigor without causing CUDA Out-of-Memory (OOM), host RAM thrashing, process instability, or corruption of verified artifacts.

| Resource | Normal Operational Target | Warning Threshold & Automated Mitigation | Hard Safety Ceiling |
|---|---|---|---|
| **GPU VRAM** | 24.0 – 27.0 GiB under sustained training | > 27.0 GiB: freeze scaling; drop batch size or resolution | 28.5 GiB (Never exceed) |
| **GPU Free VRAM Reserve** | Retain $\ge$ 3.5 – 4.0 GiB | < 3.5 GiB: trigger immediate garbage collection & cache flush | < 2.0 GiB: abort run immediately |
| **Process RAM (RSS)** | < 35.0 GB | > 40.0 GB: reduce DataLoader workers, prefetch depth, release numpy caches | 45.0 GB |
| **System Available RAM** | Maintain $\ge$ 14.0 – 16.0 GB | < 12.0 GB: halt memory-intensive tasks, disable large arrays | < 10.0 GB |
| **System Free Disk (C:)** | Maintain $\ge$ 200 GB | < 50 GB: halt checkpointing and output generation | < 25 GB |
| **GPU Training Concurrency** | Exactly 1 heavy training job | Reject multi-process GPU training attempts | 1 GPU job |
| **Large Inference Concurrency** | Exactly 1 heavy evaluation/TTA job | Sequential execution of fold models / TTA scales | 1 Inference job |

---

## 2. Two Distinct Operational Modes

### Mode 1: Stable Benchmark & Debugging
* **When Used:** Architectural unit tests, loss testing, new checkpoint smoke tests, export validation, single-image inference checks, memory profiling.
* **Configuration:**
  - Physical Batch Size: `1` or `2`
  - Input Resolution: modest crop (`256x256` or `518x518`)
  - DataLoader Workers: `2`
  - Precision: FP32 or AMP FP16/BF16 (after forward/backward stability check)
  - Objective: Deterministic execution, fast feedback loops, zero risk of OOM.

### Mode 2: Optimized Training & Production Inference
* **When Used:** Full training epochs, 5-fold cross-validation, official test-set evaluation, multi-scale TTA.
* **Configuration:**
  - Physical Batch Size: Benchmarked empirically via automated grid (`1, 2, 4, 8`) until highest safe throughput (samples/sec) is attained within 24–27 GiB.
  - Effective Batch Size: Expanded via gradient accumulation if required by optimizer dynamics.
  - Precision: `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` or `torch.float16`.
  - Matmul Acceleration: `torch.backends.cuda.matmul.allow_tf32 = True` enabled for Blackwell architecture.
  - cuDNN Benchmarking: `torch.backends.cudnn.benchmark = True` enabled for fixed-resolution training.
  - DataLoader Workers: `4` to `8` (validated against RAM consumption, never duplicating full datasets in memory).
  - Pin Memory: `pin_memory=True` enabled with non-blocking device transfers.

---

## 3. High-Resolution Probability Map & Data Handling Guardrails

Under 5-class semantic segmentation at $1024 \times 1024$:
$$\text{Size per image} = 5 \times 1024 \times 1024 \times 2\text{ bytes (FP16)} = 10,485,760\text{ bytes} \approx 10.49\text{ MB}$$
- For $4,000$ training images: $4,000 \times 10.49\text{ MB} = \mathbf{41.94\text{ GB}}$ in raw FP16 tensors.
- In FP32: $\mathbf{83.88\text{ GB}}$, which exceeds total system RAM (64 GB) and would instantly crash Windows!

### Mandatory Data Handling Rules:
1. **Never accumulate full-dataset probability maps in host RAM.**
2. **Streaming & Incremental Aggregation:** Predictions must be evaluated per-image or per-chunk using streaming confusion matrix accumulators (`SegmentationEvaluator`).
3. **Sequential Model Loading:** When evaluating multi-model ensembles or 5-fold models, models must be loaded sequentially, evaluated on the batch/stream, and deleted with explicit `del model; torch.cuda.empty_cache()` before loading the next candidate if VRAM budget exceeds 24 GiB.
4. **Disk-Backed Memory Mapping:** If intermediate logits must be stored for calibration or blending, use `np.memmap` with validated disk space reservations.

---

## 4. Preflight Diagnostics Checklist

Before launching any training or heavy inference script, execute the preflight check:

```python
import torch, psutil, shutil

def preflight_check(required_vram_gib=20.0, required_ram_gb=15.0, required_disk_gb=30.0):
    assert torch.cuda.is_available(), "CUDA is not available!"
    free_vram, total_vram = torch.cuda.mem_get_info()
    free_vram_gib = free_vram / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    free_disk_gb = shutil.disk_usage("C:\\").free / (1024**3)

    print(f"[PREFLIGHT] Free VRAM: {free_vram_gib:.2f} GiB (Required: {required_vram_gib} GiB)")
    print(f"[PREFLIGHT] Available RAM: {available_ram_gb:.2f} GB (Required: {required_ram_gb} GB)")
    print(f"[PREFLIGHT] Free Disk: {free_disk_gb:.2f} GB (Required: {required_disk_gb} GB)")

    assert free_vram_gib >= required_vram_gib, f"Insufficient VRAM headroom: {free_vram_gib:.2f} GiB"
    assert available_ram_gb >= required_ram_gb, f"Insufficient system RAM headroom: {available_ram_gb:.2f} GB"
    assert free_disk_gb >= required_disk_gb, f"Insufficient disk space: {free_disk_gb:.2f} GB"
    print("[PREFLIGHT] All resource budgets PASSED.")
```

---

## 5. Automated Batch Size & Worker Benchmarking Protocol

Never guess the batch size or worker count. Apply this measured protocol:

1. **Step 1:** Initialize model and optimizer in designated precision (BF16/FP16).
2. **Step 2:** Warm up GPU for 3 steps with physical batch size $B \in \{1, 2, 4, 8\}$.
3. **Step 3:** Profile 10 full forward + backward + optimizer steps with `torch.cuda.synchronize()`.
4. **Step 4:** Record:
   - Peak allocated VRAM (`torch.cuda.max_memory_allocated() / (1024**3)`)
   - Peak reserved VRAM (`torch.cuda.max_memory_reserved() / (1024**3)`)
   - Step time (ms) and throughput (samples/sec)
   - Finite loss assertion (`torch.isfinite(loss).all()`)
5. **Step 5:** Halt search immediately if peak reserved VRAM exceeds $27.0\text{ GiB}$.
6. **Step 6:** Select the configuration maximizing samples/sec while maintaining $\ge 4.0\text{ GiB}$ free VRAM safety buffer.

---

## 6. OOM & Failure Recovery Protocol

If an unexpected CUDA OOM occurs:
1. **Trap the Exception:** Catch `torch.cuda.OutOfMemoryError` cleanly without terminating the Python session ungracefully.
2. **Release References:** Delete local activation tensors, output dictionaries, and loss objects.
3. **Cache Invalidation:** Execute `gc.collect()` followed by `torch.cuda.empty_cache()`.
4. **Fallback Backoff:** Step down physical batch size by 50% (e.g. $4 \to 2$ or $2 \to 1$), enable gradient checkpointing, and retry once.
5. **Logging:** Log exact input shape, batch size, model architecture, and peak memory state into `EXPERIMENT_LEDGER.csv` under status `REJECT_OOM`.
6. **Artifact Preservation:** Ensure previously saved checkpoints are never touched or corrupted during an OOM event.
