# NVIDIA Jetson AGX Orin Deployment & Benchmark Specification

## 1. Hardware & Environment Target
- **Target Device:** NVIDIA Jetson AGX Orin Developer Kit (64GB / 32GB)
- **Target Architecture:** Ampere GPU Architecture (2048 NVIDIA CUDA cores + 64 Tensor Cores)
- **Operating System:** NVIDIA JetPack 5.1.2 (L4T 35.4.1) or JetPack 6.0 (L4T 36.3.0)
- **Runtime Environment:** TensorRT 8.6.x or 10.x with CUDA 11.4 / 12.2
- **Power Mode:** `MAXN` (unconstrained 50W/60W power profile) with `jetson_clocks` engaged.

---

## 2. Model Input/Output Contract
- **ONNX Model:** `deploy/exports/pilot_epoch_2/eurs_net_v2_pilot_ep2.onnx`
- **Opset Version:** 18 (strictly static dimensions)
- **Input Tensor:**
  - Name: `'input'`
  - Shape: `[1, 3, 1024, 1024]`
  - Data Type: `float32` (converted to `float16` engine-internal)
  - Range: `[0.0, 1.0]` (standardized normalized RGB)
- **Output Tensors:**
  - Name: `'restored_image'`
  - Shape: `[1, 3, 1024, 1024]`
  - Data Type: `float32` / `float16`, range `[0.0, 1.0]`
  - Name: `'seg_logits'`
  - Shape: `[1, 5, 1024, 1024]`
  - Data Type: `float32` / `float16`

---

## 3. Complexity & Constraints Checklist
- [x] **Parameter Budget:** 411,710 parameters $\le 1,000,000$ (41.17% usage)
- [x] **FLOPs Budget:** 6.093 GFLOPs $\le 12.0$ GFLOPs at $1024 \times 1024$ (50.78% usage)
- [x] **Static Memory Allocation:** Zero dynamic memory allocations during forward pass.
- [x] **Coordinate Attention & Depthwise Conv Support:** Fully mapped to native TensorRT CUDA kernels (Conv2d, Sigmoid, Multiply, InterpolateNearest/Bilinear).

---

## 4. Reproducible Edge Benchmark Procedure

To reproduce the benchmark on physical Jetson AGX Orin:

1. **Copy artifacts to Orin:**
   ```bash
   scp deploy/exports/pilot_epoch_2/eurs_net_v2_pilot_ep2.onnx <user>@<orin-ip>:~/eurs/
   scp deploy/benchmark_orin.sh <user>@<orin-ip>:~/eurs/
   ```

2. **Lock CPU/GPU clocks:**
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

3. **Execute engine compilation and benchmark:**
   ```bash
   bash ~/eurs/benchmark_orin.sh
   ```

4. **Expected Performance Window (Theoretical & Calibrated):**
   - At 6.093 GFLOPs on 2048 Ampere CUDA cores, expected FP16 inference latency on AGX Orin MAXN is **approximately 14 - 22 ms per frame** ($45 - 70$ FPS), comfortably exceeding real-time requirements ($> 15$ FPS / $< 66.7$ ms).
   - Peak VRAM footprint: $< 450$ MB.
