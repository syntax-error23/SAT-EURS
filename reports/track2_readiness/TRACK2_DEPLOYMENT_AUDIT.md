# SAT-EURS Track 2 — Edge Deployment & Complexity Budget Audit

**Auditor:** Senior Model Optimization & Embedded Edge Engineer  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Target Hardware:** NVIDIA Jetson AGX Orin Developer Kit (Ampere Architecture, 2048 CUDA Cores, 64 Tensor Cores)  
**Host Evaluation Hardware:** Intel Core Ultra 9 285K (24 cores), NVIDIA GeForce RTX 5090 (32 GB VRAM)  
**Audited Files:** `deploy/check_submission_flops.py`, `deploy/benchmark_orin.sh`, `deploy/benchmark_orin_procedure.md`, `deploy/export_calibrated_track2_onnx.py`  

---

## 1. Executive Summary & Verification Verdict

The Track 2 submission requirements demand strict adherence to static complexity budgets and single-engine ONNX deployment on edge hardware.

### Key Deployment Findings:
1. **Complexity Checker Rules Clarified:** The official complexity analysis script (`deploy/check_submission_flops.py`) computes GFLOPs **strictly for `Conv`, `Gemm`, and `MatMul` nodes** using the standard multiplier: $\text{GFLOPs} = 2 \cdot \text{MACs} / 10^9$. Elementwise operations (`Add`, `Mul`, `Relu`, `Clip`, `Sigmoid`, `Resize`, `Concat`, `Transpose`) contribute **zero GFLOPs** in the official checker.
2. **Current Candidates Safely Inside Budget:**
   - Prior Leaderboard Baseline (`eurs_net_v5_calibrated_track2.onnx`): **0.7683M parameters** ($\le 1.0\text{M}$) and **9.0500 GFLOPs** ($\le 12.0\text{ GFLOPs}$).
   - Candidate `eurs_edgesota_v2_final.onnx`: **0.3537M parameters** (64.6% headroom) and **2.2496 GFLOPs** (81.3% headroom).
   - Candidate `EXP-COMBINED-BEST.onnx`: **0.9589M parameters** (4.1% headroom) and **3.5666 GFLOPs** (70.3% headroom).
3. **Physical Hardware Disconnect (Host CPU vs. Jetson):**
   - **Critical Audit Reality:** All throughput and latency numbers reported in the repository to date (e.g., 29.4 ms, 37.8 ms, 105.6 ms) were measured on the host workstation's **Intel Core Ultra 9 CPU** using ONNX Runtime's `CPUExecutionProvider`.
   - **Zero physical tests have been run on NVIDIA Jetson AGX Orin.** All Orin latency claims (e.g. "10–14 ms on TensorRT") are purely theoretical extrapolations.
4. **Input/Output Interface Compliance:**
   - Both `eurs_net_v5_calibrated_track2.onnx` and `eurs_edgesota_v2_final.onnx` satisfy the strict `[1, 3, 1024, 1024]` single-input contract and dual-output contract (`restored_image`, `seg_logits`) with zero dynamic axes.

---

## 2. Official Complexity Budget & Counting Rules

The official checker (`deploy/check_submission_flops.py`, fetched from the competition organizers) governs acceptance:

### 2.1 Parameter Counting Rules
$$\text{Total Parameters} = \sum_{w \in \text{Initializers}} \text{size}(w)$$
- Counts all weights, biases, static batch-norm constants, embedding lookup tables, and frozen buffers present in the ONNX graph initializers.
- **Strict Cap:** $\text{Total Parameters} \le 1,000,000$ ($1.0\text{M}$).

### 2.2 FLOPs Counting Rules
Only nodes with op-type `Conv`, `Gemm`, or `MatMul` are evaluated:
1. **Convolution (`Conv`):**
   $$\text{MACs} = H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{out}} \cdot \left(\frac{C_{\text{in}}}{\text{groups}}\right) \cdot K_h \cdot K_w$$
   $$\text{GFLOPs} = \frac{2 \cdot \text{MACs}}{10^9}$$
2. **Matrix Multiplication (`Gemm` / `MatMul`):**
   $$\text{MACs} = M \cdot K \cdot N \implies \text{GFLOPs} = \frac{2 \cdot \text{MACs}}{10^9}$$
3. **Excluded Operators (Zero GFLOPs Assessed):**
   `BatchNormalization`, `Relu`, `Clip`, `Sigmoid`, `Concat`, `Add`, `Resize`, `Reshape`, `Transpose`, `Constant`, `Shape`, `Gather`, `Unsqueeze`, `Squeeze`, `Cast`, `Slice`, `Mul`, `Div`, `Sub`, `Identity`, `ConstantOfShape`, `Expand`.
- **Strict Cap:** $\text{Total Compute} \le 12.0000\text{ GFLOPs}$ evaluated at static input shape $[1, 3, 1024, 1024]$.

---

## 3. Candidate Complexity & Operator Breakdown

We ran `deploy/check_submission_flops.py` on all primary Track 2 models:

| Metric / Layer Type | Official Limit | `eurs_net_v5_calibrated` (#18699) | `eurs_edgesota_v2_final` | `EXP-COMBINED-BEST` |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameters (M)** | $\le 1.0000\text{ M}$ | **0.7683 M** | **0.3537 M** | **0.9589 M** |
| Parameter Headroom | — | 23.17% (231.7k params) | **64.63% (646.3k params)**| 4.11% (41.1k params) |
| **Total Compute (GFLOPs)** | $\le 12.0000\text{ GFLOPs}$ | **9.0500 GFLOPs** | **2.2496 GFLOPs** | **3.5666 GFLOPs** |
| Compute Headroom | — | 24.58% (2.95 GFLOPs) | **81.25% (9.75 GFLOPs)** | 70.28% (8.43 GFLOPs)|
| Conv Nodes | — | 135 nodes (9.050 GFLOPs) | 58 nodes (2.250 GFLOPs) | 68 nodes (3.567 GFLOPs) |
| Gemm / MatMul Nodes | — | 8 nodes (0.000 GFLOPs) | 4 nodes (0.000 GFLOPs) | 4 nodes (0.000 GFLOPs) |
| Elementwise / Activations | — | 103 nodes | 78 nodes | 86 nodes |
| Status Against Caps | — | **PASSED** | **PASSED** | **PASSED** |

> [!NOTE]
> **Key Architecture Insight:** Candidate `eurs_edgesota_v2_final` leaves **64.6% of parameter capacity and 81.3% of compute capacity unused**.
> This massive headroom demonstrates that the student was severely under-parameterized, directly explaining its struggle to represent rare foreground classes (Class 3 & 4) alongside complex coral textures.

---

## 4. Hardware Inference Measurements: Host vs. Embedded Edge

To preserve scientific rigor, host measurements must be strictly segregated from target hardware projections:

| Metric | Host Workstation (Measured) | Physical Jetson AGX Orin (Measured) | Physical Jetson AGX Orin (Theoretical Estimate) |
| :--- | :--- | :--- | :--- |
| **Processor / GPU** | Intel Core Ultra 9 285K / RTX 5090 | NVIDIA Jetson AGX Orin Dev Kit | Ampere GPU (2048 CUDA cores, 64 Tensor cores) |
| **Execution Provider** | ONNX Runtime `CPUExecutionProvider` | TensorRT FP16 / JetPack 5.1.2 | `trtexec --fp16 --explicitBatch` |
| **Power Profile** | Uncapped Desktop AC (125W–250W) | Not physically deployed | MAXN Mode (60W) + `jetson_clocks` |
| **EURS-Net V5 Latency** | **105.60 ms** (9.5 FPS) | **NOT MEASURED** | $\approx 22 - 30\text{ ms}$ (33–45 FPS) |
| **EdgeSOTA-V2 Latency** | **37.83 ms** (26.4 FPS) | **NOT MEASURED** | $\approx 10 - 15\text{ ms}$ (65–100 FPS) |
| **Peak Host Memory** | ~480 MB RAM | **NOT MEASURED** | $< 450\text{ MB}$ Unified VRAM |
| **Parity with PyTorch** | $\text{MAE} < 4 \times 10^{-6}$ (100% exact) | **NOT MEASURED** | Subject to FP16 TensorRT kernel rounding |

> [!WARNING]
> **Deployment Risk:** No team member has yet executed `deploy/benchmark_orin.sh` on an actual Jetson AGX Orin device.
> Any claims of "verified Orin deployment" in previous documents were speculative and must be updated to "ONNX Runtime CPU verified; Orin procedure scripted but unexecuted."

---

## 5. Standalone ONNX Input/Output Contract Verification

```
+---------------------------------------------------------------------------------------+
| INPUT TENSOR: 'input'                                                                 |
| Shape: [1, 3, 1024, 1024] | Data Type: Float32 | Value Range: [0.0, 1.0] (Normalized) |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
                      +-------------------------------------------+
                      |       STANDALONE ONNX EXECUTABLE GRAPH    |
                      |   Zero External Metadata / Zero Filenames |
                      |   Static Constant Memory Allocation       |
                      +---------------------+---------------------+
                                            |
                     +----------------------+----------------------+
                     |                                             |
                     v                                             v
+-----------------------------------------+   +-----------------------------------------+
| OUTPUT 1: 'restored_image'              |   | OUTPUT 2: 'seg_logits'                  |
| Shape: [1, 3, 1024, 1024]               |   | Shape: [1, 5, 1024, 1024]               |
| Data Type: Float32 | Range: [0.0, 1.0]  |   | Data Type: Float32 | Unbounded Logits   |
+-----------------------------------------+   +-----------------------------------------+
```

### 5.1 Verification Checklist:
- [x] Static batch dimension ($B=1$).
- [x] Static spatial resolution ($1024 \times 1024$).
- [x] Native operators supported in TensorRT: standard convolutions, depthwise convolutions, linear layers, bilinear resize with align_corners=False, clipping.
- [x] Hardswish eliminated: replaced with ONNX opset 17 compliant ReLU and Clip in MobileNetV3 encoder (`_replace_hardswish_with_onnx_safe`).
- [x] Submission archive verified: `deploy/eurs_edgesota_v2_track2_submission.zip` contains `model.onnx` at root and measures **2.67 MB** (well below the 1,024 MB competition limit).
