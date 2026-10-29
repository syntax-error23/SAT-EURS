# SAT-EURS Track 1 — Phase 1 Forensic Diagnostic: Scale Mismatch Audit

**Campaign:** SAT-EURS Track 1 Phase 1 Forensic Diagnostic  
**Date:** October 24, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** MaCVi @ WACV 2027 EURS Track 1 Testing Phase  
**Artifact Manifest:** [`reports/v28/phase1_diagnostic/scale_mismatch_summary.json`](file:///c:/Projects/SAT-EURS/reports/v28/phase1_diagnostic/scale_mismatch_summary.json)

---

## 1. Executive Summary

This diagnostic evaluates the spatial scale relationship between training and inference in the SAT-EURS segmentation pipeline.

We investigated whether the mismatch between training crop resolutions ($384 \to 512$ zoomed crops centered on rare objects) and canonical inference ($1024 \times 1024$ full scene) causes small-object degradation, and whether sliding-window tiled inference (512-tile sliding window with cosine blending) recovers small-object IoU.

Our empirical findings:
1. **Tiled Inference Yields Negligible Gain (+0.0004 $Q_{\text{seg}}$):** Directly running $512 \times 512$ sliding-window inference with 50% overlap and Hanning blending produced a negligible delta of $+0.0003$ mIoU and $+0.0007$ Fish IoU.
2. **The Resolution Bottleneck is Upstream (Image Downsampling):** Downsampling $7\text{K}$ native images (`test19` and `test20`) by $7\times$ to fit the $1024$ max-side contract reduces small fish ($100\text{ px}$) to sub-patch dimensions ($14\text{ px}$), smaller than the DINOv3 ViT-L $16 \times 16$ patch size.
3. **The In-Sample vs Out-of-Fold Generalization Gulf:** On scenes included in training, Full-20 achieves $Q_{\text{seg}} = 0.94 - 0.98$. On held-out unseen scenes, performance drops to $Q_{\text{seg}} \approx 0.458$, proving that generalization to novel undersea geometries and water scattering is the primary constraint.

---

## 2. Training vs Inference Pipeline Geometry Audit

| Parameter | Training Regime (`data/v27/dataset.py`) | Canonical Inference (`tools/v27/run_v27_test_pipeline.py`) | Discrepancy Factor |
| :--- | :--- | :--- | :---: |
| **Input Source** | `data/cached_1024/` | `data/test/degraded/` (up to $7203 \times 4804$) | $1\times$ to $7\times$ downsampling |
| **Spatial Crop Size** | Random $cs \in [384, 512]$ px | Full-Scene Resized to $\max(W, H) = 1024$ | Up to $10\times$ area ratio |
| **Crop Centering** | 70% weighted toward rare fauna | Canonical center/unpadded scene | Severe spatial bias |
| **Effective Zoom** | $1.0\times$ to $1.33\times$ zoom boost | $0.14\times$ to $0.26\times$ shrink on 4K/7K test images | **$5\times$ to $9\times$ relative scale gap** |
| **Patch Resolution** | ViT-L/16 patch covers $12 \times 12$ effective scene px | ViT-L/16 patch covers $112 \times 112$ native scene px | **$9.3\times$ spatial blur** |

---

## 3. Empirical Sliding-Window / Tiled Inference Benchmark

Evaluated across validation conditions comparing canonical 1024 full-image inference vs 512-tile sliding-window inference:

| Scene | Degradation Condition | Full 1024 $Q_{\text{seg}}$ | Tiled 512 $Q_{\text{seg}}$ | Delta $Q_{\text{seg}}$ | Fish IoU (Full $\to$ Tiled) | Latency Full vs Tiled |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fish1` | $WT=1, Depth=10\text{m}$ | 0.9801 | 0.9806 | +0.0005 | 0.950 $\to$ 0.951 | 0.04s vs 0.38s |
| `fish1` | $WT=3, Depth=30\text{m}$ | 0.9799 | 0.9806 | +0.0007 | 0.949 $\to$ 0.951 | 0.04s vs 0.38s |
| `fish1` | $WT=5, Depth=60\text{m}$ | 0.9795 | 0.9799 | +0.0004 | 0.948 $\to$ 0.949 | 0.04s vs 0.38s |
| `sharkFish`| $WT=1, Depth=10\text{m}$ | 0.9461 | 0.9462 | +0.0001 | 0.865 $\to$ 0.866 | 0.04s vs 0.38s |
| `sharkFish`| $WT=3, Depth=30\text{m}$ | 0.9466 | 0.9464 | -0.0002 | 0.867 $\to$ 0.867 | 0.04s vs 0.38s |
| `sharkFish`| $WT=5, Depth=60\text{m}$ | 0.9325 | 0.9326 | +0.0001 | 0.828 $\to$ 0.828 | 0.04s vs 0.38s |
| `seahorse1`| $WT=1, Depth=10\text{m}$ | 0.9799 | 0.9806 | +0.0007 | 1.000 $\to$ 1.000 | 0.04s vs 0.38s |
| `seahorse1`| $WT=3, Depth=30\text{m}$ | 0.9802 | 0.9808 | +0.0006 | 1.000 $\to$ 1.000 | 0.04s vs 0.38s |
| `seahorse1`| $WT=5, Depth=60\text{m}$ | 0.9796 | 0.9803 | +0.0007 | 1.000 $\to$ 1.000 | 0.04s vs 0.38s |
| **Mean** | — | **0.9654** | **0.9658** | **+0.0004** | **+0.0007** | **$9.5\times$ slower** |

---

## 4. Key Takeaways for Model Design

1. **Sliding-window tiling at 1024 scale is an ineffective intervention:**
   Because images were already pre-downsampled to $\max(W, H) = 1024$ before tiling, tiling does not recover native high-frequency resolution; it only chops the low-resolution image into smaller patches while losing broader context.
2. **Native-Scale Patch Processing is the True Solution:**
   To genuinely resolve small fish in $7\text{K}$ test scenes (`test19` and `test20`), the model must crop patches directly from the **native 7K resolution** prior to downsampling, or employ multi-scale feature pyramids that process high-resolution crops alongside the global context.
