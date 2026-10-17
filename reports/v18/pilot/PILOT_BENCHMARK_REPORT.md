# SAT-EURS Track 1 — V18 Isolated Pilot Benchmark Report

**Date:** 2026-10-17 03:04:54 UTC  
**Device:** NVIDIA GeForce RTX 5090 (`cuda`)  
**Fold:** 0 (16 train scenes, 4 held-out validation scenes)  
**Mock Mode:** False (Real Pretrained DINOv3 ViT-L + ConvNeXt-B)  

## 1. Measured Throughput & Resource Consumption

| Stage | Architecture | Train Samples | Train Time | Val Time | Throughput | Peak VRAM | Train Loss | Val Metric (Ep 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Refiner** | NAFNet Residual (1.2M) | 3200 | 178.0s | 32.9s | 18.0 img/s | 8.23 GB | 0.0917 | PSNR: 26.71 dB |
| **Stage 2: Segmenter** | DINOv3 ViT-L + ConvNeXt (401M) | 3200 | 423.5s | 62.6s | 7.6 img/s | 16.70 GB | 0.0000 | Q_seg: 0.0000 (mIoU: 0.5785, F1: 0.6624) |

## 2. Segmenter Epoch 1 Per-Class Metrics

| Class ID | Class Name | IoU | F1-Score | Precision | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | Background | 0.6688 | 0.8015 | 0.9476 | 0.6944 |
| 1 | Coral | 0.7023 | 0.8251 | 0.7122 | 0.9806 |
| 2 | Fish | 0.5215 | 0.6855 | 0.6274 | 0.7556 |
| 3 | Elasmobranchii | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | Macro-Fauna | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **Coverage Penalty:** 0.0 (UNVERIFIED_PROVISIONAL)

## 3. Full 5-Fold Training Projections (15 Epochs)

- **Stage 1 Refiner (5 folds x 15 epochs @ 210.9s/epoch):** ~4.39 hours
- **Stage 2 Segmenter (5 folds x 15 epochs @ 486.1s/epoch):** ~10.13 hours
- **Total Projected GPU Training Time:** ~14.52 hours
- **Peak VRAM Overall:** 16.70 GB (RTX 5090 Headroom: **15.14 GB**)

> [!NOTE] This benchmark ran strictly on Fold 0 for 1 epoch. Downstream OOF evaluation, Dev inference, and submission packaging were skipped to protect against incomplete packaging.
