# SAT-EURS Baseline Manifest & Artifact Provenance Report
## Gate 1 Comprehensive Environmental & Checkpoint Inventory
**Date:** October 16, 2026 | Local Time: 13:48 IST  
**Workstation:** Windows 11 Pro 64-bit | Intel Core Ultra 9 285K | NVIDIA GeForce RTX 5090 32 GB VRAM  
**Repository State:** Branch `track1-v15` | Head commit `c4732e4`  

---

## 1. Verified Hardware & Software Environment

| Component | Verified Specification | Audit Command / Method | Status |
|---|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 5090 (Blackwell Architecture) | `nvidia-smi` / `torch.cuda.get_device_name(0)` | VERIFIED |
| **GPU Total VRAM** | 32,607 MiB (31.84 GiB / 34.19 GB) | `torch.cuda.mem_get_info(0)` | VERIFIED |
| **GPU Usable Free VRAM** | 30.21 GiB (at idle, ~715 MiB desktop use) | `torch.cuda.mem_get_info(0)` | VERIFIED |
| **NVIDIA Driver** | 591.55 | `nvidia-smi` | VERIFIED |
| **CUDA Runtime** | 12.8 (PyTorch compiled runtime) / 13.1 (Driver) | `torch.version.cuda` | VERIFIED |
| **cuDNN Version** | 9.19.00 (`91900`) | `torch.backends.cudnn.version()` | VERIFIED |
| **Compute Capability** | `(12, 0)` | `torch.cuda.get_device_capability(0)` | VERIFIED |
| **BF16 Support** | Supported (`True`) | `torch.cuda.is_bf16_supported()` | VERIFIED |
| **FP16 Support** | Supported (`True`) | PyTorch Native | VERIFIED |
| **TF32 Support** | Supported (`matmul: True`, `cudnn: True` configured) | `torch.backends.cuda.matmul.allow_tf32` | VERIFIED |
| **System RAM** | 64 GB DDR5 (63.43 GB visible, 51.83 GB free) | `psutil.virtual_memory()` / WMI | VERIFIED |
| **CPU Configuration** | Intel Core Ultra 9 285K (24 cores / 24 logical) | WMI `Win32_Processor` | VERIFIED |
| **Disk Storage (C:)** | 709.4 GB Total, **660.7 GB Free** (70.7% available) | `psutil.disk_usage('C:\\')` | VERIFIED |
| **Operating System** | Windows 11 Build 26200 | `platform.platform()` | VERIFIED |
| **Python Environment** | Python 3.11.9 (`C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe`) | `sys.version` | VERIFIED |
| **PyTorch Version** | 2.11.0+cu128 | `torch.__version__` | VERIFIED |
| **Torchvision** | 0.26.0+cu128 | `torchvision.__version__` | VERIFIED |
| **Timm** | 1.0.30 | `timm.__version__` | VERIFIED |
| **SMP (SegModelsPytorch)**| 0.5.0 | `segmentation_models_pytorch.__version__`| VERIFIED |
| **ONNX Runtime** | 1.31.0 | `onnxruntime.__version__` | VERIFIED |
| **OpenCV** | 5.0.0.93 | `cv2.__version__` | VERIFIED |
| **Ptflops** | 0.7.5 | `ptflops.__version__` | VERIFIED |

---

## 2. Dataset Structure & Manifest

The dataset consists of high-resolution paired underwater images and five discrete semantic classes:
1. `0: Background`
2. `1: Coral / Substrate`
3. `2: Fish`
4. `3: Elasmobranch (Ray / Shark)`
5. `4: Other (Starfish / Seahorse / Turtle)`

| Dataset Split | Path | Image Dimensions | Contents / Quantity | Integrity Notes |
|---|---|---|---|---|
| **Clean Training Scenes** | `data/train/clean` (and `data/clean`) | $3000\times 4000$ to $4567\times 6870$ | 20 PNG images | Verified 20 unique clean underwater scenes |
| **Training Semantic Masks**| `data/train/masks` (and `data/masks`) | Identical to clean scenes | 20 PNG images | Verified discrete integer classes $\{0,1,2,3,4\}$ |
| **Degraded Training Images**| `data/train/degraded` (and `data/degraded`)| Identical to clean scenes | 20 folders $\times$ 200 images = **4,000 PNGs** | 10 water types $\times$ 20 depths per scene |
| **Cached $1024\times 1024$**| `data/cached_1024/` | Exactly $1024\times 1024$ | 20 clean, 20 masks, **4,000 degraded** | High-speed pre-resized anti-aliased dataset |
| **Dev Evaluation Set** | `data/dev/degraded` | $3000\times 4000$ to $4000\times 3000$ | 6 folders $\times$ 200 images = **1,200 PNGs** | Scenes: `fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1`. Ground-truth withheld |
| **Scene-Disjoint Splits** | `splits/manifest_5fold.json` | JSON format | 5 folds $\times$ 4 val scenes = 20 scenes | Strict scene-disjoint partition; SHA-256 verified |

---

## 3. Verified Checkpoint Inventory & Cryptographic Hashes

Every primary model and calibration artifact on disk has been verified for existence, non-corruption, and cryptographic SHA-256 integrity:

| Checkpoint Identifier | File Path | Size | SHA-256 Hash | Provenance & Description |
|---|---|---|---|---|
| **RSUIGM 20-Scene Table** | `checkpoints/track1_screening/tbl19_20sc_clean.pt` | 0.09 MB | `2c35df3b0f5d3317c3672b2867e0059b893acecd8699c5a2271383a1179d9562` | 20-scene physical calibration polynomial table (used in SOTA #18713) |
| **RSUIGM Fold 0 Table** | `checkpoints/track1_screening/tbl19_16sc_fold0.pt` | 0.09 MB | `d609740e3a6e80593ffc2684458bcf747fc7edaa90fe8f87801c6e367356d6a1` | Strict 16-scene Fold 0 calibration table (leak-free OOF control) |
| **DINOv2-Large Fold 0** | `checkpoints/v16_dinov2/fold_0/best_model.pt` | 1175.37 MB | `3bbe037c421f14ed8e961d2535b00e404ea45f4f8344484c09767e020196acf8` | DINOv2-Large (ViT-L/14) 4-level feature tap decoder (308.1M params) |
| **DINOv2-Large Fold 1** | `checkpoints/v16_dinov2/fold_1/best_model.pt` | 1175.37 MB | `9549d44c4fae89c629237db875cebdaeb88a51aa7424ad865b43eb759c8fa727` | Fold 1 SOTA model |
| **DINOv2-Large Fold 2** | `checkpoints/v16_dinov2/fold_2/best_model.pt` | 1175.37 MB | `6cebeab4e69b59e51c89be6fca554fb9ef6a08e6ca6f5ec88722aa0db8dfec75` | Fold 2 SOTA model |
| **DINOv2-Large Fold 3** | `checkpoints/v16_dinov2/fold_3/best_model.pt` | 1175.37 MB | `fe879301297e6beec68a83424d547f897f1f945763ca7c358f27cfcb9fb5e0fb` | Fold 3 SOTA model |
| **DINOv2-Large Fold 4** | `checkpoints/v16_dinov2/fold_4/best_model.pt` | 1175.37 MB | `6e828468b813d98fcbe7e8913926f25492dbe3cbeec067fc4eb668102a0a3809` | Fold 4 SOTA model |
| **ConvNeXt Fold 0** | `checkpoints/v16_convnext/fold_0/best_model.pt` | 342.26 MB | `ed36e5d39fcddfbf2d64e0e9aa6b959f4972bcdee9f493dcf90f770a8b8accbd` | ConvNeXt-Base + UPerNet (88.5M params) |
| **Physics NAFNet F0** | `checkpoints/v7_restoration/NAFNET_F0/best_model.pt`| 118.27 MB | `dadf061acba5ba3b44516cb62668276a8f32774020b44f4a81baae4e0eeba4fc` | 8-Channel Physics-Guided NAFNet restorer (Width 64, 16 blocks) |
| **EURS-Net V5 Generator**| `checkpoints/v5_gan_attn/best_v5_checkpoint.pt` | 10.19 MB | `703554a73e0c0a09e05f012217c4fe8e0e90956ec93bebe09172081d6f51cb7e` | EURS-Net V5 Generator with spatial attention |

---

## 4. Verified Submission Candidates & Production Artifacts

| Candidate Description | File Path | Format / Size | SHA-256 Hash | Compliance Status |
|---|---|---|---|---|
| **Track 1 Historical SOTA Fallback** | `deploy/eurs_track1_dryrun_compact_candidate.zip` | ZIP (564.91 MB) | `d6f4b9dfae2a084154d4820722fea48276fe1d5ae8a115557ea36417c3f7d2d6` | **VERIFIED PERFECT:** 1200 restored images + 1200 discrete masks at root. Matches historical hash exactly. |
| **Track 1 Apex SOTA Archive** | `deploy/submission_v16_apex_sota.zip` | ZIP (588.64 MB) | `943369fcdff398c1c18c9b43d4aa792baa1e343914a3dc6f186ca52bb8ced6df` | **VERIFIED PERFECT:** 1200 restored + 1200 masks at root. |
| **Track 2 Official Reference (#18699)**| `deploy/eurs_net_v5_calibrated_track2.onnx` | ONNX (3.14 MB) | `ad7bb1b1083083aee9fbbe44ce8974089d6b93077d892c20df84a267dd5fd135` | **PASSED:** 0.7683M params, 9.05 GFLOPs. Static `[1,3,1024,1024]`. |
| **Track 2 EdgeSOTA-V2 Candidate** | `deploy/eurs_edgesota_v2_final.onnx` | ONNX (5.42 MB) | `f547d0e1478311514c7f09f97b97d0c1e6f45087c85083342a154da45eba5f18` | **PASSED:** 0.3537M params, 2.2496 GFLOPs. Static `[1,3,1024,1024]`. |
| **Track 2 Combined-Best Candidate** | `deploy/EXP-COMBINED-BEST.onnx` | ONNX (7.69 MB) | `645eb785ba2bfb07a9ce4349b19b79151af627a88c58b466035395e48cbbcd52` | **PASSED:** 0.9589M params, 3.5666 GFLOPs. Static `[1,3,1024,1024]`. |

---

## 5. Official Metric Reconciliation & Scientific Provenance

| Metric Quantity | Official Submission #18713 (`EURSNetV6Verified`) | Official Submission #18699 (`EURS-Net V5-Calib`) | Newer Compact Track 2 (`EURSEdgeSOTA-V3`) | Notes on Provenance |
|---|---|---|---|---|
| **Composite Score** | **`0.6890`** | **`0.3380`** | **`0.4074`** *(Projected)* | #18713 and #18699 are confirmed official leaderboard records. `0.4074` is local projection. |
| **$Q_{\text{seg}}$** | **`0.6760`** | **`0.2450`** | **`0.3314`** *(Local held-out)* | Unweighted product: $\text{mIoU} \times \text{Macro } F_1 \times \text{coverage\_penalty}$. |
| **mIoU** | **`0.7770`** | **`0.4260`** | **`0.5389`** | Evaluated across all 5 semantic classes. |
| **Macro $F_1$** | **`0.8700`** | **`0.5760`** | **`0.6150`** | Unweighted mean across all 5 classes. |
| **Coverage Penalty** | `1.0000` | `1.0000` | `1.0000` | All present classes detected ($\text{TP} > 0$). |
| **PSNR (dB)** | **`25.41 dB`** | **`17.10 dB`** | **`18.55 dB`** | Evaluated on Dev / Held-out split. |
| **$\text{PSNR}_{\text{norm}}$** | `0.5205` | `0.1050` | `0.1775` | $\text{clip}((\text{PSNR} - 15) / 20, 0, 1)$. |
| **SSIM** | **`0.8160`** | **`0.8210`** | **`0.7278`** | Full-reference structural similarity. |
| **Sharpness Ratio** | **`0.9600`** | **`0.3190`** | **`0.4866`** | High-frequency gradient magnitude ratio. |
| **$Q_{\text{FR}}$** | `0.6974` | `0.4486` | `0.4560` | $0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{sharpness}$. |
| **$Q_{\text{NR}}$ (Estimated)**| `0.7417` | — | — | Back-calculated from reported UIQM/UCIQE/CCF. |
| **Parameters** | $> 1,540\text{M}$ (5-fold DINOv2-L) | **`0.7683M`** | **`0.3537M`** | Track 2 official cap is $\le 1.0\text{M}$. |
| **Compute (GFLOPs)** | $> 1,000\text{ GFLOPs}$ | **`9.0500 GFLOPs`** | **`2.2496 GFLOPs`** | Track 2 official cap is $\le 12.0\text{ GFLOPs}$. |
