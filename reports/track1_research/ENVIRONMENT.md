# SAT-EURS Track 1 — Environment and Hardware Specification

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Git Branch:** `track1-research`  
**Latest Git Commit:** `9a17437e39b5d2f7dc4cf84957635022ce12dc6e`  
**Remote:** `https://github.com/syntax-error23/SAT-EURS.git`

---

## 1. Physical Hardware & Operating System

| Component | Specification | Verification Source | Status |
| :--- | :--- | :--- | :---: |
| **Operating System** | Windows 11 Pro (`Windows-10-10.0.26200-SP0`) | `platform.platform()` | **VERIFIED FACT** |
| **CPU** | Intel Core Ultra 9 (`Intel64 Family 6 Model 198 Stepping 2, GenuineIntel`) | `platform.processor()` | **VERIFIED FACT** |
| **System RAM** | **63.43 GB Total**, **50.10 GB Available** | `GlobalMemoryStatusEx` API | **VERIFIED FACT** |
| **Primary GPU** | **NVIDIA GeForce RTX 5090** | `torch.cuda.get_device_name(0)` | **VERIFIED FACT** |
| **GPU Total VRAM** | **31.84 GB (32 GB physical GDDR7)** | `torch.cuda.get_device_properties(0)` | **VERIFIED FACT** |
| **Storage (Drive C:)** | **919.93 GB Total**, **694.52 GB Free** | `shutil.disk_usage("c:")` | **VERIFIED FACT** |

---

## 2. Software Runtime & Deep Learning Frameworks

| Package | Version | Provenance / Location |
| :--- | :--- | :--- |
| **Python** | `3.11.9 (tags/v3.11.9:de54cf5)` | `C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe` |
| **PyTorch** | `2.11.0+cu128` | CUDA 12.8 native binary with SM 12.0 support |
| **CUDA Runtime** | `12.8` | PyTorch bundled CUDA runtime |
| **TorchVision** | `0.26.0+cu128` | Standard torchvision build |
| **timm** | `1.0.30` | PyPI install |
| **segmentation_models_pytorch** | `0.5.0` | PyPI install |
| **OpenCV** | `5.0.0.93` (`cv2`) | Prebuilt wheel with multithreading support |
| **Pillow (PIL)** | `12.3.0` | Standard imaging library |
| **NumPy** | `2.4.6` | High-performance C array runtime |
| **ONNX Runtime** | `1.31.0` | Track 2 evaluation engine |
| **ONNX** | `1.23.2` | Graph export and validation |
| **PyYAML** | `6.0.3` | Configuration parsing |
| **ptflops** | `0.7.5` | FLOP/parameter profiling |

---

## 3. Verified Pretrained Model Weights Inventory

Pretrained checkpoints available locally without external internet downloads:

| Model Architecture | Local Path | File Size | Provenance / Status |
| :--- | :--- | :---: | :--- |
| **DINOv2 ViT-L/14** | `C:\Users\HP/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth` | 1,161.2 MB | **VERIFIED OFFLINE** (Torch Hub cached repo: `facebookresearch_dinov2_main`) |
| **DINOv2 ViT-B/14** | `C:\Users\HP/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` | 330.3 MB | **VERIFIED OFFLINE** |
| **DeepLabV3 ResNet101** | `C:\Users\HP/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth` | 233.2 MB | **VERIFIED OFFLINE** |
| **DeepLabV3 ResNet50** | `C:\Users\HP/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth` | 160.5 MB | **VERIFIED OFFLINE** |
| **ResNet101 Backbone** | `C:\Users\HP/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth` | 170.5 MB | **VERIFIED OFFLINE** |
| **SegFormer MiT-B3** | `C:\Users\HP/.cache/huggingface/hub/models--smp-hub--mit_b3.imagenet/` | 170.1 MB | **VERIFIED OFFLINE** (safetensors format) |
| **SegFormer MiT-B5** | `C:\Users\HP/.cache/huggingface/hub/models--smp-hub--mit_b5.imagenet/` | 312.7 MB | **VERIFIED OFFLINE** (safetensors format) |

---

## 4. Operational Safety Constraints

1. **VRAM Safety Threshold:** Under no circumstances should peak allocated GPU VRAM exceed 90% (28.6 GB). All training jobs must monitor VRAM and maintain a working allocation $\le 24$ GB.
2. **Precision:** Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)` or `torch.float16` with scaling to ensure optimal Tensor Core utilization on the RTX 5090 without risking OOM spikes.
3. **Reproducibility:** Seed all RNGs (Python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) explicitly and make sample indexing deterministic.
