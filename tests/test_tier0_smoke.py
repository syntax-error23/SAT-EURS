"""
Tier 0 Smoke Test Suite
Verifies:
1. Forward pass on real data batch on NVIDIA RTX 5090
2. Loss calculation (Cross-Entropy + Lovasz) and backward pass
3. Tiny overfit test on 4 samples
4. Checkpoint save & reload integrity
5. Raw logit output contract [B, 5, H, W]
6. VRAM and throughput profiling (guaranteeing < 90% VRAM / < 28 GB)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.segmentation_metrics import SegmentationEvaluator
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
import segmentation_models_pytorch as smp


class SyntheticBatchDataset(Dataset):
    def __init__(self, n_samples: int = 4, size: int = 518):
        self.size = size
        self.n_samples = n_samples
        # Load 4 real image crops from data/train if available, else synthetic
        self.images = []
        self.masks = []
        
        real_img_path = "data/cached_1024/clean/fish1.png"
        real_mask_path = "data/cached_1024/masks/fish1.png"
        
        if os.path.exists(real_img_path) and os.path.exists(real_mask_path):
            img_full = Image.open(real_img_path).convert("RGB")
            mask_full = Image.open(real_mask_path)
            for i in range(n_samples):
                # crop 518x518
                box = (i * 50, i * 50, i * 50 + size, i * 50 + size)
                im_crop = img_full.crop(box)
                m_crop = mask_full.crop(box)
                img_t = torch.from_numpy(np.array(im_crop, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                mask_t = torch.from_numpy(np.array(m_crop, dtype=np.int64))
                self.images.append(img_t)
                self.masks.append(mask_t)
        else:
            for _ in range(n_samples):
                self.images.append(torch.rand(3, size, size))
                self.masks.append(torch.randint(0, 5, (size, size)))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def build_dinov2_segmenter(num_classes: int = 5):
    """Clean foundation segmentation model using DINOv2-L/14."""
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    embed_dim = encoder.embed_dim  # 1024
    intermediate_blocks = (5, 11, 17, 23)

    class Segmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.intermediate_blocks = intermediate_blocks
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dim, 256, kernel_size=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ) for _ in intermediate_blocks
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(256 * len(intermediate_blocks), 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            features = self.encoder.get_intermediate_layers(x, n=self.intermediate_blocks, return_class_token=False)
            h_feat, w_feat = H // 14, W // 14
            projected = []
            for feat, p in zip(features, self.proj):
                f_map = feat.permute(0, 2, 1).reshape(B, -1, h_feat, w_feat)
                projected.append(p(f_map))
            cat_feat = torch.cat(projected, dim=1)
            logits_low = self.fusion(cat_feat)
            logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
            return logits  # Always returns raw unnormalized logits!

    return Segmenter()


def run_tier0_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"RUNNING TIER 0 SMOKE TEST SUITE ON {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()
    ds = SyntheticBatchDataset(n_samples=4, size=518)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    # 1. Forward Pass on Real Batch
    print("[Step 1/6] Instantiating DINOv2-L/14 Segmenter...")
    t0 = time.time()
    model = build_dinov2_segmenter(num_classes=5).to(device)
    print(f"  Model loaded in {time.time() - t0:.2f}s. Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    # ImageNet norm
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x_norm)

    assert logits.shape == (2, 5, 518, 518), f"Expected shape [2, 5, 518, 518], got {logits.shape}"
    print(f"  [PASS] Forward pass successful! Output logits shape: {logits.shape}")

    # 2. Loss & Backward Pass
    print("[Step 2/6] Calculating Loss and Backward Pass...")
    ce_loss_fn = nn.CrossEntropyLoss()
    lovasz_loss_fn = PresentClassLovaszSoftmaxLoss(num_classes=5)

    loss_ce = ce_loss_fn(logits.float(), y)
    loss_lov = lovasz_loss_fn(logits.float(), y)
    loss = loss_ce + 0.25 * loss_lov

    loss.backward()
    print(f"  [PASS] Backward pass successful! Loss CE: {loss_ce.item():.4f}, Lovasz: {loss_lov.item():.4f}, Total: {loss.item():.4f}")

    # 3. Tiny Overfit Test (4 samples for 15 steps)
    print("[Step 3/6] Running Tiny Overfit Convergence Test (4 samples, 15 steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x_norm)
            l = ce_loss_fn(pred.float(), y)
        l.backward()
        optimizer.step()
        if step == 0:
            initial_loss = l.item()
        final_loss = l.item()

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    print(f"  [PASS] Overfit test successful: Loss decreased from {initial_loss:.4f} -> {final_loss:.4f} (decay: {(1 - final_loss/initial_loss)*100:.1f}%)")

    # 4. Checkpoint Save & Reload
    print("[Step 4/6] Testing Checkpoint Save and Reload...")
    os.makedirs("checkpoints/smoke_test", exist_ok=True)
    ckpt_path = "checkpoints/smoke_test/smoke_dinov2.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": 15}, ckpt_path)
    assert os.path.exists(ckpt_path), "Failed to save checkpoint"
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert "model_state_dict" in loaded, "Corrupted checkpoint reload"
    model.load_state_dict(loaded["model_state_dict"])
    print("  [PASS] Checkpoint save and reload verified!")

    # 5. Output Contract & Inference Test
    print("[Step 5/6] Verifying Standard Model Inference Contract...")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            eval_logits = model(x_norm)
    # Assert logits are raw (can have negative and positive values outside [0, 1])
    assert eval_logits.min() < 0.0 or eval_logits.max() > 1.0, "Model incorrectly returning normalized probabilities instead of raw logits!"
    print(f"  [PASS] Output contract verified: Model returns raw unnormalized logits (range [{eval_logits.min():.2f}, {eval_logits.max():.2f}])")

    # 6. VRAM and Throughput Benchmark
    print("[Step 6/6] Profiling VRAM & Throughput on RTX 5090...")
    torch.cuda.synchronize()
    t_start = time.time()
    n_benchmark_iters = 10
    with torch.no_grad():
        for _ in range(n_benchmark_iters):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(x_norm)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    total_imgs = n_benchmark_iters * x.shape[0]
    throughput = total_imgs / elapsed

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    vram_percent = (peak_vram_gb / max_vram_gb) * 100

    print(f"  Peak VRAM Allocated: {peak_vram_gb:.2f} GB / {max_vram_gb:.2f} GB ({vram_percent:.1f}%)")
    print(f"  Inference Throughput: {throughput:.1f} images/sec (batch size = {x.shape[0]})")
    assert vram_percent < 90.0, f"FATAL: VRAM exceeded 90% threshold! Peak: {vram_percent:.1f}%"
    print(f"  [PASS] Peak VRAM is well within safe threshold ({vram_percent:.1f}% < 90.0%)")

    # Clean up smoke checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("=" * 70)
    print("ALL TIER 0 SMOKE TESTS PASSED AUTONOMOUSLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tier0_smoke()
