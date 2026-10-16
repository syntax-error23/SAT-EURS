"""
DeepLabV3-ResNet101 Training for Track 1 Underwater Semantic Segmentation
Pretrained on ImageNet/COCO, fine-tuned on EURS dataset to achieve mIoU >= 0.75 - 0.85
without hallucinating rare classes on pure scenes.
"""

import os
import os.path as osp
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image

# 1. Dataset definition
class EURSSegDataset(Dataset):
    def __init__(self, data_root: str = "data/train", crop_size: int = 512, samples_per_epoch: int = 1600):
        self.data_root = data_root
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.degraded_dir = osp.join(data_root, "degraded")

        self.scenes = sorted([
            osp.splitext(f)[0] for f in os.listdir(self.clean_dir)
            if f.endswith((".png", ".jpg"))
        ])

        # Preload clean images and masks into RAM
        print(f"[Dataset] Preloading {len(self.scenes)} scene masks and clean images into RAM...")
        self.clean_images = {}
        self.masks = {}
        self.scene_classes = {}

        for sc in self.scenes:
            im_path = osp.join(self.clean_dir, f"{sc}.png")
            mask_path = osp.join(self.masks_dir, f"{sc}.png")
            with Image.open(im_path) as im:
                self.clean_images[sc] = np.array(im.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_path) as m:
                mask_arr = np.array(m, dtype=np.uint8)
                self.masks[sc] = mask_arr
                self.scene_classes[sc] = set(np.unique(mask_arr))

        # Build list of degraded files per scene
        self.degraded_files = {}
        for sc in self.scenes:
            sc_deg_dir = osp.join(self.degraded_dir, sc)
            if osp.exists(sc_deg_dir):
                self.degraded_files[sc] = [
                    osp.join(sc_deg_dir, f) for f in os.listdir(sc_deg_dir)
                    if f.endswith((".png", ".jpg"))
                ]
            else:
                self.degraded_files[sc] = []

        print(f"[Dataset] Successfully cached all {len(self.scenes)} scenes!")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Weighted scene sampling: oversample rare class scenes (ray1, sharkFish, starfish2, turtles1, seahorse1)
        rare_scenes = ["ray1", "sharkFish", "starfish2", "turtles1", "seahorse1"]
        if random.random() < 0.40:
            sc = random.choice(rare_scenes)
        else:
            sc = random.choice(self.scenes)

        clean_im = self.clean_images[sc]
        mask = self.masks[sc]
        h, w = mask.shape

        # Randomly choose between clean image (50%) and degraded variant (50%) for robust feature invariance
        use_degraded = (random.random() < 0.50) and len(self.degraded_files[sc]) > 0
        if use_degraded:
            deg_path = random.choice(self.degraded_files[sc])
            with Image.open(deg_path) as dim:
                if dim.size != (w, h):
                    dim = dim.resize((w, h), Image.BILINEAR)
                img = np.array(dim.convert("RGB"), dtype=np.uint8)
        else:
            img = clean_im

        # Random crop of crop_size x crop_size
        cs = self.crop_size
        if h > cs and w > cs:
            # If rare scene, try to center crop around present object
            present_classes = self.scene_classes[sc]
            target_class = 3 if 3 in present_classes else (4 if 4 in present_classes else None)
            
            top, left = 0, 0
            found_crop = False
            if target_class is not None and random.random() < 0.70:
                ys, xs = np.where(mask == target_class)
                if len(ys) > 0:
                    c_idx = random.randint(0, len(ys) - 1)
                    center_y, center_x = ys[c_idx], xs[c_idx]
                    top = max(0, min(h - cs, center_y - cs // 2))
                    left = max(0, min(w - cs, center_x - cs // 2))
                    found_crop = True

            if not found_crop:
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)

            img_crop = img[top:top+cs, left:left+cs]
            mask_crop = mask[top:top+cs, left:left+cs]
        else:
            # Resize if smaller than crop size
            img_crop = np.array(Image.fromarray(img).resize((cs, cs), Image.BILINEAR))
            mask_crop = np.array(Image.fromarray(mask).resize((cs, cs), Image.NEAREST))

        # Data augmentation: Horizontal & Vertical Flip
        if random.random() < 0.5:
            img_crop = np.fliplr(img_crop)
            mask_crop = np.fliplr(mask_crop)
        if random.random() < 0.3:
            img_crop = np.flipud(img_crop)
            mask_crop = np.flipud(mask_crop)

        # Normalize with standard ImageNet mean and std for DeepLabV3
        img_crop = img_crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_crop = (img_crop - mean) / std

        img_tensor = torch.from_numpy(img_crop.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask_crop.copy()).long()
        return img_tensor, mask_tensor


def build_deeplabv3_model(num_classes: int = 5) -> nn.Module:
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    # Replace final classification conv layers
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def train_deeplabv3(
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1.5e-4,
    save_path: str = "checkpoints/deeplabv3_resnet101_best.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Initializing DeepLabV3-ResNet101 on {device}...")
    model = build_deeplabv3_model(num_classes=5).to(device)

    # Class weights to balance foreground vs background without over-boosting
    # Classes: 0: bg, 1: fish, 2: coral, 3: shark/ray, 4: starfish/turtle/wreck
    class_weights = torch.tensor([0.80, 1.20, 1.30, 2.50, 2.50], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    dataset = EURSSegDataset(data_root="data/train", crop_size=512, samples_per_epoch=1600)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    print("=" * 72)
    print("      DEEPLABV3-RESNET101 HIGH-CAPACITY UNDERWATER SEGMENTATION ENGINE")
    print(f"      Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (images, masks) in enumerate(dataloader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                main_loss = ce_loss_fn(outputs['out'], masks)
                if 'aux' in outputs:
                    aux_loss = ce_loss_fn(outputs['aux'], masks)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    loss = main_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 50 == 0:
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch}/{epochs} Complete - Avg Loss: {epoch_loss:.4f} ({elapsed:.1f}s) ===")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"  >>> Checkpoint saved to {save_path} (Loss: {best_loss:.4f})")

    print("[Training Finished] DeepLabV3-ResNet101 ready for decoupled Track 1 deployment!")


if __name__ == "__main__":
    train_deeplabv3()
