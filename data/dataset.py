"""
EURSDataset: Scene-Disjoint Paired Underwater Dataset
Implements paired loading of (degraded, clean, mask) with strict scene-disjoint verification.
"""

import os
import re
import json
from typing import Tuple, Optional, Callable, Dict, Any, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class EURSDataset(Dataset):
    """
    EURS PyTorch Dataset:
    - Loads triplets: (degraded_image, clean_image, mask_image)
    - Enforces scene-disjoint train/val partitions using a persistent manifest.json
    - Enforces deterministic pairing: degraded_scene_id == clean_scene_id == mask_scene_id
    - Enforces input/target contracts:
        degraded: float32, [3, 1024, 1024], range [0.0, 1.0]
        clean:    float32, [3, 1024, 1024], range [0.0, 1.0]
        mask:     int64,   [1024, 1024],    values in {0, 1, 2, 3, 4}
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        manifest_path: str = 'd:/MACVI/splits/manifest.json',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.split = split.lower()
        self.transform = transform
        self.target_size = target_size

        assert self.split in ('train', 'val', 'all'), f"Invalid split: {self.split}"

        self.clean_dir = os.path.join(self.data_dir, 'clean')
        self.masks_dir = os.path.join(self.data_dir, 'masks')
        self.degraded_dir = os.path.join(self.data_dir, 'degraded')

        assert os.path.exists(self.clean_dir), f"Clean directory not found: {self.clean_dir}"
        assert os.path.exists(self.masks_dir), f"Masks directory not found: {self.masks_dir}"
        assert os.path.exists(self.degraded_dir), f"Degraded directory not found: {self.degraded_dir}"

        # Load frozen persistent manifest
        assert os.path.exists(manifest_path), f"Manifest file not found: {manifest_path}"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        train_scenes = set(manifest['train_scenes'])
        val_scenes = set(manifest['val_scenes'])

        # Invariant: train and val scene sets must be strictly disjoint
        assert train_scenes.isdisjoint(val_scenes), "FATAL: Data leakage detected in manifest!"

        if self.split == 'train':
            active_scenes = sorted(list(train_scenes))
        elif self.split == 'val':
            active_scenes = sorted(list(val_scenes))
        else:
            active_scenes = sorted(list(train_scenes | val_scenes))

        self.samples: List[Dict[str, str]] = []

        # Index all samples strictly enforcing pairing invariant
        for scene_id in active_scenes:
            clean_file = f"{scene_id}.png"
            clean_path = os.path.join(self.clean_dir, clean_file)
            if not os.path.exists(clean_path):
                # Check for alternative extensions
                for ext in ('.jpg', '.jpeg'):
                    alt = os.path.join(self.clean_dir, f"{scene_id}{ext}")
                    if os.path.exists(alt):
                        clean_path = alt
                        break

            mask_file = f"{scene_id}.png"
            mask_path = os.path.join(self.masks_dir, mask_file)
            if not os.path.exists(mask_path):
                for ext in ('.jpg', '.jpeg'):
                    alt = os.path.join(self.masks_dir, f"{scene_id}{ext}")
                    if os.path.exists(alt):
                        mask_path = alt
                        break

            assert os.path.exists(clean_path), f"Clean reference missing for scene {scene_id}: {clean_path}"
            assert os.path.exists(mask_path), f"Mask reference missing for scene {scene_id}: {mask_path}"

            scene_deg_dir = os.path.join(self.degraded_dir, scene_id)
            if os.path.isdir(scene_deg_dir):
                deg_files = sorted([
                    f for f in os.listdir(scene_deg_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                for df in deg_files:
                    deg_path = os.path.join(scene_deg_dir, df)
                    self.samples.append({
                        'scene_id': scene_id,
                        'degraded_path': deg_path,
                        'clean_path': clean_path,
                        'mask_path': mask_path
                    })
            else:
                # If degraded variations are stored directly as scene_id_*.png
                raise FileNotFoundError(f"Degraded directory missing for scene {scene_id}: {scene_deg_dir}")

        print(f"[EURSDataset] Initialized '{self.split}' split: {len(self.samples)} samples across {len(active_scenes)} scenes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample_meta = self.samples[idx]
        scene_id = sample_meta['scene_id']
        deg_path = sample_meta['degraded_path']
        clean_path = sample_meta['clean_path']
        mask_path = sample_meta['mask_path']

        # Pairing verification invariant
        deg_scene_check = os.path.basename(os.path.dirname(deg_path))
        clean_scene_check = os.path.splitext(os.path.basename(clean_path))[0]
        mask_scene_check = os.path.splitext(os.path.basename(mask_path))[0]
        assert deg_scene_check == clean_scene_check == mask_scene_check == scene_id, (
            f"Corrupted sample pairing detected at index {idx}! "
            f"DegScene={deg_scene_check}, CleanScene={clean_scene_check}, MaskScene={mask_scene_check}"
        )

        # Load degraded image (convert BGR to RGB)
        degraded = cv2.imread(deg_path, cv2.IMREAD_COLOR)
        if degraded is None:
            raise IOError(f"Failed to read image: {deg_path}")
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # Load clean ground truth image (convert BGR to RGB)
        clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        if clean is None:
            raise IOError(f"Failed to read image: {clean_path}")
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        # Load ground truth semantic mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")

        # Initial resize to target_size if not already matching
        target_w, target_h = self.target_size
        if (degraded.shape[1], degraded.shape[0]) != (target_w, target_h):
            degraded = cv2.resize(degraded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if (clean.shape[1], clean.shape[0]) != (target_w, target_h):
            clean = cv2.resize(clean, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if (mask.shape[1], mask.shape[0]) != (target_w, target_h):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Apply synchronized spatial transformations if provided
        if self.transform is not None:
            degraded, clean, mask = self.transform(degraded, clean, mask, seed=idx)

        # Validate mask values
        unique_classes = np.unique(mask)
        assert np.all(np.isin(unique_classes, [0, 1, 2, 3, 4])), (
            f"Invalid mask values found in {mask_path}: {unique_classes}"
        )

        # Convert to PyTorch tensors with strict type/range contracts
        # Degraded & Clean: float32, [3, H, W], range [0.0, 1.0]
        deg_tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).float() / 255.0
        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0

        # Mask: int64 (torch.long), [H, W], values {0, 1, 2, 3, 4}
        mask_tensor = torch.from_numpy(mask).long()

        # Invariant asserts on returned tensors
        assert deg_tensor.shape == (3, target_h, target_w)
        assert clean_tensor.shape == (3, target_h, target_w)
        assert mask_tensor.shape == (target_h, target_w)
        assert deg_tensor.min() >= 0.0 and deg_tensor.max() <= 1.0
        assert clean_tensor.min() >= 0.0 and clean_tensor.max() <= 1.0

        meta = {
            'scene_id': scene_id,
            'degraded_file': os.path.basename(deg_path),
            'clean_file': os.path.basename(clean_path),
            'mask_file': os.path.basename(mask_path),
        }

        return deg_tensor, clean_tensor, mask_tensor, meta
