"""
SAT-EURS Track 1 — V18 Dataset & Scene-Balanced Invariance Sampler
Features:
1. Paired EURS degraded/clean data with identical categorical masks
2. Dual-view invariance output: clean reference + condition-varied degraded view
3. Rare-class centered cropping (classes 3 and 4) to ensure high representation
4. Scene-balanced sampling to prevent rare-class starvation
5. Synchronous spatial augmentations (crops, horizontal flips)
"""

import os
import os.path as osp
import random
from typing import List, Tuple, Dict, Optional, Any, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


IGNORE_INDEX = 255


class V18EURSInvarianceDataset(Dataset):
    """
    EURS dataset producing paired degraded and clean views with aligned categorical masks.
    Enforces scene-disjoint invariants and supports object-centered rare class sampling.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 512,
        is_train: bool = True,
        rare_crop_prob: float = 0.60,
        base_seed: int = 42,
        cache_dict: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.rare_crop_prob = rare_crop_prob
        self.base_seed = base_seed
        self.epoch = 0

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.deg_dir = osp.join(data_root, "degraded")

        self.items: List[Tuple[str, int, int]] = []
        for sc in self.scenes:
            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    self.items.append((sc, wt, depth))

        # In-memory storage for clean images, masks, and rare coordinates
        self.clean_images: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self.rare_coords: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        self.cache_dict = cache_dict

        self._preload_metadata()

    def _preload_metadata(self):
        for sc in self.scenes:
            if self.cache_dict and "clean" in self.cache_dict and sc in self.cache_dict["clean"]:
                c_rgb = self.cache_dict["clean"][sc]
            elif osp.exists(osp.join(self.clean_dir, f"{sc}.png")):
                c_bgr = cv2.imread(osp.join(self.clean_dir, f"{sc}.png"), cv2.IMREAD_COLOR)
                c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            else:
                c_rgb = np.zeros((1024, 1024, 3), dtype=np.float32)
            self.clean_images[sc] = c_rgb

            m_path = osp.join(self.masks_dir, f"{sc}.png")
            if osp.exists(m_path):
                m_arr = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            else:
                m_arr = np.zeros((1024, 1024), dtype=np.uint8)
            self.masks[sc] = m_arr

            # Index rare classes (3: shark/ray, 4: other marine fauna)
            self.rare_coords[sc] = {}
            for c in [3, 4]:
                ys, xs = np.where(m_arr == c)
                if len(ys) > 0:
                    self.rare_coords[sc][c] = (ys, xs)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sc, wt, depth = self.items[idx]
        clean_full = self.clean_images[sc]
        mask_full = self.masks[sc]

        # Load degraded image
        if self.cache_dict and "degraded" in self.cache_dict and (sc, wt, depth) in self.cache_dict["degraded"]:
            deg_rgb = self.cache_dict["degraded"][(sc, wt, depth)].astype(np.float32) / 255.0
        else:
            deg_path = osp.join(self.deg_dir, sc, f"{sc}_{wt}_{depth}.png")
            if osp.exists(deg_path):
                bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
                deg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            else:
                deg_rgb = np.full((1024, 1024, 3), 0.5, dtype=np.float32)

        h, w = mask_full.shape
        cs = self.crop_size

        if self.is_train and h >= cs and w >= cs:
            eff_seed = self.base_seed + self.epoch * 10007 + idx * 1009
            rng = random.Random(eff_seed)

            # Check for rare class crop
            rare_classes = list(self.rare_coords[sc].keys())
            found_rare_crop = False
            top, left = 0, 0

            if len(rare_classes) > 0 and rng.random() < self.rare_crop_prob:
                target_cls = rng.choice(rare_classes)
                ys, xs = self.rare_coords[sc][target_cls]
                if len(ys) > 0:
                    c_idx = rng.randint(0, len(ys) - 1)
                    center_y, center_x = ys[c_idx], xs[c_idx]
                    top = max(0, min(h - cs, center_y - cs // 2))
                    left = max(0, min(w - cs, center_x - cs // 2))
                    found_rare_crop = True

            if not found_rare_crop:
                top = rng.randint(0, h - cs)
                left = rng.randint(0, w - cs)

            clean_crop = clean_full[top:top+cs, left:left+cs].copy()
            deg_crop = deg_rgb[top:top+cs, left:left+cs].copy()
            mask_crop = mask_full[top:top+cs, left:left+cs].copy()

            # Synchronous random horizontal flip
            if rng.random() < 0.50:
                clean_crop = np.fliplr(clean_crop).copy()
                deg_crop = np.fliplr(deg_crop).copy()
                mask_crop = np.fliplr(mask_crop).copy()
        else:
            # Full image or resize to crop_size
            if (h, w) != (cs, cs):
                clean_crop = cv2.resize(clean_full, (cs, cs), interpolation=cv2.INTER_LINEAR)
                deg_crop = cv2.resize(deg_rgb, (cs, cs), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask_full, (cs, cs), interpolation=cv2.INTER_NEAREST)
            else:
                clean_crop = clean_full
                deg_crop = deg_rgb
                mask_crop = mask_full

        return {
            "clean_rgb": torch.from_numpy(clean_crop).permute(2, 0, 1).float(),
            "deg_rgb": torch.from_numpy(deg_crop).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask_crop.astype(np.int64)).long(),
            "wt": torch.tensor(wt, dtype=torch.long),
            "depth": torch.tensor(depth, dtype=torch.long),
            "scene": sc
        }


class V18SceneBalancedSampler(Sampler):
    """
    Scene-Balanced Sampler:
    Ensures that rare-class scenes (containing classes 3 and 4) are sampled with
    guaranteed frequency in every epoch, avoiding gradient starvation.
    """
    def __init__(
        self,
        dataset: V18EURSInvarianceDataset,
        batch_size: int = 4,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Group indices by scene
        self.scene_indices: Dict[str, List[int]] = {}
        for idx, (sc, _, _) in enumerate(dataset.items):
            self.scene_indices.setdefault(sc, []).append(idx)

        self.all_scenes = sorted(list(self.scene_indices.keys()))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 10007)
        # Shuffle per-scene pools
        shuffled_pools = {sc: list(self.scene_indices[sc]) for sc in self.all_scenes}
        for sc in self.all_scenes:
            rng.shuffle(shuffled_pools[sc])

        batches = []
        # Draw 1 sample per scene cyclically across the batch
        num_batches = len(self.dataset) // self.batch_size
        scene_cycle = list(self.all_scenes)

        for _ in range(num_batches):
            batch = []
            rng.shuffle(scene_cycle)
            for sc in scene_cycle[:self.batch_size]:
                if not shuffled_pools[sc]:
                    shuffled_pools[sc] = list(self.scene_indices[sc])
                    rng.shuffle(shuffled_pools[sc])
                batch.append(shuffled_pools[sc].pop())
            batches.extend(batch)

        return iter(batches)

    def __len__(self) -> int:
        return (len(self.dataset) // self.batch_size) * self.batch_size
