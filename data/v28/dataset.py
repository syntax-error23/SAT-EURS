"""
SAT-EURS Track 1 — V28 Degradation-Aware Dataset & Sampler
Supports:
1. Scene-disjoint training and evaluation across 5 folds (16 train / 4 val scenes per fold)
2. Paired degraded underwater images (data/cached_1024/degraded/) with synchronized masks
3. Targeted small-object awareness: Class 2 (Fish), Class 3 (Shark/Ray), Class 4 (Macro-fauna)
4. Scale-aware random zoom jitter (crop sizes 384..512 resized to 512x512)
5. Strict geometric synchronization across inputs
"""

import os
import os.path as osp
import random
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class V28DegradationAwareDataset(Dataset):
    """
    Degradation-Aware Underwater Dataset for SAT-EURS Track 1.
    Pairs 200 degradation variants per scene with ground truth segmentation masks.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        target_size: int = 512,
        is_train: bool = True,
        object_crop_prob: float = 0.70,
        min_crop_ratio: float = 0.75,  # 384 / 512
        base_seed: int = 42
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.data_root = data_root
        self.target_size = target_size
        self.is_train = is_train
        self.object_crop_prob = object_crop_prob
        self.min_crop_ratio = min_crop_ratio
        self.base_seed = base_seed
        self.epoch = 0

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.deg_dir = osp.join(data_root, "degraded")

        # Build list of (scene, wt, depth) items
        self.items: List[Tuple[str, int, int]] = []
        for sc in self.scenes:
            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    self.items.append((sc, wt, depth))

        # In-memory storage for clean references, masks, and object coordinates
        self.clean_images: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self.object_coords: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        self._preload_metadata()

    def _preload_metadata(self):
        for sc in self.scenes:
            c_path = osp.join(self.clean_dir, f"{sc}.png")
            if osp.exists(c_path):
                c_bgr = cv2.imread(c_path, cv2.IMREAD_COLOR)
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

            # Index foreground classes: 2 (Fish), 3 (Shark/Ray), 4 (Macro-fauna), 1 (Coral)
            self.object_coords[sc] = {}
            for c in [2, 3, 4, 1]:
                ys, xs = np.where(m_arr == c)
                if len(ys) > 0:
                    self.object_coords[sc][c] = (ys, xs)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sc, wt, depth = self.items[idx]
        mask_full = self.masks[sc]
        clean_full = self.clean_images[sc]

        deg_path = osp.join(self.deg_dir, sc, f"{sc}_{wt}_{depth}.png")
        if osp.exists(deg_path):
            bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
            deg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            deg_rgb = np.full((1024, 1024, 3), 0.5, dtype=np.float32)

        h, w = mask_full.shape
        ts = self.target_size

        if self.is_train and h >= ts and w >= ts:
            eff_seed = self.base_seed + self.epoch * 10007 + idx * 1009
            rng = random.Random(eff_seed)

            # Sample crop size with zoom jitter in [min_crop_ratio * ts, ts]
            min_size = int(self.min_crop_ratio * ts)
            cs = rng.randint(min_size, ts)

            # Prioritize rare fauna: Macro (4), Shark/Ray (3), Fish (2)
            avail_classes = list(self.object_coords[sc].keys())
            found_object_crop = False
            top, left = 0, 0

            if len(avail_classes) > 0 and rng.random() < self.object_crop_prob:
                weights = [4.0 if c in (3, 4) else (3.0 if c == 2 else 1.0) for c in avail_classes]
                target_cls = rng.choices(avail_classes, weights=weights, k=1)[0]
                ys, xs = self.object_coords[sc][target_cls]
                if len(ys) > 0:
                    c_idx = rng.randint(0, len(ys) - 1)
                    center_y, center_x = ys[c_idx], xs[c_idx]
                    top = max(0, min(h - cs, center_y - cs // 2))
                    left = max(0, min(w - cs, center_x - cs // 2))
                    found_object_crop = True

            if not found_object_crop:
                top = rng.randint(0, h - cs)
                left = rng.randint(0, w - cs)

            clean_crop = clean_full[top:top+cs, left:left+cs].copy()
            deg_crop = deg_rgb[top:top+cs, left:left+cs].copy()
            mask_crop = mask_full[top:top+cs, left:left+cs].copy()

            # Resize to target_size (512x512) if zoomed
            if cs != ts:
                clean_crop = cv2.resize(clean_crop, (ts, ts), interpolation=cv2.INTER_LINEAR)
                deg_crop = cv2.resize(deg_crop, (ts, ts), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask_crop, (ts, ts), interpolation=cv2.INTER_NEAREST)

            # Synchronous random horizontal flip
            if rng.random() < 0.50:
                clean_crop = np.fliplr(clean_crop).copy()
                deg_crop = np.fliplr(deg_crop).copy()
                mask_crop = np.fliplr(mask_crop).copy()
        else:
            # Validation mode
            if (h, w) != (ts, ts):
                clean_crop = cv2.resize(clean_full, (ts, ts), interpolation=cv2.INTER_LINEAR)
                deg_crop = cv2.resize(deg_rgb, (ts, ts), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask_full, (ts, ts), interpolation=cv2.INTER_NEAREST)
            else:
                clean_crop = clean_full
                deg_crop = deg_rgb
                mask_crop = mask_full

        # Convert to torch tensors
        t_clean = torch.from_numpy(clean_crop).permute(2, 0, 1).float()
        t_deg = torch.from_numpy(deg_crop).permute(2, 0, 1).float()
        t_mask = torch.from_numpy(mask_crop).long()

        return {
            "scene": sc,
            "wt": torch.tensor(wt, dtype=torch.long),
            "depth": torch.tensor(depth, dtype=torch.float32),
            "deg_rgb": t_deg,
            "clean_rgb": t_clean,
            "mask": t_mask,
        }


class V28SceneBalancedSampler(Sampler):
    """
    Balances sampling across training scenes within each batch so rare scenes
    (ray1, sharkFish, seahorse1, starfish2, turtles1) receive equal gradient updates.
    """
    def __init__(self, dataset: V28DegradationAwareDataset, batch_size: int = 8, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Group indices by scene
        self.scene_to_indices: Dict[str, List[int]] = {}
        for idx, (sc, _, _) in enumerate(dataset.items):
            if sc not in self.scene_to_indices:
                self.scene_to_indices[sc] = []
            self.scene_to_indices[sc].append(idx)
        self.scenes = sorted(list(self.scene_to_indices.keys()))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 9973)
        shuffled_pools = {sc: list(self.scene_to_indices[sc]) for sc in self.scenes}
        for sc in self.scenes:
            rng.shuffle(shuffled_pools[sc])

        total_samples = len(self.dataset)
        batches = []
        num_batches = total_samples // self.batch_size

        for _ in range(num_batches):
            batch = []
            # Round-robin across scenes for balanced representation
            selected_scenes = rng.sample(self.scenes, k=min(self.batch_size, len(self.scenes)))
            while len(selected_scenes) < self.batch_size:
                selected_scenes.append(rng.choice(self.scenes))

            for sc in selected_scenes:
                if not shuffled_pools[sc]:
                    shuffled_pools[sc] = list(self.scene_to_indices[sc])
                    rng.shuffle(shuffled_pools[sc])
                batch.append(shuffled_pools[sc].pop())
            batches.extend(batch)

        return iter(batches)

    def __len__(self) -> int:
        return (len(self.dataset) // self.batch_size) * self.batch_size
