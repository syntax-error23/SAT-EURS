"""
Custom Samplers for EURS Training
1. StandardSampler: Standard random permutation of all dataset variation indices.
2. SceneBalancedSampler: Uniformly samples a scene, then uniformly samples a variation
   within that scene to eliminate bias toward scenes with larger variation counts.
Both samplers strictly guarantee identical sample counts per epoch:
  len(sampler) == len(dataset)
ensuring exact identical optimizer updates per epoch:
  N_updates = ceil(N_samples / (batch_size * grad_accum))
"""

import os
import sys
from typing import Iterator, List, Dict
import numpy as np
import torch
from torch.utils.data import Sampler


class StandardSampler(Sampler[int]):
    """
    Standard Sampler with deterministic seedable random permutation.
    Yields exactly len(dataset) indices per epoch.
    """
    def __init__(self, dataset, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(n).tolist()
        else:
            indices = list(range(n))
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class SceneBalancedSampler(Sampler[int]):
    """
    Scene-Balanced Sampler:
    1. Groups dataset sample indices by scene_id.
    2. At each iteration, uniformly selects a scene, then uniformly selects a variation within that scene.
    3. Yields exactly len(dataset) samples per epoch to preserve identical epoch step semantics.
    """
    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        # Group indices by scene_id
        self.scene_to_indices: Dict[str, List[int]] = {}
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            scene_id = sample.get("scene_id", f"scene_{idx}")
            if scene_id not in self.scene_to_indices:
                self.scene_to_indices[scene_id] = []
            self.scene_to_indices[scene_id].append(idx)

        self.scene_ids = sorted(list(self.scene_to_indices.keys()))
        assert len(self.scene_ids) > 0, "No scenes found in dataset for SceneBalancedSampler!"

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        n_total = len(self.dataset)
        sampled_indices: List[int] = []

        for _ in range(n_total):
            chosen_scene = rng.choice(self.scene_ids)
            variant_indices = self.scene_to_indices[chosen_scene]
            chosen_idx = rng.choice(variant_indices)
            sampled_indices.append(int(chosen_idx))

        return iter(sampled_indices)

    def __len__(self) -> int:
        return len(self.dataset)
