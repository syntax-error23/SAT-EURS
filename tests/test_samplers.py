"""
Samplers Unit Test (Phase 2A Pre-Flight)
Verifies:
1. StandardSampler & SceneBalancedSampler yield exactly len(dataset) items per epoch.
2. Identical optimizer step semantics: ceil(len(sampler) / (B * accum)) is identical.
3. StandardSampler yields valid complete permutations.
4. SceneBalancedSampler balances scene representation even with heavily skewed variants.
5. Determinism: epoch-based seed advancement is reproducible.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import math
from collections import Counter
from typing import Dict, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.samplers import StandardSampler, SceneBalancedSampler


class MockDataset(Dataset):
    def __init__(self, scene_counts: Dict[str, int]):
        self.samples: List[Dict[str, Any]] = []
        for scene_id, count in scene_counts.items():
            for v in range(count):
                self.samples.append({
                    "scene_id": scene_id,
                    "var_id": v
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def test_samplers():
    print("=" * 60)
    print("Test 2A.4: Dual Samplers & Fixed Epoch Step Semantics")
    print("=" * 60)

    # Heavily imbalanced mock dataset:
    # Scene A has 100 variants, Scene B has 10 variants, Scene C has 10 variants
    scene_counts = {"scene_A": 100, "scene_B": 10, "scene_C": 10}
    dataset = MockDataset(scene_counts)
    total_len = len(dataset) # 120
    assert total_len == 120

    std_sampler = StandardSampler(dataset, shuffle=True, seed=42)
    bal_sampler = SceneBalancedSampler(dataset, seed=42)

    # 1. Strict Sample Count Equality
    assert len(std_sampler) == total_len, f"Expected {total_len}, got {len(std_sampler)}"
    assert len(bal_sampler) == total_len, f"Expected {total_len}, got {len(bal_sampler)}"
    print(f"  [PASS] Sample Count Invariant: Both samplers report len = {total_len}")

    # 2. Strict Optimizer Step Semantics
    batch_size = 4
    grad_accum = 2
    eff_batch = batch_size * grad_accum
    expected_steps = math.ceil(total_len / eff_batch)

    std_indices = list(iter(std_sampler))
    bal_indices = list(iter(bal_sampler))

    assert len(std_indices) == total_len
    assert len(bal_indices) == total_len
    assert math.ceil(len(std_indices) / eff_batch) == expected_steps
    assert math.ceil(len(bal_indices) / eff_batch) == expected_steps
    print(f"  [PASS] Fixed Epoch Step Semantics: Both yield exactly {expected_steps} optimizer steps/epoch.")

    # 3. Standard Sampler is a complete permutation
    assert sorted(std_indices) == list(range(total_len))
    print("  [PASS] StandardSampler produces a complete permutation of all variations.")

    # 4. SceneBalancedSampler balances scene selection
    # In standard sampling, scene_A is 100/120 = 83.3% of draws.
    # In scene-balanced sampling, scene_A is drawn ~33.3% of the time (uniform across 3 scenes).
    bal_scenes = [dataset[i]["scene_id"] for i in bal_indices]
    scene_freq = Counter(bal_scenes)
    print(f"  [AUDIT] SceneBalanced draw frequencies over {total_len} samples: {dict(scene_freq)}")

    # Each of the 3 scenes should be around 120 / 3 = 40 samples (between 25 and 55 with high probability)
    for s in ["scene_A", "scene_B", "scene_C"]:
        assert 20 <= scene_freq[s] <= 60, f"Scene balance failed for {s}: {scene_freq[s]}"
    print("  [PASS] SceneBalancedSampler successfully eliminates scene variation skew.")

    # 5. Deterministic Seed Advancement
    std_sampler.set_epoch(0)
    e0_a = list(iter(std_sampler))
    std_sampler.set_epoch(0)
    e0_b = list(iter(std_sampler))
    assert e0_a == e0_b, "Sampler must be deterministic for identical epoch/seed!"

    std_sampler.set_epoch(1)
    e1 = list(iter(std_sampler))
    assert e0_a != e1, "Sampler should yield different order for next epoch!"

    bal_sampler.set_epoch(0)
    b0_a = list(iter(bal_sampler))
    bal_sampler.set_epoch(0)
    b0_b = list(iter(bal_sampler))
    assert b0_a == b0_b, "SceneBalancedSampler must be deterministic for identical epoch/seed!"

    bal_sampler.set_epoch(1)
    b1 = list(iter(bal_sampler))
    assert b0_a != b1, "SceneBalancedSampler should advance on next epoch!"
    print("  [PASS] Deterministic seed advancement across epochs verified.")

    print("=" * 60)
    print("WORK PACKAGE 2A.4 SAMPLERS VERIFICATION: 100% PASSED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_samplers()
