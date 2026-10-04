"""
Test Dataset Integrity & Pairing Invariants (Stage 2 Verification)
Verifies:
1. Scene-disjoint split invariant (zero data leakage).
2. Sample-level pairing invariant (degraded_scene == clean_scene == mask_scene).
3. Shape, dtype, and numerical range contracts for images and masks.
4. DataLoader batch collation sanity check.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import EURSDataset


def test_split_disjointness():
    print("=" * 60)
    print("Test 2.1: Scene-Disjoint Split Invariant Audit")
    print("=" * 60)
    manifest_path = 'd:/MACVI/splits/manifest.json'
    assert os.path.exists(manifest_path), f"Missing manifest: {manifest_path}"

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    train_scenes = set(manifest['train_scenes'])
    val_scenes = set(manifest['val_scenes'])

    assert len(train_scenes) > 0, "Train scenes empty!"
    assert len(val_scenes) > 0, "Val scenes empty!"
    assert train_scenes.isdisjoint(val_scenes), f"Data leakage! Overlap: {train_scenes & val_scenes}"

    print(f"  [PASS] Train scenes ({len(train_scenes)}): {sorted(list(train_scenes))}")
    print(f"  [PASS] Val scenes   ({len(val_scenes)}): {sorted(list(val_scenes))}")
    print("  [PASS] Overlap is strictly empty: 0% data leakage.\n")


def test_dataset_pairing_and_contracts():
    print("=" * 60)
    print("Test 2.2: Pairing Integrity & Tensor Contracts Audit")
    print("=" * 60)

    for split in ('train', 'val'):
        dataset = EURSDataset(
            data_dir='d:/MACVI/data/mini_dev',
            split=split,
            manifest_path='d:/MACVI/splits/manifest.json'
        )

        assert len(dataset) > 0, f"Dataset split '{split}' is empty!"

        for i in range(len(dataset)):
            deg, clean, mask, meta = dataset[i]

            # Invariant 1: Shapes
            assert deg.shape == (3, 1024, 1024), f"Bad degraded shape: {deg.shape}"
            assert clean.shape == (3, 1024, 1024), f"Bad clean shape: {clean.shape}"
            assert mask.shape == (1024, 1024), f"Bad mask shape: {mask.shape}"

            # Invariant 2: Dtypes
            assert deg.dtype == torch.float32, f"Bad deg dtype: {deg.dtype}"
            assert clean.dtype == torch.float32, f"Bad clean dtype: {clean.dtype}"
            assert mask.dtype == torch.int64, f"Bad mask dtype: {mask.dtype}"

            # Invariant 3: Value ranges
            assert torch.isfinite(deg).all(), "Degraded contains NaN/Inf!"
            assert torch.isfinite(clean).all(), "Clean contains NaN/Inf!"
            assert deg.min() >= 0.0 and deg.max() <= 1.0, f"Degraded out of [0, 1]: min={deg.min()}, max={deg.max()}"
            assert clean.min() >= 0.0 and clean.max() <= 1.0, f"Clean out of [0, 1]: min={clean.min()}, max={clean.max()}"

            # Invariant 4: Mask classes
            unique_classes = torch.unique(mask).tolist()
            assert set(unique_classes).issubset({0, 1, 2, 3, 4}), f"Invalid mask classes: {unique_classes}"

            # Invariant 5: Metadata pairing
            deg_scene = meta['scene_id']
            assert meta['clean_file'].startswith(deg_scene)
            assert meta['mask_file'].startswith(deg_scene)

        print(f"  [PASS] Verified {len(dataset)} samples in '{split}' split against all pairing & contract invariants.")
    print("All sample pairings and contracts passed!\n")


def test_dataloader_batch_collation():
    print("=" * 60)
    print("Test 2.3: DataLoader Batch Collation Audit")
    print("=" * 60)

    dataset = EURSDataset(
        data_dir='d:/MACVI/data/mini_dev',
        split='train',
        manifest_path='d:/MACVI/splits/manifest.json'
    )
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    deg_batch, clean_batch, mask_batch, meta = next(iter(loader))

    assert deg_batch.shape == (batch_size, 3, 1024, 1024), f"Bad deg batch shape: {deg_batch.shape}"
    assert clean_batch.shape == (batch_size, 3, 1024, 1024), f"Bad clean batch shape: {clean_batch.shape}"
    assert mask_batch.shape == (batch_size, 1024, 1024), f"Bad mask batch shape: {mask_batch.shape}"

    print(f"  [PASS] Degraded Batch Tensor: {list(deg_batch.shape)}, dtype={deg_batch.dtype}")
    print(f"  [PASS] Clean Batch Tensor:    {list(clean_batch.shape)}, dtype={clean_batch.dtype}")
    print(f"  [PASS] Mask Batch Tensor:     {list(mask_batch.shape)}, dtype={mask_batch.dtype}")
    print("DataLoader batch collation passed!\n")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  RUNNING STAGE 2 DATASET INTEGRITY VERIFICATION SUITE")
    print("#" * 60 + "\n")

    test_split_disjointness()
    test_dataset_pairing_and_contracts()
    test_dataloader_batch_collation()

    print("=" * 60)
    print("STAGE 2 VERIFICATION PASSED: ZERO DATA LEAKAGE & PERFECT PAIRING")
    print("=" * 60 + "\n")
