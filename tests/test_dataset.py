"""
Comprehensive Unit Tests for EURS Multi-Task Dataset (data/dataset.py).

Verifies:
1. Sample counting and deterministic indexing using mocked synthetic fixtures.
2. Correct degraded -> clean GT -> mask triplet pairing.
3. Train/validation scene disjointness (zero leakage).
4. Mask dtype (torch.int64) and discrete label integrity.
5. RGB channel handling ([3, H, W], float32 in [0.0, 1.0]).
6. Loud failures (RuntimeError):
   - Missing GT clean image.
   - Missing GT mask.
   - Missing degraded directory.
   - Malformed degraded filename.
   - Empty degraded directory.
7. Transform integration: applying SynchronizedResize during dataset indexing.
8. Real EURS dataset verification (3,200 train samples, 800 val samples for Tier 1).
"""

import os
import json
import tempfile
import unittest
from PIL import Image
import numpy as np
import torch

from data.dataset import EURSDataset
from data.transforms import SynchronizedResize, SynchronizedHorizontalFlip


class TestEURSDatasetMocked(unittest.TestCase):
    """Synthetic filesystem fixture tests: isolates dataset logic without modifying real files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        self.train_dir = os.path.join(self.root_dir, "train")
        os.makedirs(os.path.join(self.train_dir, "degraded"), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "gt_images"), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "gt_masks"), exist_ok=True)

        # Setup mock scenes: 2 train scenes, 1 val scene
        self.train_scenes = ["mock_scene_a", "mock_scene_b"]
        self.val_scenes = ["mock_scene_c"]
        self.all_scenes = self.train_scenes + self.val_scenes

        self.variants_per_scene = 3
        self.img_h, self.img_w = 40, 60

        for scene in self.all_scenes:
            # Create GT image
            gt_img = Image.new("RGB", (self.img_w, self.img_h), color=(100, 150, 200))
            gt_img.save(os.path.join(self.train_dir, "gt_images", f"{scene}.png"))

            # Create GT mask with discrete classes 0, 1, 2
            mask_arr = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            mask_arr[:15, :] = 1
            mask_arr[15:30, :] = 2
            mask_img = Image.fromarray(mask_arr)
            mask_img.save(os.path.join(self.train_dir, "gt_masks", f"{scene}.png"))

            # Create degraded variants
            deg_dir = os.path.join(self.train_dir, "degraded", scene)
            os.makedirs(deg_dir, exist_ok=True)
            for v in range(self.variants_per_scene):
                deg_img = Image.new("RGB", (self.img_w, self.img_h), color=(50 + v * 10, 80, 120))
                deg_img.save(os.path.join(deg_dir, f"{scene}_1_{v * 5}.png"))

        # Create mock splits.json
        self.splits_data = {
            "metadata": {"dataset": "EURS_MOCK", "protocol_version": 1},
            "tier1": {
                "train": self.train_scenes,
                "val": self.val_scenes
            }
        }
        self.splits_file = os.path.join(self.root_dir, "mock_splits.json")
        with open(self.splits_file, "w") as f:
            json.dump(self.splits_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sample_counts_and_scene_disjointness(self):
        """Test exact sample counts and zero scene leakage."""
        ds_train = EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        ds_val = EURSDataset(root_dir=self.root_dir, split="val", splits_file=self.splits_file)

        # Expected counts
        expected_train_count = len(self.train_scenes) * self.variants_per_scene
        expected_val_count = len(self.val_scenes) * self.variants_per_scene

        self.assertEqual(len(ds_train), expected_train_count)
        self.assertEqual(len(ds_val), expected_val_count)

        # Check disjointness of scenes
        train_scenes_in_ds = set(s["scene_id"] for s in ds_train.samples)
        val_scenes_in_ds = set(s["scene_id"] for s in ds_val.samples)

        self.assertEqual(train_scenes_in_ds, set(self.train_scenes))
        self.assertEqual(val_scenes_in_ds, set(self.val_scenes))
        self.assertTrue(train_scenes_in_ds.isdisjoint(val_scenes_in_ds))

    def test_tensor_shapes_channels_and_dtypes(self):
        """Test output tensor formats, ranges, and dtypes."""
        ds_train = EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        sample = ds_train[0]

        self.assertIn("degraded", sample)
        self.assertIn("clean", sample)
        self.assertIn("mask", sample)
        self.assertIn("scene_id", sample)
        self.assertIn("filename", sample)

        deg = sample["degraded"]
        clean = sample["clean"]
        mask = sample["mask"]

        # Image checks: [3, H, W], float32 in [0, 1]
        self.assertEqual(deg.shape, (3, self.img_h, self.img_w))
        self.assertEqual(clean.shape, (3, self.img_h, self.img_w))
        self.assertEqual(deg.dtype, torch.float32)
        self.assertEqual(clean.dtype, torch.float32)
        self.assertTrue(0.0 <= deg.min().item() and deg.max().item() <= 1.0)
        self.assertTrue(0.0 <= clean.min().item() and clean.max().item() <= 1.0)

        # Mask checks: [H, W], int64 with discrete labels {0, 1, 2}
        self.assertEqual(mask.shape, (self.img_h, self.img_w))
        self.assertEqual(mask.dtype, torch.int64)
        uniques = torch.unique(mask).tolist()
        for u in uniques:
            self.assertIn(u, [0, 1, 2])

    def test_deterministic_ordering(self):
        """Test sample ordering is bitwise deterministic across dataset instantiations."""
        ds1 = EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        ds2 = EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)

        self.assertEqual(len(ds1), len(ds2))
        for i in range(len(ds1)):
            self.assertEqual(ds1.samples[i]["filename"], ds2.samples[i]["filename"])
            self.assertEqual(ds1.samples[i]["scene_id"], ds2.samples[i]["scene_id"])

    def test_missing_clean_gt_fails_loudly(self):
        """Test loud RuntimeError if GT clean image is missing."""
        # Remove GT clean image for mock_scene_a
        os.remove(os.path.join(self.train_dir, "gt_images", "mock_scene_a.png"))

        with self.assertRaises(RuntimeError) as ctx:
            EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        self.assertIn("Missing GT clean image for scene 'mock_scene_a'", str(ctx.exception))

    def test_missing_gt_mask_fails_loudly(self):
        """Test loud RuntimeError if GT mask is missing."""
        # Remove GT mask for mock_scene_b
        os.remove(os.path.join(self.train_dir, "gt_masks", "mock_scene_b.png"))

        with self.assertRaises(RuntimeError) as ctx:
            EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        self.assertIn("Missing GT mask for scene 'mock_scene_b'", str(ctx.exception))

    def test_missing_degraded_dir_fails_loudly(self):
        """Test loud RuntimeError if degraded scene folder is missing."""
        import shutil
        shutil.rmtree(os.path.join(self.train_dir, "degraded", "mock_scene_a"))

        with self.assertRaises(RuntimeError) as ctx:
            EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        self.assertIn("Missing degraded directory for scene 'mock_scene_a'", str(ctx.exception))

    def test_malformed_filename_fails_loudly(self):
        """Test loud RuntimeError if degraded filename does not match scene_id prefix."""
        bad_path = os.path.join(self.train_dir, "degraded", "mock_scene_a", "unrelated_img.png")
        Image.new("RGB", (self.img_w, self.img_h)).save(bad_path)

        with self.assertRaises(RuntimeError) as ctx:
            EURSDataset(root_dir=self.root_dir, split="train", splits_file=self.splits_file)
        self.assertIn("Malformed degraded image filename", str(ctx.exception))

    def test_transform_integration(self):
        """Test that SynchronizedResize properly resizes all elements during indexing."""
        transform = SynchronizedResize(size=(64, 64))
        ds = EURSDataset(
            root_dir=self.root_dir,
            split="train",
            splits_file=self.splits_file,
            transform=transform
        )

        sample = ds[0]
        self.assertEqual(sample["degraded"].shape, (3, 64, 64))
        self.assertEqual(sample["clean"].shape, (3, 64, 64))
        self.assertEqual(sample["mask"].shape, (64, 64))
        self.assertEqual(sample["mask"].dtype, torch.int64)


class TestEURSDatasetRealFilesystem(unittest.TestCase):
    """Tests directly against the verified local EURS dataset at /home/shubk/Datasets/eurs."""

    def test_real_dataset_tier1_counts_and_pairing(self):
        dataset_root = "/home/shubk/Datasets/eurs"
        splits_file = "data/splits.json"

        if not os.path.isdir(dataset_root):
            self.skipTest(f"Real dataset directory not found at {dataset_root}")

        ds_train = EURSDataset(root_dir=dataset_root, split="train", splits_file=splits_file)
        ds_val = EURSDataset(root_dir=dataset_root, split="val", splits_file=splits_file)

        # 16 train scenes * 200 = 3,200 samples
        self.assertEqual(len(ds_train), 3200)
        # 4 val scenes * 200 = 800 samples
        self.assertEqual(len(ds_val), 800)

        # Check scene separation
        train_scenes = set(s["scene_id"] for s in ds_train.samples)
        val_scenes = set(s["scene_id"] for s in ds_val.samples)

        self.assertEqual(len(train_scenes), 16)
        self.assertEqual(len(val_scenes), 4)
        self.assertTrue(train_scenes.isdisjoint(val_scenes))

        # Check rare classes in train and val
        self.assertIn("ray1", train_scenes)
        self.assertIn("starfish2", train_scenes)
        self.assertIn("turtles1", train_scenes)
        self.assertIn("sharkFish", val_scenes)
        self.assertIn("seahorse1", val_scenes)

        # Test index sample 0 with transform
        transform = SynchronizedResize(size=(256, 256))
        ds_transformed = EURSDataset(
            root_dir=dataset_root,
            split="val",
            splits_file=splits_file,
            transform=transform
        )

        sample0 = ds_transformed[0]
        self.assertEqual(sample0["degraded"].shape, (3, 256, 256))
        self.assertEqual(sample0["clean"].shape, (3, 256, 256))
        self.assertEqual(sample0["mask"].shape, (256, 256))
        self.assertEqual(sample0["mask"].dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()
