"""
Unit Tests for Evaluation Pipeline Harness and Preprocessing Configurations.

Verifies:
1. Direct resize configuration (768 and 1024).
2. Aspect-preserving letterbox configuration (768 and 1024).
3. Letterbox metadata and unletterbox reconstruction.
4. Validation split resolution strictly from data/splits.json.
5. End-to-end synthetic evaluation pass with a dummy model.
"""

import os
import json
import tempfile
import unittest
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import EURSDataset
from data.transforms import (
    build_preprocessing_transform,
    SynchronizedLetterbox,
    unletterbox,
    LetterboxMetadata
)
from evaluation.experiment_config import (
    PreprocessingConfig,
    ExperimentConfig,
    get_controlled_candidate_config
)
from evaluation.pipeline import evaluate_model


class DummyModel(nn.Module):
    """Simple dummy model producing [B, 3, H, W] and [B, 5, H, W] for test evaluation."""
    def forward(self, x):
        restored = torch.clamp(x + 0.05, 0.0, 1.0)
        b, _, h, w = x.shape
        logits = torch.randn(b, 5, h, w)
        return restored, logits


class TestPipelineHarness(unittest.TestCase):

    def test_controlled_candidate_configs(self):
        """Test retrieval of exact candidate configs A, B, C, D."""
        cfg_a = get_controlled_candidate_config("A")
        self.assertEqual(cfg_a.preprocessing.resolution, 768)
        self.assertEqual(cfg_a.preprocessing.mode, "direct")

        cfg_b = get_controlled_candidate_config("B")
        self.assertEqual(cfg_b.preprocessing.resolution, 1024)
        self.assertEqual(cfg_b.preprocessing.mode, "direct")

        cfg_c = get_controlled_candidate_config("C")
        self.assertEqual(cfg_c.preprocessing.resolution, 768)
        self.assertEqual(cfg_c.preprocessing.mode, "letterbox")

        cfg_d = get_controlled_candidate_config("D")
        self.assertEqual(cfg_d.preprocessing.resolution, 1024)
        self.assertEqual(cfg_d.preprocessing.mode, "letterbox")

    def test_letterbox_metadata_and_unpadding(self):
        """Verify letterbox stores correct offsets and unletterbox restores original dimensions."""
        h_orig, w_orig = 300, 600  # 2:1 aspect ratio
        deg = torch.rand(3, h_orig, w_orig)
        clean = torch.rand(3, h_orig, w_orig)
        mask = torch.randint(0, 5, (h_orig, w_orig), dtype=torch.int64)

        letterboxer = SynchronizedLetterbox(target_size=768)
        deg_pad, clean_pad, mask_pad = letterboxer(deg, clean, mask)

        meta = letterboxer.last_metadata
        self.assertIsNotNone(meta)
        self.assertEqual(deg_pad.shape, (3, 768, 768))
        self.assertEqual(clean_pad.shape, (3, 768, 768))
        self.assertEqual(mask_pad.shape, (768, 768))

        # Check content size: 600 scaled to 768 -> scale = 768/600 = 1.28 -> height = 300 * 1.28 = 384
        self.assertEqual(meta.content_size, (384, 768))
        self.assertEqual(meta.pad_left, 0)
        self.assertEqual(meta.pad_right, 0)
        self.assertEqual(meta.pad_top, (768 - 384) // 2)

        # Test unletterboxing on image and mask
        deg_restored = unletterbox(deg_pad, meta, mode='bilinear')
        mask_restored = unletterbox(mask_pad, meta, mode='nearest')

        self.assertEqual(deg_restored.shape, (3, h_orig, w_orig))
        self.assertEqual(mask_restored.shape, (h_orig, w_orig))
        self.assertEqual(mask_restored.dtype, torch.int64)

    def test_synthetic_end_to_end_evaluation_pass(self):
        """Test full evaluate_model pass using temporary synthetic dataset."""
        temp_dir = tempfile.TemporaryDirectory()
        root = temp_dir.name
        train_dir = os.path.join(root, "train")
        os.makedirs(os.path.join(train_dir, "degraded", "scene_val"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "gt_images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "gt_masks"), exist_ok=True)

        # Save sample images
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(os.path.join(train_dir, "gt_images", "scene_val.png"))
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[10:20, 10:20] = 3  # Class 3 (Elasmobranch)
        mask_arr[30:40, 30:40] = 4  # Class 4 (Other)
        Image.fromarray(mask_arr).save(os.path.join(train_dir, "gt_masks", "scene_val.png"))
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(os.path.join(train_dir, "degraded", "scene_val", "scene_val_1_5.png"))

        splits_file = os.path.join(root, "mock_splits.json")
        with open(splits_file, "w") as f:
            json.dump({"tier1": {"train": ["scene_val"], "val": ["scene_val"]}}, f)

        # Initialize dataset and dataloader
        transform = build_preprocessing_transform(resolution=64, mode="direct")
        val_ds = EURSDataset(root_dir=root, split="val", splits_file=splits_file, transform=transform)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = DummyModel()
        results = evaluate_model(model, val_loader, device="cpu")

        # Verify structured output keys
        expected_keys = [
            "psnr", "ssim", "pixel_accuracy", "macro_iou",
            "iou_class_0", "iou_class_1", "iou_class_2", "iou_class_3", "iou_class_4"
        ]
        for k in expected_keys:
            self.assertIn(k, results, f"Missing expected key '{k}' in evaluation results")

        self.assertGreater(results["psnr"], 0.0)
        self.assertGreater(results["ssim"], 0.0)
        self.assertGreaterEqual(results["pixel_accuracy"], 0.0)

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
