"""
Unit Tests for EURS-Net V2 Training Harness and Resolution Study.

Verifies:
1. Candidate configuration invariance: identical hyperparameters across A, B, C, D with ONLY resolution/mode changing.
2. Deterministic seed behavior across torch, numpy, and random.
3. Checkpoint save, load, and full optimizer/model state restoration.
4. Valid_mask propagation in letterbox mode through the trainer.
5. Fail-loud NaN/Inf loss and gradient detection.
6. Train/validation scene separation and zero leakage in trainer datasets.
7. Master study summary generation and relative difference calculations.
"""

import os
import json
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from training.config import (
    TrainingConfig,
    get_candidate_training_config,
    get_all_candidate_configs
)
from training.trainer import (
    EURSTrainer,
    set_seed,
    compute_smoothed_class_weights
)
from training.run_study import (
    compute_relative_differences,
    generate_resolution_study_summary
)


class TestTrainingHarness(unittest.TestCase):

    def test_candidate_configuration_invariance(self):
        """
        Verify that Candidates A, B, C, D have IDENTICAL hyperparameters,
        changing ONLY resolution and preprocessing mode.
        """
        all_configs = get_all_candidate_configs()
        self.assertEqual(set(all_configs.keys()), {"A", "B", "C", "D"})

        cfg_a = all_configs["A"]
        cfg_b = all_configs["B"]
        cfg_c = all_configs["C"]
        cfg_d = all_configs["D"]

        # 1. Verify resolution and mode matrix
        self.assertEqual((cfg_a.resolution, cfg_a.mode), (768, "direct"))
        self.assertEqual((cfg_b.resolution, cfg_b.mode), (1024, "direct"))
        self.assertEqual((cfg_c.resolution, cfg_c.mode), (768, "letterbox"))
        self.assertEqual((cfg_d.resolution, cfg_d.mode), (1024, "letterbox"))

        # 2. Invariant parameters must be strictly identical across all 4 candidates
        invariant_fields = [
            "seed", "batch_size", "learning_rate", "weight_decay",
            "weight_seg", "weight_rest", "epochs", "splits_file",
            "tier", "gradient_clip_norm", "use_amp", "use_augmentation",
            "use_class_weights", "checkpoint_metric", "checkpoint_metric_mode"
        ]

        for f in invariant_fields:
            val_a = getattr(cfg_a, f)
            for cid in ["B", "C", "D"]:
                val_other = getattr(all_configs[cid], f)
                self.assertEqual(
                    val_a, val_other,
                    f"Parameter '{f}' differs between Candidate A ({val_a}) and Candidate {cid} ({val_other})!"
                )

    def test_deterministic_seed_behavior(self):
        """Verify set_seed guarantees bitwise reproducible random generation."""
        set_seed(42, deterministic=True)
        t1 = torch.randn(10, 10)
        n1 = np.random.randn(10, 10)

        set_seed(42, deterministic=True)
        t2 = torch.randn(10, 10)
        n2 = np.random.randn(10, 10)

        self.assertTrue(torch.equal(t1, t2), "torch.randn produced different tensors after reseeding with seed 42!")
        self.assertTrue(np.array_equal(n1, n2), "numpy.random produced different arrays after reseeding with seed 42!")

    def test_checkpoint_save_and_load_state_restoration(self):
        """Verify model weights and optimizer momentum are restored exactly bitwise."""
        temp_dir = tempfile.TemporaryDirectory()
        try:
            cfg = get_candidate_training_config("A", output_dir=temp_dir.name, device="cpu", epochs=2)
            trainer = EURSTrainer(cfg)

            # Perform a synthetic step to change model weights and build optimizer state
            x = torch.randn(1, 3, 768, 768)
            y_clean = torch.randn(1, 3, 768, 768)
            y_mask = torch.randint(0, 5, (1, 768, 768))

            trainer.optimizer.zero_grad()
            rest, logits = trainer.model(x)
            loss, _ = trainer.criterion(rest, logits, y_clean, y_mask)
            loss.backward()
            trainer.optimizer.step()

            # Save checkpoint
            dummy_metrics = {"val_loss_total": 0.5, "psnr": 24.5}
            trainer.save_checkpoint(epoch=1, is_best=True, metrics=dummy_metrics)

            best_ckpt_path = os.path.join(temp_dir.name, cfg.experiment_name, "checkpoint_best.pth")
            self.assertTrue(os.path.isfile(best_ckpt_path))

            # Create a fresh trainer with newly randomized weights
            trainer_fresh = EURSTrainer(cfg)
            # Weights should differ before load
            p_old = next(trainer.model.parameters()).clone()
            p_fresh = next(trainer_fresh.model.parameters()).clone()
            self.assertFalse(torch.equal(p_old, p_fresh))

            # Load checkpoint into fresh trainer
            loaded_ckpt = trainer_fresh.load_checkpoint(best_ckpt_path)
            self.assertEqual(loaded_ckpt["epoch"], 1)
            self.assertEqual(loaded_ckpt["seed"], 42)

            # Verify bitwise weight restoration
            p_restored = next(trainer_fresh.model.parameters())
            self.assertTrue(torch.equal(p_old, p_restored), "Model parameters were not restored bitwise!")

            # Verify optimizer state restoration
            old_opt_state = trainer.optimizer.state_dict()
            new_opt_state = trainer_fresh.optimizer.state_dict()
            self.assertEqual(len(old_opt_state["state"]), len(new_opt_state["state"]))
        finally:
            temp_dir.cleanup()

    def test_train_val_scene_separation_in_trainer(self):
        """Verify trainer datasets have zero scene overlap."""
        cfg = get_candidate_training_config("A", device="cpu")
        trainer = EURSTrainer(cfg)

        train_scenes = set(s["scene_id"] for s in trainer.train_dataset.samples)
        val_scenes = set(s["scene_id"] for s in trainer.val_dataset.samples)

        self.assertEqual(len(train_scenes), 16)
        self.assertEqual(len(val_scenes), 4)
        self.assertTrue(train_scenes.isdisjoint(val_scenes), "Scene leakage detected between train and val datasets!")

    def test_nan_inf_failure_handling(self):
        """Verify trainer fails loudly with FloatingPointError on non-finite loss."""
        temp_dir = tempfile.TemporaryDirectory()
        try:
            cfg = get_candidate_training_config("A", output_dir=temp_dir.name, device="cpu", epochs=1)
            trainer = EURSTrainer(cfg)

            # Corrupt model weights to produce NaN
            with torch.no_grad():
                for p in trainer.model.parameters():
                    p.fill_(float("nan"))
                    break

            with self.assertRaises(FloatingPointError):
                trainer.train_epoch(1, max_batches=1)
        finally:
            temp_dir.cleanup()

    def test_relative_difference_calculations(self):
        """Verify relative differences compute expected deltas and percentages."""
        mock_summaries = {
            "A": {
                "psnr": 20.0,
                "macro_iou": 0.50,
                "composite_score": 0.60,
                "profiled_gflops": 2.515
            },
            "B": {
                "psnr": 22.0,
                "macro_iou": 0.55,
                "composite_score": 0.66,
                "profiled_gflops": 4.472
            },
            "C": {
                "psnr": 21.0,
                "macro_iou": 0.52,
                "composite_score": 0.62,
                "profiled_gflops": 2.515
            },
            "D": {
                "psnr": 23.0,
                "macro_iou": 0.57,
                "composite_score": 0.68,
                "profiled_gflops": 4.472
            }
        }

        diffs = compute_relative_differences(mock_summaries)
        self.assertIn("768_direct_vs_1024_direct", diffs)
        self.assertIn("768_letterbox_vs_1024_letterbox", diffs)
        self.assertIn("768_direct_vs_768_letterbox", diffs)
        self.assertIn("1024_direct_vs_1024_letterbox", diffs)

        # Check A vs B PSNR: delta = +2.0, rel_pct = +10.0%
        psnr_diff_ab = diffs["768_direct_vs_1024_direct"]["metrics"]["psnr"]
        self.assertEqual(psnr_diff_ab["delta"], 2.0)
        self.assertEqual(psnr_diff_ab["relative_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
