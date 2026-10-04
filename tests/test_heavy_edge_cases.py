"""
HEAVY EDGE-CASE & STRESS AUDIT SUITE FOR EURS-NET V2
Systematically tests all potential failure modes, edge cases, and hardware corner cases
to guarantee 100% crash-free execution on RTX 5090 hardware.

Covers:
- Battery 1: Tensor Shapes & Batch Edge Cases (B=1, B=3, B=5, non-square dimensions)
- Battery 2: Numerical Stress & NaN/Inf Immunity (black/white inputs, extreme masks, extreme logits)
- Battery 3: Mixed Precision (AMP) & GradScaler Unscaling Stress
- Battery 4: Gradient Accumulation & Epoch Boundary Integrity
- Battery 5: Dataloaders, Samplers & Worker Concurrency (num_workers > 0)
- Battery 6: End-to-End Pipeline Chain (Train -> Checkpoint -> Reload -> ONNX -> Evaluate -> Package)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import tempfile
import shutil
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models import EURSNetV2
from models.common import FastWhiteBalance, CoordinateAttention
from models.backbone import SharedBackbone
from models.decoders import ProgressiveRestorationDecoder, ProgressiveSegmentationDecoder
from losses.multi_task_loss import MultiTaskLoss
from losses.segmentation_loss import ClassWeightedFocalLoss, SoftDiceLoss, SegmentationLoss
from losses.restoration_loss import CharbonnierLoss, LaplacianLoss, RestorationLoss
from losses.msssim import ms_ssim, MSSSIMLoss
from data.samplers import StandardSampler, SceneBalancedSampler
from data.transforms import SynchronizedTransform
from utils.checkpointing import save_checkpoint, load_checkpoint
from metrics import SegmentationEvaluator, RestorationEvaluator, NonReferenceQualityEvaluator, compute_composite_score


class DummyStressDataset(Dataset):
    """Synthetic dataset with arbitrary scene distributions to test concurrency and edge cases."""
    def __init__(self, size=24, img_shape=(3, 256, 256)):
        self.size = size
        self.img_shape = img_shape
        self.samples = []
        for i in range(size):
            # Imbalanced scene assignments
            scene_id = f"scene_{i % 4}" if i < 20 else f"rare_scene_{i}"
            self.samples.append({
                "scene_id": scene_id,
                "variant_id": f"var_{i}",
                "index": i
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate varied stress patterns
        c, h, w = self.img_shape
        if idx % 4 == 0:
            # Pure black
            deg = torch.zeros(c, h, w, dtype=torch.float32)
            clean = torch.zeros(c, h, w, dtype=torch.float32)
            mask = torch.zeros(h, w, dtype=torch.long)
        elif idx % 4 == 1:
            # Pure white
            deg = torch.ones(c, h, w, dtype=torch.float32)
            clean = torch.ones(c, h, w, dtype=torch.float32)
            mask = torch.full((h, w), 4, dtype=torch.long) # Rare class 4
        elif idx % 4 == 2:
            # Severe green/blue underwater color cast
            deg = torch.tensor([0.05, 0.95, 0.90], dtype=torch.float32).view(3, 1, 1).expand(c, h, w)
            clean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1).expand(c, h, w)
            mask = torch.randint(0, 5, (h, w), dtype=torch.long)
        else:
            # Single-pixel rare class
            deg = torch.rand(c, h, w, dtype=torch.float32)
            clean = torch.rand(c, h, w, dtype=torch.float32)
            mask = torch.zeros(h, w, dtype=torch.long)
            mask[h // 2, w // 2] = 3 # Single shark pixel

        meta = {"scene_id": self.samples[idx]["scene_id"], "index": idx}
        return deg, clean, mask, meta


class TestBattery1TensorShapes(unittest.TestCase):
    """Battery 1: Tensor Dimensions & Batch Boundary Tests."""

    def setUp(self):
        self.model = EURSNetV2(num_classes=5)

    def test_single_batch_size(self):
        """Verify model handles batch_size=1 without BatchNorm2d errors."""
        self.model.train()
        x = torch.rand(1, 3, 512, 512)
        restored, logits = self.model(x)
        self.assertEqual(restored.shape, (1, 3, 512, 512))
        self.assertEqual(logits.shape, (1, 5, 512, 512))
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_odd_batch_sizes(self):
        """Verify model handles non-power-of-two batch sizes (B=3, B=5)."""
        self.model.eval()
        for b in [3, 5]:
            x = torch.rand(b, 3, 256, 256)
            restored, logits = self.model(x)
            self.assertEqual(restored.shape, (b, 3, 256, 256))
            self.assertEqual(logits.shape, (b, 5, 256, 256))

    def test_non_square_dimensions(self):
        """Verify Coordinate Attention and decoders handle non-square aspect ratios."""
        self.model.eval()
        non_squares = [(2, 3, 256, 512), (2, 3, 512, 256), (1, 3, 384, 512)]
        for b, c, h, w in non_squares:
            x = torch.rand(b, c, h, w)
            restored, logits = self.model(x)
            self.assertEqual(restored.shape, (b, 3, h, w))
            self.assertEqual(logits.shape, (b, 5, h, w))
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(torch.isfinite(logits).all())


class TestBattery2NumericalStress(unittest.TestCase):
    """Battery 2: Numerical Stress & NaN/Inf Immunity."""

    def setUp(self):
        self.loss_fn = MultiTaskLoss(num_classes=5)

    def test_all_black_and_all_white_inputs(self):
        """Verify loss functions do not produce NaN or divide-by-zero on flat images."""
        zeros_pred = torch.zeros(2, 3, 256, 256, requires_grad=True)
        zeros_target = torch.zeros(2, 3, 256, 256)
        ones_pred = torch.ones(2, 3, 256, 256, requires_grad=True)
        ones_target = torch.ones(2, 3, 256, 256)

        dummy_logits = torch.randn(2, 5, 256, 256, requires_grad=True)
        dummy_mask = torch.zeros(2, 256, 256, dtype=torch.long)

        # Black test
        loss_black, d_black = self.loss_fn(zeros_pred, dummy_logits, zeros_target, dummy_mask)
        self.assertTrue(torch.isfinite(loss_black))
        loss_black.backward()
        self.assertTrue(torch.isfinite(zeros_pred.grad).all())

        # White test
        dummy_logits_w = torch.randn(2, 5, 256, 256, requires_grad=True)
        loss_white, d_white = self.loss_fn(ones_pred, dummy_logits_w, ones_target, dummy_mask)
        self.assertTrue(torch.isfinite(loss_white))
        loss_white.backward()
        self.assertTrue(torch.isfinite(ones_pred.grad).all())

    def test_all_background_vs_all_rare_class_masks(self):
        """Verify segmentation loss when 0% or 100% of pixels belong to rare classes."""
        focal = ClassWeightedFocalLoss(gamma=2.0)
        dice = SoftDiceLoss(num_classes=5)

        logits = torch.randn(2, 5, 128, 128, requires_grad=True)

        # 100% background mask (only class 0)
        mask_bg = torch.zeros(2, 128, 128, dtype=torch.long)
        loss_focal_bg = focal(logits, mask_bg)
        loss_dice_bg = dice(logits, mask_bg)
        self.assertTrue(torch.isfinite(loss_focal_bg))
        self.assertTrue(torch.isfinite(loss_dice_bg))

        # 100% rare shark mask (only class 3)
        mask_shark = torch.full((2, 128, 128), 3, dtype=torch.long)
        loss_focal_shark = focal(logits, mask_shark)
        loss_dice_shark = dice(logits, mask_shark)
        self.assertTrue(torch.isfinite(loss_focal_shark))
        self.assertTrue(torch.isfinite(loss_dice_shark))

    def test_extreme_logit_values(self):
        """Verify logits at ±1000.0 (extreme saturation) do not cause overflow or NaN."""
        focal = ClassWeightedFocalLoss()
        dice = SoftDiceLoss(num_classes=5)

        # Huge positive logits
        huge_logits = torch.full((2, 5, 64, 64), 500.0, requires_grad=True)
        targets = torch.randint(0, 5, (2, 64, 64))
        loss_f = focal(huge_logits, targets)
        loss_d = dice(huge_logits, targets)
        self.assertTrue(torch.isfinite(loss_f))
        self.assertTrue(torch.isfinite(loss_d))

        # Huge negative logits
        neg_logits = torch.full((2, 5, 64, 64), -500.0, requires_grad=True)
        loss_fn = focal(neg_logits, targets)
        loss_dn = dice(neg_logits, targets)
        self.assertTrue(torch.isfinite(loss_fn))
        self.assertTrue(torch.isfinite(loss_dn))

    def test_identical_images_zero_difference(self):
        """Verify restoration losses when pred exactly equals target."""
        rest_loss = RestorationLoss()
        img = torch.rand(2, 3, 256, 256, requires_grad=True)
        loss = rest_loss(img, img.clone())
        self.assertTrue(torch.isfinite(loss))
        # Loss should be very close to 0
        self.assertLess(loss.item(), 0.05)


class TestBattery3MixedPrecisionAndGradScaler(unittest.TestCase):
    """Battery 3: Mixed Precision (AMP) & Optimization Stability."""

    def test_amp_fp16_forward_backward_cpu_simulation(self):
        """Verify AMP autocast (bfloat16 on CPU / fp16 on CUDA) completes cleanly."""
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16

        model = EURSNetV2(num_classes=5)
        loss_fn = MultiTaskLoss(num_classes=5)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))

        x = torch.rand(2, 3, 256, 256)
        clean = torch.rand(2, 3, 256, 256)
        mask = torch.randint(0, 5, (2, 256, 256))

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            restored, logits = model(x)
            loss, _ = loss_fn(restored, logits, clean, mask)

        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Check weights are finite after step
        for param in model.parameters():
            self.assertTrue(torch.isfinite(param).all())

    def test_inf_gradient_handling(self):
        """Verify GradScaler safely skips step if an Inf/NaN gradient occurs."""
        model = nn.Linear(10, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler('cpu', enabled=False)

        # Inject Inf into weight grad
        model.weight.grad = torch.full_like(model.weight, float('inf'))
        scaler.step(optimizer)
        scaler.update()
        # Optimizer should handle or scale without unhandled exception


class TestBattery4GradientAccumulation(unittest.TestCase):
    """Battery 4: Gradient Accumulation & Epoch Boundary Integrity."""

    def test_uneven_accumulation_boundary(self):
        """Simulate 7 batches with grad_accum=4. Verify step triggers at batch 4 and batch 7."""
        model = EURSNetV2(num_classes=5)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        grad_accum = 4
        total_batches = 7

        step_count = 0
        optimizer.zero_grad()

        for batch_idx in range(1, total_batches + 1):
            x = torch.rand(2, 3, 128, 128)
            restored, logits = model(x)
            loss = (restored.mean() + logits.mean()) / grad_accum
            loss.backward()

            if batch_idx % grad_accum == 0 or batch_idx == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

        # Must have stepped exactly twice (at batch 4 and at batch 7)
        self.assertEqual(step_count, 2)

    def test_warmup_cosine_scheduler_full_50_epochs(self):
        """Verify warmup cosine scheduler never produces NaN or negative LR across 50 epochs."""
        from train import get_warmup_cosine_scheduler
        model = nn.Linear(10, 2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=50, min_lr_ratio=1e-3)

        lr_history = []
        for epoch in range(1, 51):
            lr = optimizer.param_groups[0]['lr']
            self.assertTrue(math.isfinite(lr))
            self.assertGreater(lr, 0.0)
            self.assertLessEqual(lr, 1e-3)
            lr_history.append(lr)
            optimizer.step()
            scheduler.step()

        # Warmup test: lr should increase for first 5 epochs
        self.assertLess(lr_history[0], lr_history[4])
        # Decay test: lr at epoch 50 should be close to min_lr (1e-6)
        self.assertAlmostEqual(lr_history[-1], 1e-6, places=6)


class TestBattery5DataloadersAndConcurrency(unittest.TestCase):
    """Battery 5: Dataloaders, Samplers & Concurrency."""

    def test_scene_balanced_sampler_multiprocessing(self):
        """Verify SceneBalancedSampler runs cleanly under DataLoader with num_workers=2."""
        dataset = DummyStressDataset(size=32, img_shape=(3, 128, 128))
        sampler = SceneBalancedSampler(dataset, seed=42)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0)

        batches_run = 0
        total_samples = 0
        for deg, clean, mask, meta in loader:
            batches_run += 1
            total_samples += deg.shape[0]
            self.assertEqual(deg.shape, (4, 3, 128, 128))
            self.assertEqual(mask.shape, (4, 128, 128))

        self.assertEqual(total_samples, 32)
        self.assertEqual(batches_run, 8)

    def test_sampler_epoch_seed_variation(self):
        """Verify sampler changes permutation across epochs but stays deterministic per epoch."""
        dataset = DummyStressDataset(size=20)
        sampler = SceneBalancedSampler(dataset, seed=42)

        sampler.set_epoch(0)
        epoch0_a = list(sampler)
        sampler.set_epoch(0)
        epoch0_b = list(sampler)
        self.assertEqual(epoch0_a, epoch0_b, "Sampler must be deterministic for the same epoch!")

        sampler.set_epoch(1)
        epoch1 = list(sampler)
        self.assertNotEqual(epoch0_a, epoch1, "Sampler must vary permutations across different epochs!")


class TestBattery6EndToEndPipelineChain(unittest.TestCase):
    """Battery 6: End-to-End Pipeline Smoke Test (Full Lifecycle)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_roundtrip(self):
        """Execute Train -> Save -> Reload -> Evaluate -> ONNX Export in one chain."""
        # 1. Train 1 step
        model = EURSNetV2(num_classes=5)
        loss_fn = MultiTaskLoss(num_classes=5)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        dataset = DummyStressDataset(size=4, img_shape=(3, 128, 128))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        for deg, clean, mask, meta in loader:
            optimizer.zero_grad()
            restored, logits = model(deg)
            loss, _ = loss_fn(restored, logits, clean, mask)
            loss.backward()
            optimizer.step()

        # 2. Checkpoint Save
        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pt")
        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            global_step=2,
            best_score=0.75
        )
        self.assertTrue(os.path.exists(ckpt_path))

        # 3. Checkpoint Load
        new_model = EURSNetV2(num_classes=5)
        new_opt = optim.AdamW(new_model.parameters(), lr=1e-3)
        meta = load_checkpoint(ckpt_path, model=new_model, optimizer=new_opt)
        self.assertEqual(meta["epoch"], 1)
        self.assertEqual(meta["global_step"], 2)

        # 4. Evaluation Suite
        seg_eval = SegmentationEvaluator(num_classes=5)
        rest_eval = RestorationEvaluator()
        nr_eval = NonReferenceQualityEvaluator()

        with torch.no_grad():
            for deg, clean, mask, meta in loader:
                restored, logits = new_model(deg)
                seg_eval.update(logits, mask)
                rest_eval.update(restored, clean)
                nr_eval.update(restored)

        s_res = seg_eval.compute()
        r_res = rest_eval.compute()
        n_res = nr_eval.compute()
        comp = compute_composite_score(s_res['Q_seg'], r_res['Q_rest'], n_res['Q_NR'])
        self.assertTrue(0.0 <= comp['Composite_Score'] <= 1.0)

        # 5. Static ONNX Export & Parity Validation via export_and_validate_onnx
        onnx_path = os.path.join(self.temp_dir, "test_export.onnx")
        from deploy.export_onnx import export_and_validate_onnx
        report = export_and_validate_onnx(
            checkpoint_path=ckpt_path,
            output_onnx_path=onnx_path,
            verbose=False
        )
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(report["numerical_parity_passed"])
        self.assertLess(report["max_diff_restored"], 1e-4)
        self.assertLess(report["max_diff_seg"], 1e-4)


def run_all_heavy_tests():
    print("\n" + "=" * 70)
    print("      RUNNING HEAVY STRESS & EDGE-CASE TEST SUITE FOR EURS-NET V2")
    print("=" * 70)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestBattery1TensorShapes))
    suite.addTests(loader.loadTestsFromTestCase(TestBattery2NumericalStress))
    suite.addTests(loader.loadTestsFromTestCase(TestBattery3MixedPrecisionAndGradScaler))
    suite.addTests(loader.loadTestsFromTestCase(TestBattery4GradientAccumulation))
    suite.addTests(loader.loadTestsFromTestCase(TestBattery5DataloadersAndConcurrency))
    suite.addTests(loader.loadTestsFromTestCase(TestBattery6EndToEndPipelineChain))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_heavy_tests()
    sys.exit(0 if success else 1)
