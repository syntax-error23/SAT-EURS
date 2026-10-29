"""
Loss Contract Tests for SAT-EURS.

Verifies:
1. Restoration loss accepts aligned [B, 3, H, W] RGB tensors.
2. Segmentation loss accepts unnormalized [B, C, H, W] logits and [B, H, W] int64 masks.
3. MultiTaskLoss produces a finite scalar total loss.
4. Backpropagation produces finite gradients across model parameters.
5. Rejection of shape/dtype mismatches in loss inputs.
"""

import unittest
import torch
from losses.multi_task_loss import MultiTaskLoss
from losses.segmentation_loss import SegmentationLoss
from losses.restoration_loss import RestorationLoss
from models.eurs_net_v2 import EURSNetV2


class TestLossContract(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.loss_fn = MultiTaskLoss(weight_seg=1.0, weight_rest=0.5)

    def test_segmentation_loss_contract(self):
        """Test SegmentationLoss accepts raw logits and integer targets."""
        b, c, h, w = 2, 5, 64, 64
        logits = torch.randn(b, c, h, w, requires_grad=True)
        targets = torch.randint(0, c, (b, h, w), dtype=torch.int64)

        seg_fn = SegmentationLoss(num_classes=5)
        loss = seg_fn(logits, targets)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Backprop test
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_restoration_loss_contract(self):
        """Test RestorationLoss accepts aligned RGB tensors in [0, 1]."""
        b, c, h, w = 2, 3, 64, 64
        pred = torch.rand(b, c, h, w, requires_grad=True)
        target = torch.rand(b, c, h, w)

        rest_fn = RestorationLoss()
        loss = rest_fn(pred, target)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Backprop test
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_multitask_loss_and_gradient_flow(self):
        """Test end-to-end forward and backward pass with EURSNetV2."""
        model = EURSNetV2(num_classes=5)
        model.train()

        b, h, w = 1, 64, 64
        x = torch.rand(b, 3, h, w)
        clean_gt = torch.rand(b, 3, h, w)
        mask_gt = torch.randint(0, 5, (b, h, w), dtype=torch.int64)

        restored_rgb, seg_logits = model(x)
        total_loss, loss_dict = self.loss_fn(restored_rgb, seg_logits, clean_gt, mask_gt)

        self.assertTrue(torch.isfinite(total_loss).item())
        self.assertIn("loss_total", loss_dict)
        self.assertIn("loss_seg", loss_dict)
        self.assertIn("loss_rest", loss_dict)

        # Gradients propagate through both restoration and segmentation branches
        total_loss.backward()

        trainable_params_with_grad = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Parameter {name} has None gradient!")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf gradient!")
                trainable_params_with_grad += 1

        self.assertGreater(trainable_params_with_grad, 50)

    def test_loss_dimension_mismatch_fails(self):
        """Test that spatial mismatch between logits and targets raises an error."""
        logits = torch.randn(2, 5, 64, 64)
        mismatched_target = torch.randint(0, 5, (2, 32, 32), dtype=torch.int64)

        seg_fn = SegmentationLoss(num_classes=5)
        with self.assertRaises(RuntimeError):
            seg_fn(logits, mismatched_target)

    def test_restoration_loss_padding_invariance(self):
        """
        Verify restoration loss is strictly invariant to padding pixel manipulation.
        Padding pixels must contribute ZERO and not alter loss value or normalization.
        """
        b, c, h, w = 2, 3, 64, 64
        # Valid region is centered [16:48, 16:48] (32x32)
        valid_mask = torch.zeros((h, w), dtype=torch.bool)
        valid_mask[16:48, 16:48] = True

        # Base content
        content_pred = torch.rand(b, c, 32, 32)
        content_target = torch.rand(b, c, 32, 32)

        # Setup pred1/target1: zero padding
        pred1 = torch.zeros(b, c, h, w)
        target1 = torch.zeros(b, c, h, w)
        pred1[:, :, 16:48, 16:48] = content_pred
        target1[:, :, 16:48, 16:48] = content_target

        # Setup pred2/target2: identical content, but extreme non-zero noise in padding
        pred2 = torch.ones(b, c, h, w) * 0.99
        target2 = torch.ones(b, c, h, w) * 0.01
        pred2[:, :, 16:48, 16:48] = content_pred
        target2[:, :, 16:48, 16:48] = content_target

        rest_fn = RestorationLoss()
        loss1 = rest_fn(pred1, target1, valid_mask=valid_mask)
        loss2 = rest_fn(pred2, target2, valid_mask=valid_mask)

        # Invariance assertion: loss must be identical regardless of padding values
        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-6),
                        f"Restoration loss differed when padding changed: {loss1.item()} vs {loss2.item()}")

        # Content sensitivity check: modifying valid content MUST alter loss
        pred3 = pred1.clone()
        pred3[:, :, 24:32, 24:32] += 0.3
        loss3 = rest_fn(pred3, target1, valid_mask=valid_mask)
        self.assertNotAlmostEqual(loss1.item(), loss3.item(), places=4)

    def test_segmentation_loss_padding_invariance_and_no_class0_training(self):
        """
        Verify segmentation loss is strictly invariant to padding pixel manipulation.
        Padding pixels must contribute ZERO, cannot train Class 0, and receive zero gradients.
        """
        b, num_classes, h, w = 2, 5, 64, 64
        valid_mask = torch.zeros((h, w), dtype=torch.bool)
        valid_mask[16:48, 16:48] = True

        content_logits = torch.randn(b, num_classes, 32, 32)
        content_target = torch.randint(1, num_classes, (b, 32, 32), dtype=torch.int64)

        # Pair 1: padding targets are 0 (standard letterbox mask padding)
        logits1 = torch.zeros(b, num_classes, h, w, requires_grad=True)
        with torch.no_grad():
            logits1[:, :, 16:48, 16:48] = content_logits
        target1 = torch.zeros(b, h, w, dtype=torch.int64)
        target1[:, 16:48, 16:48] = content_target

        # Pair 2: identical content, but arbitrary noise in padding logits and targets set to 4
        logits2 = torch.zeros(b, num_classes, h, w)
        logits2[:, :, 16:48, 16:48] = content_logits
        # Drastically perturb padding logits
        logits2[:, 0, :16, :] = 10.0
        logits2[:, 4, 48:, :] = -10.0
        target2 = torch.full((b, h, w), 4, dtype=torch.int64)
        target2[:, 16:48, 16:48] = content_target

        seg_fn = SegmentationLoss(num_classes=num_classes)
        loss1 = seg_fn(logits1, target1, valid_mask=valid_mask)
        loss2 = seg_fn(logits2, target2, valid_mask=valid_mask)

        # Invariance assertion
        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-6),
                        f"Segmentation loss differed when padding changed: {loss1.item()} vs {loss2.item()}")

        # Gradient isolation check: backprop through logits1
        loss1.backward()
        self.assertIsNotNone(logits1.grad)
        # Content region MUST have non-zero gradients
        content_grad = logits1.grad[:, :, 16:48, 16:48]
        self.assertTrue((content_grad.abs() > 0.0).any(), "Content region received zero gradient!")
        # Padding region MUST have strictly zero gradients (cannot train Class 0 or any class on padding)
        pad_mask_expanded = (~valid_mask).unsqueeze(0).unsqueeze(0).expand_as(logits1)
        pad_grad = logits1.grad[pad_mask_expanded]
        self.assertTrue((pad_grad == 0.0).all(), "Padding pixels received non-zero gradient!")

    def test_multitask_loss_letterbox_propagation_and_invariance(self):
        """Verify MultiTaskLoss propagates valid_mask and is invariant to padding."""
        b, h, w = 1, 64, 64
        valid_mask = torch.zeros((h, w), dtype=torch.bool)
        valid_mask[16:48, 16:48] = True

        pred_rgb1 = torch.zeros(b, 3, h, w, requires_grad=True)
        clean_rgb1 = torch.zeros(b, 3, h, w)
        seg_logits1 = torch.zeros(b, 5, h, w, requires_grad=True)
        mask_target1 = torch.zeros(b, h, w, dtype=torch.int64)

        with torch.no_grad():
            pred_rgb1[:, :, 16:48, 16:48] = torch.rand(b, 3, 32, 32)
            clean_rgb1[:, :, 16:48, 16:48] = torch.rand(b, 3, 32, 32)
            seg_logits1[:, :, 16:48, 16:48] = torch.randn(b, 5, 32, 32)
            mask_target1[:, 16:48, 16:48] = torch.randint(0, 5, (b, 32, 32))

        pred_rgb2 = pred_rgb1.clone().detach()
        clean_rgb2 = clean_rgb1.clone().detach()
        seg_logits2 = seg_logits1.clone().detach()
        mask_target2 = mask_target1.clone().detach()

        # Corrupt padding regions completely
        pred_rgb2[:, :, :16, :] = 1.0
        clean_rgb2[:, :, :16, :] = 0.5
        seg_logits2[:, :, 48:, :] = 5.0
        mask_target2[:, 48:, :] = 3

        loss1, dict1 = self.loss_fn(pred_rgb1, seg_logits1, clean_rgb1, mask_target1, valid_mask=valid_mask)
        loss2, dict2 = self.loss_fn(pred_rgb2, seg_logits2, clean_rgb2, mask_target2, valid_mask=valid_mask)

        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-6))
        self.assertAlmostEqual(dict1["loss_seg"], dict2["loss_seg"], places=5)
        self.assertAlmostEqual(dict1["loss_rest"], dict2["loss_rest"], places=5)

        # Gradient propagation test through multitask loss
        loss1.backward()
        self.assertTrue(torch.isfinite(pred_rgb1.grad).all())
        self.assertTrue(torch.isfinite(seg_logits1.grad).all())
        # Zero gradients in padding
        pad_mask_rgb = (~valid_mask).unsqueeze(0).unsqueeze(0).expand_as(pred_rgb1)
        self.assertTrue((pred_rgb1.grad[pad_mask_rgb] == 0.0).all())

    def test_direct_resize_mode_unaffected(self):
        """Verify that when valid_mask is None (direct resize mode), standard behavior is preserved."""
        b, h, w = 1, 64, 64
        pred_rgb = torch.rand(b, 3, h, w)
        clean_rgb = torch.rand(b, 3, h, w)
        seg_logits = torch.randn(b, 5, h, w)
        mask_target = torch.randint(0, 5, (b, h, w), dtype=torch.int64)

        loss_none, dict_none = self.loss_fn(pred_rgb, seg_logits, clean_rgb, mask_target, valid_mask=None)
        loss_default, dict_default = self.loss_fn(pred_rgb, seg_logits, clean_rgb, mask_target)

        self.assertEqual(loss_none.item(), loss_default.item())
        self.assertEqual(dict_none["loss_seg"], dict_default["loss_seg"])
        self.assertEqual(dict_none["loss_rest"], dict_default["loss_rest"])

    def test_loss_matches_isolated_content_tensor(self):
        """
        Mathematical proof of zero padding influence and proper normalization:
        Loss on letterboxed canvas with valid_mask MUST match loss on the isolated content tensor directly.
        """
        b, h_content, w_content = 1, 32, 32
        content_pred = torch.rand(b, 3, h_content, w_content)
        content_target = torch.rand(b, 3, h_content, w_content)

        # Create padded canvas 64x64
        h_canvas, w_canvas = 64, 64
        padded_pred = torch.ones(b, 3, h_canvas, w_canvas) * 0.8
        padded_target = torch.ones(b, 3, h_canvas, w_canvas) * 0.2
        padded_pred[:, :, 16:48, 16:48] = content_pred
        padded_target[:, :, 16:48, 16:48] = content_target

        valid_mask = torch.zeros((h_canvas, w_canvas), dtype=torch.bool)
        valid_mask[16:48, 16:48] = True

        # Charbonnier comparison
        from losses.restoration_loss import CharbonnierLoss
        charb = CharbonnierLoss()
        loss_isolated = charb(content_pred, content_target)
        loss_letterbox = charb(padded_pred, padded_target, valid_mask=valid_mask)
        self.assertTrue(torch.allclose(loss_isolated, loss_letterbox, atol=1e-6),
                        f"Charbonnier letterbox loss {loss_letterbox.item()} != isolated {loss_isolated.item()}")

        # Laplacian comparison
        from losses.restoration_loss import LaplacianLoss
        lap = LaplacianLoss()
        loss_lap_isolated = lap(content_pred, content_target)
        loss_lap_letterbox = lap(padded_pred, padded_target, valid_mask=valid_mask)
        self.assertTrue(torch.allclose(loss_lap_isolated, loss_lap_letterbox, atol=1e-6),
                        f"Laplacian letterbox loss {loss_lap_letterbox.item()} != isolated {loss_lap_isolated.item()}")

        # Focal comparison
        from losses.segmentation_loss import ClassWeightedFocalLoss
        content_logits = torch.randn(b, 5, h_content, w_content)
        content_mask = torch.randint(0, 5, (b, h_content, w_content))
        padded_logits = torch.zeros(b, 5, h_canvas, w_canvas)
        padded_logits[:, :, 16:48, 16:48] = content_logits
        padded_mask = torch.zeros(b, h_canvas, w_canvas, dtype=torch.int64)
        padded_mask[:, 16:48, 16:48] = content_mask

        focal = ClassWeightedFocalLoss()
        loss_focal_isolated = focal(content_logits, content_mask)
        loss_focal_letterbox = focal(padded_logits, padded_mask, valid_mask=valid_mask)
        self.assertTrue(torch.allclose(loss_focal_isolated, loss_focal_letterbox, atol=1e-6),
                        f"Focal letterbox loss {loss_focal_letterbox.item()} != isolated {loss_focal_isolated.item()}")


if __name__ == "__main__":
    unittest.main()

