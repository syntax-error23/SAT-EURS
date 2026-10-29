"""
Unit Tests for Evaluation Metrics (evaluation/metrics.py).

Verifies:
1. Exact mathematical calculation of PSNR on deterministic synthetic tensors.
2. Exact mathematical calculation of per-class IoU, macro IoU, and pixel accuracy.
3. Explicit exposure of rare Class 3 and Class 4 IoUs.
4. Correct handling of raw logits vs integer predicted masks.
5. Correct handling of padding exclusion via valid_mask.
6. MetricAccumulator batch aggregation.
"""

import math
import unittest
import numpy as np
import torch

from evaluation.metrics import (
    compute_segmentation_metrics,
    compute_restoration_metrics,
    compute_psnr,
    MetricAccumulator
)


class TestMetricsContract(unittest.TestCase):

    def test_psnr_exact_mathematical_values(self):
        """Verify PSNR on deterministic inputs with known MSE."""
        # 1. Identical images -> PSNR = 100.0 dB
        img_a = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        psnr_identical, norm_identical = compute_psnr(img_a, img_a)
        self.assertEqual(psnr_identical, 100.0)
        self.assertEqual(norm_identical, 1.0)

        # 2. Known difference: MSE = 0.01 -> 10 * log10(1 / 0.01) = 20.0 dB
        # Let diff be 0.1 uniformly: 0.1^2 = 0.01
        img_b = img_a + 0.1
        psnr_known, _ = compute_psnr(img_b, img_a)
        self.assertAlmostEqual(psnr_known, 20.0, places=4)

    def test_segmentation_exact_iou_and_accuracy(self):
        """Verify per-class IoU, macro IoU, and pixel accuracy on deterministic synthetic masks."""
        # 4x4 mask with 4 classes
        # Class 0: top-left (2x2) = 4 px
        # Class 1: top-right (2x2) = 4 px
        # Class 3 (Elasmo): bottom-left (2x2) = 4 px
        # Class 4 (Other): bottom-right (2x2) = 4 px
        target = torch.zeros((4, 4), dtype=torch.int64)
        target[0:2, 0:2] = 0
        target[0:2, 2:4] = 1
        target[2:4, 0:2] = 3
        target[2:4, 2:4] = 4

        # Predicted:
        # Class 0: perfect match (4 TP, 0 FP, 0 FN) -> IoU = 4/4 = 1.0
        # Class 1: 2 px match out of 4, 2 px predicted as class 2 -> intersection=2, union=4 -> IoU = 2/4 = 0.5
        # Class 3: perfect match (4 TP, 0 FP, 0 FN) -> IoU = 4/4 = 1.0
        # Class 4: 2 px match out of 4, 2 px predicted as class 2 -> intersection=2, union=4 -> IoU = 2/4 = 0.5
        pred = target.clone()
        pred[0, 2:4] = 2  # 2 pixels of Class 1 predicted as Class 2
        pred[2, 2:4] = 2  # 2 pixels of Class 4 predicted as Class 2

        metrics = compute_segmentation_metrics(pred=pred, target=target, num_classes=5)

        # 1. Explicit rare class exposure
        self.assertIn("iou_class_3", metrics)
        self.assertIn("iou_class_4", metrics)
        self.assertEqual(metrics["iou_class_0"], 1.0)
        self.assertAlmostEqual(metrics["iou_class_1"], 0.5, places=4)
        self.assertEqual(metrics["iou_class_3"], 1.0)
        self.assertAlmostEqual(metrics["iou_class_4"], 0.5, places=4)

        # 2. Pixel accuracy: 12 out of 16 correct = 12 / 16 = 0.75
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.75, places=4)

        # 3. Raw logits input format
        logits = torch.zeros((1, 5, 4, 4), dtype=torch.float32)
        for y in range(4):
            for x in range(4):
                logits[0, pred[y, x], y, x] = 10.0  # High logit for predicted class

        metrics_logits = compute_segmentation_metrics(pred=logits, target=target.unsqueeze(0), num_classes=5)
        self.assertAlmostEqual(metrics_logits["pixel_accuracy"], 0.75, places=4)
        self.assertEqual(metrics_logits["iou_class_0"], 1.0)
        self.assertEqual(metrics_logits["iou_class_3"], 1.0)
        self.assertAlmostEqual(metrics_logits["iou_class_4"], 0.5, places=4)

    def test_padding_exclusion_via_valid_mask(self):
        """Verify that padding pixels marked False in valid_mask are strictly excluded from metrics."""
        # 4x4 image with 2x4 padding at bottom
        target = torch.ones((4, 4), dtype=torch.int64) * 3  # Class 3 everywhere
        pred = target.clone()
        # Corrupt the bottom 2 rows (padding area)
        pred[2:4, :] = 0

        # Without valid_mask: accuracy = 8/16 = 0.5
        m_no_mask = compute_segmentation_metrics(pred=pred, target=target, num_classes=5)
        self.assertAlmostEqual(m_no_mask["pixel_accuracy"], 0.5, places=4)

        # With valid_mask excluding bottom 2 rows: accuracy = 8/8 = 1.0!
        valid_mask = torch.zeros((4, 4), dtype=torch.bool)
        valid_mask[0:2, :] = True

        m_with_mask = compute_segmentation_metrics(pred=pred, target=target, valid_mask=valid_mask, num_classes=5)
        self.assertAlmostEqual(m_with_mask["pixel_accuracy"], 1.0, places=4)
        self.assertEqual(m_with_mask["iou_class_3"], 1.0)

    def test_metric_accumulator(self):
        """Test accumulation across multiple batches."""
        acc = MetricAccumulator(num_classes=5)

        for _ in range(3):
            pred_rgb = torch.rand(2, 3, 32, 32)
            target_rgb = pred_rgb.clone()
            logits = torch.randn(2, 5, 32, 32)
            mask = logits.argmax(dim=1)

            acc.update(
                pred_rgb=pred_rgb,
                target_rgb=target_rgb,
                pred_logits_or_mask=logits,
                target_mask=mask
            )

        res = acc.compute()
        self.assertEqual(res["num_evaluated_samples"], 6)
        self.assertEqual(res["psnr"], 100.0)
        self.assertEqual(res["pixel_accuracy"], 1.0)
        self.assertEqual(res["macro_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
