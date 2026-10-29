"""
Model Contract Tests for EURS-Net V2.

Verifies:
1. Output contract at 512x512, 768x768, 1024x1024, and non-square 384x768.
2. Input and output tensor shapes, dtypes, and value ranges:
   - restored image in [0.0, 1.0], float32.
   - seg_logits shape [N, 5, H, W], float32, raw unnormalized logits.
3. Spatial dimensions strictly preserved (H_out == H_in, W_out == W_in).
4. Rejection of invalid spatial resolutions not divisible by 32.
"""

import unittest
import torch
from models.eurs_net_v2 import EURSNetV2


class TestModelContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = EURSNetV2(num_classes=5)
        cls.model.eval()

    def test_model_contract_resolutions(self):
        """Tests forward pass across 512x512, 768x768, 1024x1024, and non-square 384x768."""
        candidate_resolutions = [
            (512, 512),
            (768, 768),
            (1024, 1024),
            (384, 768),  # Non-square: both divisible by 32 (384/32=12, 768/32=24)
        ]

        batch_size = 1

        for h, w in candidate_resolutions:
            x = torch.rand(batch_size, 3, h, w, dtype=torch.float32)

            with torch.no_grad():
                restored_rgb, seg_logits = self.model(x)

            # 1. Input/Output Shape Preservation
            self.assertEqual(
                restored_rgb.shape, (batch_size, 3, h, w),
                f"Restoration shape mismatch at {h}x{w}: {restored_rgb.shape}"
            )
            self.assertEqual(
                seg_logits.shape, (batch_size, 5, h, w),
                f"Segmentation logits shape mismatch at {h}x{w}: {seg_logits.shape}"
            )

            # 2. Dtype Verification
            self.assertEqual(restored_rgb.dtype, torch.float32)
            self.assertEqual(seg_logits.dtype, torch.float32)

            # 3. Dynamic Range Verification
            self.assertTrue(
                torch.isfinite(restored_rgb).all(),
                f"Non-finite values detected in restored RGB at {h}x{w}"
            )
            self.assertTrue(
                torch.isfinite(seg_logits).all(),
                f"Non-finite values detected in seg logits at {h}x{w}"
            )
            self.assertGreaterEqual(
                restored_rgb.min().item(), 0.0,
                f"Restored RGB min value < 0.0 at {h}x{w}: {restored_rgb.min().item()}"
            )
            self.assertLessEqual(
                restored_rgb.max().item(), 1.0,
                f"Restored RGB max value > 1.0 at {h}x{w}: {restored_rgb.max().item()}"
            )

            # 4. Raw Logits Verification (Must NOT be softmaxed)
            # Logits should contain negative values and not sum to 1.0 along class dimension
            has_negative = (seg_logits < 0.0).any().item()
            class_sums = seg_logits.sum(dim=1)
            is_softmaxed = torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-3)
            self.assertTrue(
                has_negative or not is_softmaxed,
                f"seg_logits appears to be accidentally softmaxed at {h}x{w}!"
            )

    def test_invalid_resolutions_not_divisible_by_32_fail(self):
        """Verify that spatial dimensions not divisible by 32 raise dimension mismatch errors."""
        invalid_sizes = [
            (500, 500),  # 500 is not divisible by 32
            (100, 150),  # Neither is divisible by 32
        ]

        for h, w in invalid_sizes:
            x = torch.rand(1, 3, h, w, dtype=torch.float32)
            # Progressive FPN additions will fail due to floor division in downsampling stages
            with self.assertRaises(RuntimeError):
                with torch.no_grad():
                    self.model(x)


if __name__ == "__main__":
    unittest.main()
