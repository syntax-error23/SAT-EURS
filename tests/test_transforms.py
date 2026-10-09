"""
Unit tests for data/transforms.py.

Verifies:
1. Spatial alignment across degraded, clean_gt, and mask.
2. Horizontal flip correctness.
3. Vertical flip correctness.
4. Parameterized resize dimension correctness (1024x1024, 768x768, 512x512, 384x768).
5. Nearest-neighbor preservation of discrete integer mask labels {0, 1, 2, 3, 4}.
6. Identical spatial transformation applied across all paired inputs.
7. Non-square input and output handling.
8. 3-channel RGB images and 1-channel (2D and 3D) mask support with dtype preservation.
9. Spatial dimension mismatch error handling.
"""

import unittest
import torch
from data.transforms import (
    SynchronizedResize,
    SynchronizedHorizontalFlip,
    SynchronizedVerticalFlip,
    SynchronizedRandomHorizontalFlip,
    SynchronizedRandomVerticalFlip,
    SynchronizedCompose,
)


class TestSynchronizedTransforms(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_resize_dimension_correctness(self):
        """Test parameterized resize across multiple candidate resolutions."""
        in_h, in_w = 300, 400
        deg = torch.randn(3, in_h, in_w)
        gt = torch.randn(3, in_h, in_w)
        mask = torch.randint(0, 5, (in_h, in_w), dtype=torch.int64)

        test_sizes = [
            (1024, 1024),
            (768, 768),
            (512, 512),
            (384, 768),
            512,
            768,
            1024,
        ]

        for size in test_sizes:
            resizer = SynchronizedResize(size=size)
            expected_h = size if isinstance(size, int) else size[0]
            expected_w = size if isinstance(size, int) else size[1]

            out_deg, out_gt, out_mask = resizer(deg, gt, mask)

            self.assertEqual(out_deg.shape, (3, expected_h, expected_w), f"Failed for size={size}")
            self.assertEqual(out_gt.shape, (3, expected_h, expected_w), f"Failed for size={size}")
            self.assertEqual(out_mask.shape, (expected_h, expected_w), f"Failed for size={size}")

    def test_nearest_neighbor_mask_label_preservation(self):
        """Verify nearest-neighbor interpolation never invents intermediate fractional class IDs."""
        in_h, in_w = 200, 300
        deg = torch.rand(3, in_h, in_w, dtype=torch.float32)
        gt = torch.rand(3, in_h, in_w, dtype=torch.float32)
        
        # Create mask with all 5 discrete class IDs: {0, 1, 2, 3, 4}
        mask = torch.zeros(in_h, in_w, dtype=torch.int64)
        mask[0:40, :] = 0
        mask[40:80, :] = 1
        mask[80:120, :] = 2
        mask[120:160, :] = 3
        mask[160:200, :] = 4

        for target_size in [(768, 768), (1024, 1024), (384, 768), (128, 128)]:
            resizer = SynchronizedResize(size=target_size)
            _, _, out_mask = resizer(deg, gt, mask)

            unique_labels = torch.unique(out_mask).tolist()
            self.assertEqual(out_mask.dtype, torch.int64, "Mask dtype must remain torch.int64")
            for label in unique_labels:
                self.assertIn(label, [0, 1, 2, 3, 4], f"Disallowed interpolated label {label} in {unique_labels}")

    def test_horizontal_flip_correctness(self):
        """Verify horizontal flip mirrors pixels horizontally on all 3 tensors."""
        h, w = 10, 12
        deg = torch.arange(h * w, dtype=torch.float32).reshape(1, h, w).repeat(3, 1, 1)
        gt = deg.clone() * 2.0
        mask = torch.arange(h * w, dtype=torch.int64).reshape(h, w) % 5

        flipper = SynchronizedHorizontalFlip()
        out_deg, out_gt, out_mask = flipper(deg, gt, mask)

        # Check left pixel moved to right
        for y in range(h):
            for x in range(w):
                self.assertEqual(out_deg[0, y, x].item(), deg[0, y, w - 1 - x].item())
                self.assertEqual(out_gt[0, y, x].item(), gt[0, y, w - 1 - x].item())
                self.assertEqual(out_mask[y, x].item(), mask[y, w - 1 - x].item())

    def test_vertical_flip_correctness(self):
        """Verify vertical flip mirrors pixels vertically on all 3 tensors."""
        h, w = 10, 12
        deg = torch.arange(h * w, dtype=torch.float32).reshape(1, h, w).repeat(3, 1, 1)
        gt = deg.clone() * 2.0
        mask = torch.arange(h * w, dtype=torch.int64).reshape(h, w) % 5

        flipper = SynchronizedVerticalFlip()
        out_deg, out_gt, out_mask = flipper(deg, gt, mask)

        # Check top pixel moved to bottom
        for y in range(h):
            for x in range(w):
                self.assertEqual(out_deg[0, y, x].item(), deg[0, h - 1 - y, x].item())
                self.assertEqual(out_gt[0, y, x].item(), gt[0, h - 1 - y, x].item())
                self.assertEqual(out_mask[y, x].item(), mask[h - 1 - y, x].item())

    def test_spatial_alignment_preservation(self):
        """Verify high-contrast marker placed at (y, x) remains aligned across (deg, gt, mask)."""
        h, w = 100, 150
        deg = torch.zeros(3, h, w, dtype=torch.float32)
        gt = torch.zeros(3, h, w, dtype=torch.float32)
        mask = torch.zeros(h, w, dtype=torch.int64)

        # Place a single prominent square marker at (y in [20..30], x in [40..60])
        marker_y1, marker_y2 = 20, 30
        marker_x1, marker_x2 = 40, 60
        deg[:, marker_y1:marker_y2, marker_x1:marker_x2] = 1.0
        gt[:, marker_y1:marker_y2, marker_x1:marker_x2] = 1.0
        mask[marker_y1:marker_y2, marker_x1:marker_x2] = 3  # Class 3 marker

        pipeline = SynchronizedCompose([
            SynchronizedHorizontalFlip(),
            SynchronizedVerticalFlip(),
            SynchronizedResize((200, 300)),
        ])

        out_deg, out_gt, out_mask = pipeline(deg, gt, mask)

        # Locate marker in output mask
        mask_marker = (out_mask == 3)
        deg_marker = (out_deg[0] > 0.5)
        gt_marker = (out_gt[0] > 0.5)

        # Check that where mask has the marker, images also have the marker
        intersection_deg = torch.logical_and(mask_marker, deg_marker)
        intersection_gt = torch.logical_and(mask_marker, gt_marker)

        self.assertTrue(torch.equal(intersection_deg, mask_marker), "Degraded marker misaligned with mask!")
        self.assertTrue(torch.equal(intersection_gt, mask_marker), "GT marker misaligned with mask!")

    def test_random_flip_identical_application(self):
        """Verify SynchronizedRandomHorizontalFlip applies identically to all 3 or none."""
        h, w = 20, 30
        deg = torch.arange(h * w, dtype=torch.float32).reshape(1, h, w).repeat(3, 1, 1)
        gt = deg.clone()
        mask = torch.arange(h * w, dtype=torch.int64).reshape(h, w)

        random_flipper = SynchronizedRandomHorizontalFlip(p=0.5)

        for _ in range(50):
            out_deg, out_gt, out_mask = random_flipper(deg, gt, mask)
            deg_flipped = not torch.equal(out_deg, deg)
            gt_flipped = not torch.equal(out_gt, gt)
            mask_flipped = not torch.equal(out_mask, mask)

            # All three must be flipped, or none
            self.assertEqual(deg_flipped, gt_flipped)
            self.assertEqual(deg_flipped, mask_flipped)

    def test_channel_and_dimension_support(self):
        """Test with 3D mask [1, H, W] and 2D mask [H, W], ensuring channel semantics are preserved."""
        h, w = 60, 80
        deg = torch.rand(3, h, w)
        gt = torch.rand(3, h, w)
        
        # Case A: 2D mask [H, W]
        mask_2d = torch.randint(0, 5, (h, w), dtype=torch.int64)
        resizer = SynchronizedResize((120, 160))
        _, _, out_mask_2d = resizer(deg, gt, mask_2d)
        self.assertEqual(out_mask_2d.shape, (120, 160))
        self.assertEqual(out_mask_2d.dtype, torch.int64)

        # Case B: 3D mask [1, H, W]
        mask_3d = torch.randint(0, 5, (1, h, w), dtype=torch.int64)
        _, _, out_mask_3d = resizer(deg, gt, mask_3d)
        self.assertEqual(out_mask_3d.shape, (1, 120, 160))
        self.assertEqual(out_mask_3d.dtype, torch.int64)

    def test_non_square_aspect_ratios(self):
        """Test with extreme non-square inputs (e.g. 100x300, 300x100) resized to (768, 768) and (384, 768)."""
        aspect_inputs = [(100, 300), (300, 100)]
        target_sizes = [(768, 768), (384, 768), (1024, 512)]

        for (in_h, in_w) in aspect_inputs:
            deg = torch.rand(3, in_h, in_w)
            gt = torch.rand(3, in_h, in_w)
            mask = torch.randint(0, 5, (in_h, in_w), dtype=torch.int64)

            for target in target_sizes:
                resizer = SynchronizedResize(target)
                out_deg, out_gt, out_mask = resizer(deg, gt, mask)
                self.assertEqual(out_deg.shape, (3, target[0], target[1]))
                self.assertEqual(out_gt.shape, (3, target[0], target[1]))
                self.assertEqual(out_mask.shape, (target[0], target[1]))

    def test_dimension_mismatch_error_handling(self):
        """Verify error is raised if input spatial dimensions do not match before transform."""
        deg = torch.rand(3, 100, 100)
        gt = torch.rand(3, 100, 100)
        mismatched_mask = torch.randint(0, 5, (100, 105), dtype=torch.int64)

        resizer = SynchronizedResize((64, 64))
        with self.assertRaises(ValueError):
            resizer(deg, gt, mismatched_mask)

    def test_synchronized_random_crop(self):
        """Verify SynchronizedRandomCrop applies identical crop box to all 3 tensors synchronously."""
        from data.transforms import SynchronizedRandomCrop
        h, w = 120, 150
        deg = torch.randn(3, h, w)
        gt = torch.randn(3, h, w)
        mask = torch.randint(0, 5, (h, w), dtype=torch.int64)

        crop_sizes = [(64, 64), (50, 80), 64]
        for csize in crop_sizes:
            cropper = SynchronizedRandomCrop(size=csize)
            expected_h = csize if isinstance(csize, int) else csize[0]
            expected_w = csize if isinstance(csize, int) else csize[1]

            out_deg, out_gt, out_mask = cropper(deg, gt, mask)

            self.assertEqual(out_deg.shape, (3, expected_h, expected_w))
            self.assertEqual(out_gt.shape, (3, expected_h, expected_w))
            self.assertEqual(out_mask.shape, (expected_h, expected_w))
            self.assertEqual(out_mask.dtype, torch.int64)

        # Test padding if needed
        small_deg = torch.rand(3, 30, 40)
        small_gt = torch.rand(3, 30, 40)
        small_mask = torch.randint(0, 5, (30, 40), dtype=torch.int64)

        cropper_pad = SynchronizedRandomCrop(size=(50, 60), pad_if_needed=True)
        pad_deg, pad_gt, pad_mask = cropper_pad(small_deg, small_gt, small_mask)
        self.assertEqual(pad_deg.shape, (3, 50, 60))
        self.assertEqual(pad_mask.shape, (50, 60))


if __name__ == '__main__':
    unittest.main()
