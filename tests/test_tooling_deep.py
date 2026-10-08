"""
tests/test_tooling_deep.py
--------------------------
Comprehensive Deep Test Suite for Tooling & Logging components:
- tools/eda_stats.py
- tools/visualizer.py
- tools/make_submission.py
- experiments/leaderboard.md

Covers positive workflows, negative paths, boundary conditions, and edge cases.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

# Import our tools
from tools.eda_stats import (
    CLASS_NAMES,
    NUM_CLASSES,
    analyze_dataset_masks,
    parse_args as eda_parse_args,
    print_eda_report,
)
from tools.visualizer import (
    CLASS_METADATA,
    colorize_mask,
    discover_common_stems,
    find_paired_file,
    load_image_normalized,
    load_mask,
    render_tri_panel,
    main as visualizer_main,
)
from tools.make_submission import (
    VALID_CLASS_IDS,
    calculate_target_dims,
    create_submission_zip,
    validate_and_stage_files,
    validate_mask,
    main as submission_main,
)


class TestEdaStats:
    """Test suite for tools/eda_stats.py"""

    def test_eda_normal_flow(self, tmp_path: Path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        # Create mask with all 5 classes
        m = np.zeros((100, 100), dtype=np.uint8)
        m[0:10, 0:10] = 1   # Coral: 100 px
        m[10:20, 0:10] = 2  # Fish: 100 px
        m[20:30, 0:10] = 3  # Shark: 100 px
        m[30:40, 0:10] = 4  # Other: 100 px
        # Remaining 9600 px is Background (0)
        Image.fromarray(m, mode="L").save(masks_dir / "mask_1.png")

        stats, records = analyze_dataset_masks(mask_dir=masks_dir, c0=1.02)

        assert stats["dataset_summary"]["total_masks_analyzed"] == 1
        assert stats["dataset_summary"]["total_pixels_counted"] == 10000
        assert len(records) == 1
        assert len(stats["class_statistics"]) == 5

        # Check weights: rarer classes must have strictly higher weights than background
        weights = stats["pytorch_weight_tensor"]
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]
        assert weights[3] > weights[0]

    def test_eda_missing_class(self, tmp_path: Path):
        """Edge case: Dataset contains zero pixels for some classes."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        # Mask containing ONLY background (0) and coral (1)
        m = np.zeros((50, 50), dtype=np.uint8)
        m[0:10, 0:10] = 1
        Image.fromarray(m, mode="L").save(masks_dir / "mask_only_coral.png")

        stats, records = analyze_dataset_masks(mask_dir=masks_dir, c0=1.02)
        assert stats["dataset_summary"]["total_masks_analyzed"] == 1
        class_stats = {c["class_id"]: c for c in stats["class_statistics"]}
        assert class_stats[2]["pixel_count"] == 0
        assert class_stats[3]["pixel_count"] == 0
        assert class_stats[4]["pixel_count"] == 0
        # Weights should still be finite and valid
        assert all(w > 0 for w in stats["pytorch_weight_tensor"])

    def test_eda_out_of_range_pixels(self, tmp_path: Path):
        """Edge case: Mask contains illegal class values (e.g., 5, 255)."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        m = np.zeros((50, 50), dtype=np.uint8)
        m[0:5, 0:5] = 5    # Invalid class 5
        m[5:10, 0:5] = 255  # Invalid class 255
        Image.fromarray(m, mode="L").save(masks_dir / "corrupted_mask.png")

        stats, _ = analyze_dataset_masks(mask_dir=masks_dir, c0=1.02)
        invalid = stats["dataset_summary"]["invalid_pixel_values_encountered"]
        assert 5 in invalid
        assert 255 in invalid
        assert invalid[5] == 25
        assert invalid[255] == 25

    def test_eda_multichannel_mask(self, tmp_path: Path):
        """Edge case: Mask image has 3 or 4 channels instead of 1."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        # 3-channel RGB image
        rgb_mask = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb_mask[0:10, 0:10, :] = 1
        Image.fromarray(rgb_mask, mode="RGB").save(masks_dir / "rgb_mask.png")

        stats, _ = analyze_dataset_masks(mask_dir=masks_dir, c0=1.02)
        assert stats["dataset_summary"]["total_masks_analyzed"] == 1
        assert stats["class_statistics"][1]["pixel_count"] == 100

    def test_eda_dimension_mismatch_detection(self, tmp_path: Path):
        """Edge case: Paired image has different dimension from mask."""
        masks_dir = tmp_path / "masks"
        images_dir = tmp_path / "images"
        masks_dir.mkdir()
        images_dir.mkdir()

        # Mask is 100x100
        Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L").save(masks_dir / "sample_1.png")
        # Image is 200x200
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8), mode="RGB").save(images_dir / "sample_1.png")

        stats, _ = analyze_dataset_masks(mask_dir=masks_dir, images_dir=images_dir)
        assert stats["dataset_summary"]["dimension_mismatches_with_images"] == 1

    def test_eda_empty_directory_raises(self, tmp_path: Path):
        """Edge case: Mask directory is empty."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No mask image files found"):
            analyze_dataset_masks(mask_dir=empty_dir)


class TestVisualizer:
    """Test suite for tools/visualizer.py"""

    def test_colorize_mask_all_classes(self):
        """Verify color mapping across all 5 classes and out-of-range pixels."""
        mask = np.array([[0, 1], [2, 3], [4, 99]], dtype=np.int64)
        colored = colorize_mask(mask)
        assert colored.shape == (3, 2, 3)
        # Background
        np.testing.assert_allclose(colored[0, 0], CLASS_METADATA[0]["color"])
        # Coral
        np.testing.assert_allclose(colored[0, 1], CLASS_METADATA[1]["color"])
        # Fish
        np.testing.assert_allclose(colored[1, 0], CLASS_METADATA[2]["color"])
        # Shark
        np.testing.assert_allclose(colored[1, 1], CLASS_METADATA[3]["color"])
        # Other
        np.testing.assert_allclose(colored[2, 0], CLASS_METADATA[4]["color"])
        # Out-of-bounds index 99 mapped to red [1.0, 0.0, 0.0]
        np.testing.assert_allclose(colored[2, 1], [1.0, 0.0, 0.0])

    def test_render_tri_panel_file_generation(self, tmp_path: Path):
        """Verify 3-panel figure generation with labels and legend."""
        deg = np.ones((64, 64, 3), dtype=np.float32) * 0.4
        clean = np.ones((64, 64, 3), dtype=np.float32) * 0.8
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[10:30, 10:30] = 1
        out_file = tmp_path / "plot_test.png"

        saved = render_tri_panel(
            degraded_img=deg,
            clean_img=clean,
            mask_arr=mask,
            sample_id="test_001",
            output_path=out_file,
            overlay=False,
        )
        assert saved.exists()
        assert saved.stat().st_size > 1000

    def test_render_5_panel_prediction_mode(self, tmp_path: Path):
        """Verify extended 5-panel mode when predictions are supplied."""
        deg = np.ones((64, 64, 3), dtype=np.float32) * 0.4
        clean = np.ones((64, 64, 3), dtype=np.float32) * 0.8
        mask = np.zeros((64, 64), dtype=np.int64)
        pred_res = np.ones((64, 64, 3), dtype=np.float32) * 0.75
        pred_mask = np.ones((64, 64), dtype=np.int64) * 2

        out_file = tmp_path / "plot_5panel.png"
        saved = render_tri_panel(
            degraded_img=deg,
            clean_img=clean,
            mask_arr=mask,
            sample_id="test_5p",
            output_path=out_file,
            pred_restored=pred_res,
            pred_mask=pred_mask,
        )
        assert saved.exists()
        assert saved.stat().st_size > 1000

    def test_visualizer_cli_end_to_end(self, tmp_path: Path):
        """Test full visualizer CLI workflow with multiple samples."""
        d_dir = tmp_path / "degraded"
        c_dir = tmp_path / "clean"
        m_dir = tmp_path / "masks"
        p_dir = tmp_path / "plots"
        for p in [d_dir, c_dir, m_dir, p_dir]:
            p.mkdir()

        # Create two matching samples
        for s in ["img_01", "img_02"]:
            Image.new("RGB", (60, 60), (50, 100, 150)).save(d_dir / f"{s}.png")
            Image.new("RGB", (60, 60), (80, 180, 220)).save(c_dir / f"{s}.png")
            m = np.zeros((60, 60), dtype=np.uint8)
            m[10:20, 10:20] = 2
            Image.fromarray(m, mode="L").save(m_dir / f"{s}.png")

        ret = visualizer_main([
            "--degraded-dir", str(d_dir),
            "--clean-dir", str(c_dir),
            "--mask-dir", str(m_dir),
            "--output-dir", str(p_dir),
            "--num-samples", "2",
        ])
        assert ret == 0
        assert (p_dir / "compare_img_01.png").exists()
        assert (p_dir / "compare_img_02.png").exists()


class TestMakeSubmission:
    """Test suite for tools/make_submission.py"""

    def test_calculate_target_dims(self):
        # Already under 1024: no change
        assert calculate_target_dims(800, 600, 1024) == (800, 600)
        # 1920x1080 -> 1024x576
        assert calculate_target_dims(1920, 1080, 1024) == (1024, 576)
        # Portrait: 1080x1920 -> 576x1024
        assert calculate_target_dims(1080, 1920, 1024) == (576, 1024)
        # Square: 2048x2048 -> 1024x1024
        assert calculate_target_dims(2048, 2048, 1024) == (1024, 1024)

    def test_validate_mask_valid_cases(self, tmp_path: Path):
        m = np.array([[0, 1], [2, 3], [4, 0]], dtype=np.uint8)
        p = tmp_path / "valid_mask.png"
        Image.fromarray(m, mode="L").save(p)

        valid, msg, arr = validate_mask(p)
        assert valid is True
        assert arr.shape == (3, 2)

    def test_validate_mask_invalid_values(self, tmp_path: Path):
        m = np.array([[0, 1], [2, 5]], dtype=np.uint8)  # 5 is illegal
        p = tmp_path / "invalid_mask.png"
        Image.fromarray(m, mode="L").save(p)

        valid, msg, _ = validate_mask(p)
        assert valid is False
        assert "Invalid class values" in msg

    def test_submission_full_pipeline_with_resizing(self, tmp_path: Path):
        restored = tmp_path / "restored"
        masks = tmp_path / "seg_masks"
        out_zip = tmp_path / "test_sub.zip"
        restored.mkdir()
        masks.mkdir()

        # Sample 1: Small image (500x300)
        Image.new("RGB", (500, 300), (100, 150, 200)).save(restored / "s1.png")
        m1 = np.zeros((300, 500), dtype=np.uint8)
        m1[10:50, 10:50] = 1
        Image.fromarray(m1, mode="L").save(masks / "s1.png")

        # Sample 2: Oversized image (1920x1080)
        Image.new("RGB", (1920, 1080), (80, 120, 180)).save(restored / "s2.png")
        m2 = np.zeros((1080, 1920), dtype=np.uint8)
        m2[100:300, 100:300] = 3
        Image.fromarray(m2, mode="L").save(masks / "s2.png")

        ret = submission_main([
            "--restored-dir", str(restored),
            "--masks-dir", str(masks),
            "--output-zip", str(out_zip),
            "--max-dim", "1024",
        ])
        assert ret == 0
        assert out_zip.exists()

        # Inspect Zip structure and resizing
        with zipfile.ZipFile(out_zip, "r") as zf:
            assert zf.testzip() is None
            names = set(zf.namelist())
            assert "restored/s1.png" in names
            assert "restored/s2.png" in names
            assert "seg_masks/s1.png" in names
            assert "seg_masks/s2.png" in names

            # Check s2 was resized to max_dim 1024
            with zf.open("restored/s2.png") as f:
                img = Image.open(f)
                assert img.size == (1024, 576)

            with zf.open("seg_masks/s2.png") as f:
                mask = Image.open(f)
                assert mask.size == (1024, 576)
                # Verify nearest-neighbor preserved integer class values strictly
                u = set(np.unique(np.array(mask)))
                assert u.issubset({0, 1, 2, 3, 4})

    def test_submission_rejects_unpaired_samples(self, tmp_path: Path):
        restored = tmp_path / "restored"
        masks = tmp_path / "seg_masks"
        restored.mkdir()
        masks.mkdir()

        Image.new("RGB", (100, 100)).save(restored / "s1.png")
        Image.new("RGB", (100, 100)).save(restored / "s2.png")
        # Only s1 mask provided
        Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L").save(masks / "s1.png")

        ret = submission_main([
            "--restored-dir", str(restored),
            "--masks-dir", str(masks),
            "--output-zip", str(tmp_path / "fail.zip"),
        ])
        assert ret != 0
        assert not (tmp_path / "fail.zip").exists()

    def test_submission_rejects_dimension_mismatch(self, tmp_path: Path):
        restored = tmp_path / "restored"
        masks = tmp_path / "seg_masks"
        restored.mkdir()
        masks.mkdir()

        Image.new("RGB", (200, 100)).save(restored / "s1.png")
        Image.fromarray(np.zeros((150, 100), dtype=np.uint8), mode="L").save(masks / "s1.png")

        ret = submission_main([
            "--restored-dir", str(restored),
            "--masks-dir", str(masks),
            "--output-zip", str(tmp_path / "fail.zip"),
        ])
        assert ret != 0
        assert not (tmp_path / "fail.zip").exists()

    def test_calculate_target_dims_invalid_raises(self):
        """Edge case: Non-positive dimensions must raise ValueError."""
        with pytest.raises(ValueError, match="max_dim must be positive"):
            calculate_target_dims(100, 100, max_dim=0)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            calculate_target_dims(-10, 100, max_dim=1024)

    def test_submission_stale_staging_purged(self, tmp_path: Path):
        """Edge case: Staging dir contains old leftover png files."""
        restored = tmp_path / "restored"
        masks = tmp_path / "seg_masks"
        staging = tmp_path / "custom_staging"
        out_zip = tmp_path / "clean_sub.zip"
        restored.mkdir()
        masks.mkdir()
        staging.mkdir()

        # Pre-populate staging with a stale file
        old_restored = staging / "restored"
        old_restored.mkdir()
        Image.new("RGB", (50, 50)).save(old_restored / "stale_old_file.png")

        # Current valid sample
        Image.new("RGB", (100, 100)).save(restored / "sample_real.png")
        Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L").save(masks / "sample_real.png")

        ret = submission_main([
            "--restored-dir", str(restored),
            "--masks-dir", str(masks),
            "--staging-dir", str(staging),
            "--output-zip", str(out_zip),
        ])
        assert ret == 0

        # Verify stale file was purged and not included in zip
        with zipfile.ZipFile(out_zip, "r") as zf:
            namelist = zf.namelist()
            assert "restored/stale_old_file.png" not in namelist
            assert "restored/sample_real.png" in namelist


class TestEdgeCasesAndInit:
    """Test suite for tools package exports and edge cases."""

    def test_eda_c0_invalid_raises(self, tmp_path: Path):
        """Edge case: c0 <= 1.0 must raise ValueError."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L").save(masks_dir / "m.png")

        with pytest.raises(ValueError, match="strictly > 1.0"):
            analyze_dataset_masks(mask_dir=masks_dir, c0=1.0)

    def test_visualizer_overlay_mismatched_dimensions(self, tmp_path: Path):
        """Edge case: Clean image and mask have different dimensions in overlay mode."""
        clean = np.ones((100, 100, 3), dtype=np.float32)
        deg = np.ones((100, 100, 3), dtype=np.float32)
        mask = np.zeros((50, 50), dtype=np.int64)  # Mismatched 50x50 vs 100x100!
        out_file = tmp_path / "overlay_mismatch.png"

        # Must not raise broadcasting error!
        saved = render_tri_panel(
            degraded_img=deg,
            clean_img=clean,
            mask_arr=mask,
            sample_id="mismatch_dim",
            output_path=out_file,
            overlay=True,
        )
        assert saved.exists()

    def test_visualizer_num_samples_zero_fails(self, tmp_path: Path):
        """Edge case: num_samples <= 0 must return error code 1."""
        d_dir = tmp_path / "d"
        c_dir = tmp_path / "c"
        m_dir = tmp_path / "m"
        for p in [d_dir, c_dir, m_dir]:
            p.mkdir()
            Image.new("RGB", (10, 10)).save(p / "s.png")

        ret = visualizer_main([
            "--degraded-dir", str(d_dir),
            "--clean-dir", str(c_dir),
            "--mask-dir", str(m_dir),
            "--num-samples", "0",
        ])
        assert ret == 1

    def test_tools_init_exports(self):
        """Verify that tools package exposes all public functions."""
        import tools
        assert hasattr(tools, "analyze_dataset_masks")
        assert hasattr(tools, "colorize_mask")
        assert hasattr(tools, "render_tri_panel")
        assert hasattr(tools, "calculate_target_dims")
        assert hasattr(tools, "validate_mask")
        assert hasattr(tools, "create_submission_zip")


class TestLeaderboardDoc:
    """Test suite for experiments/leaderboard.md"""

    def test_leaderboard_file_structure(self):
        doc_path = Path(r"c:\Users\ARIN\Documents\Projects\SAT-EURS\experiments\leaderboard.md")
        assert doc_path.exists()
        content = doc_path.read_text(encoding="utf-8")
        assert "EURS Challenge" in content
        assert "Experiment ID" in content
        assert "Local mIoU" in content
        assert "Local PSNR" in content
        assert "Track 1 Composite Score" in content
