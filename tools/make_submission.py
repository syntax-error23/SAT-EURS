"""
tools/make_submission.py
------------------------
Challenge Submission Packaging and Validation Utility for EURS Challenge
(Embedded Underwater Restoration and Segmentation @ MaCVi / WACV 2027).

Role & Ownership: Tooling & Logging (Arin)

Functionality:
- Ingests raw model outputs from either a unified predictions directory or separate
  restored/mask folders.
- Strict compliance validation against official challenge rules:
    * 1:1 paired correspondence between restored images and segmentation masks.
    * Masks must be single-channel uint8 PNG with pixel values strictly in range [0..4].
    * Restored images must be valid 8-bit RGB images.
    * Matching dimensions (H, W) between paired restored image and segmentation mask.
    * Aspect-ratio preserving resizing: Longest side capped at 1024px.
      (Images resized with Bicubic interpolation; Masks strictly with Nearest Neighbor).
- Stages resized files into canonical root layout:
    submission.zip
    ├── restored/
    │   ├── sample_0001.png
    │   └── ...
    └── seg_masks/
        ├── sample_0001.png
        └── ...
- Compresses into a single compliant submission.zip archive with integrity verification.
- Rejects incomplete sets, corrupted files, or invalid class values before archiving.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_submission")

VALID_CLASS_IDS = {0, 1, 2, 3, 4}
DEFAULT_MAX_DIM = 1024


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package model predictions into an official EURS challenge submission.zip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Root predictions directory containing 'restored/' and 'seg_masks/' subfolders.",
    )
    parser.add_argument(
        "--restored-dir",
        type=Path,
        default=None,
        help="Path to folder containing predicted restored RGB images (if not using --pred-dir).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Path to folder containing predicted segmentation masks (if not using --pred-dir).",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Optional reference ground-truth directory to verify all test samples are present.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("submission.zip"),
        help="Path to write the final submission zip archive.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Maximum allowable length for the longest image dimension (aspect-ratio preserved).",
    )
    parser.add_argument(
        "--force-max-dim",
        action="store_true",
        help="Upscale smaller images so the longest dimension is exactly --max-dim (default: only cap down).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Optional staging directory to retain resized files (defaults to temporary directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute all validation and resizing simulations without creating the final .zip archive.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file validation info messages.",
    )
    return parser.parse_args(argv)


def resolve_input_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Resolves and validates the paths to restored and mask prediction directories."""
    restored_dir: Optional[Path] = args.restored_dir
    masks_dir: Optional[Path] = args.masks_dir

    if args.pred_dir is not None:
        p = args.pred_dir
        if not p.exists():
            raise FileNotFoundError(f"--pred-dir does not exist: {p.resolve()}")

        # Search for restored folder
        for cand in ["restored", "images", "restored_images"]:
            if (p / cand).is_dir():
                restored_dir = p / cand
                break

        # Search for masks folder
        for cand in ["seg_masks", "masks", "segmentation", "pred_masks"]:
            if (p / cand).is_dir():
                masks_dir = p / cand
                break

    if restored_dir is None or not restored_dir.exists():
        raise FileNotFoundError(
            f"Restored directory could not be resolved. Please specify --restored-dir or ensure "
            f"'restored/' exists inside --pred-dir."
        )

    if masks_dir is None or not masks_dir.exists():
        raise FileNotFoundError(
            f"Segmentation masks directory could not be resolved. Please specify --masks-dir or ensure "
            f"'seg_masks/' exists inside --pred-dir."
        )

    return restored_dir, masks_dir


def calculate_target_dims(
    orig_w: int,
    orig_h: int,
    max_dim: int,
    force_max_dim: bool = False,
) -> Tuple[int, int]:
    """
    Computes aspect-ratio preserved dimensions capping the longest side at max_dim.
    """
    longest = max(orig_w, orig_h)
    if longest <= max_dim and not force_max_dim:
        return orig_w, orig_h

    scale = max_dim / float(longest)
    if orig_w >= orig_h:
        new_w = max_dim
        new_h = max(1, int(round(orig_h * scale)))
    else:
        new_h = max_dim
        new_w = max(1, int(round(orig_w * scale)))

    return new_w, new_h


def validate_mask(mask_path: Path) -> Tuple[bool, str, np.ndarray]:
    """
    Validates that a mask is a valid single-channel PNG with values in {0, 1, 2, 3, 4}.
    """
    try:
        with Image.open(mask_path) as img:
            arr = np.array(img)
            if arr.ndim > 2:
                # Check if multi-channel array has redundant identical channels
                if arr.shape[2] == 3 or arr.shape[2] == 4:
                    if np.array_equal(arr[:, :, 0], arr[:, :, 1]):
                        arr = arr[:, :, 0]
                    else:
                        return False, f"Mask has {arr.shape[2]} non-identical channels (expected single-channel).", arr
                else:
                    return False, f"Unexpected mask shape: {arr.shape}", arr

            # Check pixel values
            unique_vals = set(np.unique(arr).tolist())
            invalid_vals = unique_vals - VALID_CLASS_IDS
            if invalid_vals:
                return (
                    False,
                    f"Invalid class values detected: {invalid_vals}. Allowed classes are strictly {VALID_CLASS_IDS}.",
                    arr,
                )

            return True, "Valid", arr
    except Exception as e:
        return False, f"Exception opening mask: {e}", np.array([])


def validate_and_stage_files(
    restored_dir: Path,
    masks_dir: Path,
    staging_dir: Path,
    ref_dir: Optional[Path] = None,
    max_dim: int = DEFAULT_MAX_DIM,
    force_max_dim: bool = False,
    quiet: bool = False,
) -> Dict:
    """
    Validates paired restored images and masks, resizes them with aspect ratio preservation,
    and saves them into staging_dir/restored and staging_dir/seg_masks.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    restored_files = {
        p.stem: p for p in restored_dir.glob("*") if p.suffix.lower() in valid_extensions and p.is_file()
    }
    mask_files = {
        p.stem: p for p in masks_dir.glob("*") if p.suffix.lower() in valid_extensions and p.is_file()
    }

    if not restored_files:
        raise ValueError(f"No valid image files found in restored directory: {restored_dir.resolve()}")
    if not mask_files:
        raise ValueError(f"No valid mask files found in masks directory: {masks_dir.resolve()}")

    restored_stems = set(restored_files.keys())
    mask_stems = set(mask_files.keys())

    # Check 1: Completeness between restored and masks
    missing_in_masks = restored_stems - mask_stems
    missing_in_restored = mask_stems - restored_stems

    validation_errors: List[str] = []
    if missing_in_masks:
        sample_missing = sorted(list(missing_in_masks))[:5]
        validation_errors.append(
            f"Restored folder has {len(missing_in_masks)} samples missing corresponding masks! Examples: {sample_missing}"
        )
    if missing_in_restored:
        sample_missing = sorted(list(missing_in_restored))[:5]
        validation_errors.append(
            f"Masks folder has {len(missing_in_restored)} samples missing corresponding restored images! Examples: {sample_missing}"
        )

    # Check reference set if specified
    if ref_dir is not None:
        ref_stems = {
            p.stem for p in ref_dir.glob("*") if p.suffix.lower() in valid_extensions and p.is_file()
        }
        missing_ref = ref_stems - restored_stems
        if missing_ref:
            validation_errors.append(
                f"Missing {len(missing_ref)} samples required by reference directory {ref_dir}! Examples: {sorted(list(missing_ref))[:5]}"
            )

    if validation_errors:
        for err in validation_errors:
            logger.error(f"[VALIDATION FAILED] {err}")
        raise RuntimeError("Completeness validation failed. Aborting submission packaging.")

    common_stems = sorted(list(restored_stems & mask_stems))
    total_samples = len(common_stems)
    logger.info(f"Verified 1:1 match for {total_samples} sample pairs across restored and mask folders.")

    # Target staging subfolders
    stage_restored = staging_dir / "restored"
    stage_masks = staging_dir / "seg_masks"
    stage_restored.mkdir(parents=True, exist_ok=True)
    stage_masks.mkdir(parents=True, exist_ok=True)

    resized_count = 0
    identical_dim_count = 0

    for idx, stem in enumerate(common_stems, start=1):
        r_path = restored_files[stem]
        m_path = mask_files[stem]

        # 1. Validate Mask
        is_mask_valid, mask_msg, mask_arr = validate_mask(m_path)
        if not is_mask_valid:
            raise ValueError(f"Mask validation failed for {m_path.name}: {mask_msg}")

        # 2. Open Restored Image
        try:
            with Image.open(r_path) as r_img:
                r_rgb = r_img.convert("RGB")
                r_w, r_h = r_rgb.size
        except Exception as e:
            raise ValueError(f"Failed to open restored image {r_path.name}: {e}")

        # 3. Check shape parity between image and mask
        m_h, m_w = mask_arr.shape[:2]
        if (r_w, r_h) != (m_w, m_h):
            raise ValueError(
                f"Dimension mismatch for {stem}: Restored image is {r_w}x{r_h}, but mask is {m_w}x{m_h}!"
            )

        # 4. Aspect-ratio preserving target dimensions
        target_w, target_h = calculate_target_dims(
            orig_w=r_w,
            orig_h=r_h,
            max_dim=max_dim,
            force_max_dim=force_max_dim,
        )

        needs_resize = (target_w, target_h) != (r_w, r_h)
        if needs_resize:
            resized_count += 1
            # Resize image with BICUBIC
            r_resized = r_rgb.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
            # Resize mask with NEAREST NEIGHBOR to preserve integer class indices
            m_pil = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
            m_resized = m_pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        else:
            identical_dim_count += 1
            r_resized = r_rgb
            m_resized = Image.fromarray(mask_arr.astype(np.uint8), mode="L")

        # Double check resized mask pixel values
        resized_mask_vals = set(np.unique(np.array(m_resized)).tolist())
        if resized_mask_vals - VALID_CLASS_IDS:
            raise RuntimeError(
                f"Interpolation introduced illegal class values in {stem}: {resized_mask_vals}"
            )

        # Save to staging directory in canonical PNG format
        target_img_path = stage_restored / f"{stem}.png"
        target_mask_path = stage_masks / f"{stem}.png"

        r_resized.save(target_img_path, format="PNG", optimize=True)
        m_resized.save(target_mask_path, format="PNG", optimize=True)

        if not quiet and idx % 200 == 0:
            logger.info(f"Processed and validated {idx}/{total_samples} samples...")

    return {
        "total_samples": total_samples,
        "resized_samples": resized_count,
        "unaltered_samples": identical_dim_count,
        "max_dim_cap": max_dim,
    }


def create_submission_zip(staging_dir: Path, output_zip_path: Path) -> Path:
    """
    Compresses staging_dir/restored and staging_dir/seg_masks into a clean zip archive.
    """
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    stage_restored = staging_dir / "restored"
    stage_masks = staging_dir / "seg_masks"

    temp_zip = output_zip_path.with_suffix(".tmp.zip")
    if temp_zip.exists():
        temp_zip.unlink()

    logger.info(f"Archiving files into {output_zip_path}...")
    with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add restored
        for img_path in sorted(stage_restored.glob("*.png")):
            arcname = f"restored/{img_path.name}"
            zf.write(img_path, arcname=arcname)

        # Add seg_masks
        for mask_path in sorted(stage_masks.glob("*.png")):
            arcname = f"seg_masks/{mask_path.name}"
            zf.write(mask_path, arcname=arcname)

    # Move atomically to final destination
    if output_zip_path.exists():
        output_zip_path.unlink()
    temp_zip.rename(output_zip_path)

    # Verify zip integrity
    with zipfile.ZipFile(output_zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"Corrupt entry detected in generated zip: {bad_file}")
        namelist = zf.namelist()

    restored_in_zip = [n for n in namelist if n.startswith("restored/") and not n.endswith("/")]
    masks_in_zip = [n for n in namelist if n.startswith("seg_masks/") and not n.endswith("/")]

    logger.info(f"Zip Verification PASSED: {len(restored_in_zip)} restored images & {len(masks_in_zip)} masks.")
    return output_zip_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        restored_dir, masks_dir = resolve_input_dirs(args)
    except Exception as e:
        logger.error(f"Input path resolution error: {e}")
        return 1

    logger.info("=" * 70)
    logger.info("   EURS CHALLENGE: SUBMISSION PACKAGING & VALIDATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Restored Images Source : {restored_dir.resolve()}")
    logger.info(f"Segmentation Source    : {masks_dir.resolve()}")
    logger.info(f"Longest Side Cap       : {args.max_dim}px (Aspect-Ratio Preserved)")
    logger.info(f"Output Archive Target  : {args.output_zip.resolve()}")

    is_custom_staging = args.staging_dir is not None
    staging_path = args.staging_dir if is_custom_staging else Path(tempfile.mkdtemp(prefix="eurs_submit_"))

    try:
        stats = validate_and_stage_files(
            restored_dir=restored_dir,
            masks_dir=masks_dir,
            staging_dir=staging_path,
            ref_dir=args.ref_dir,
            max_dim=args.max_dim,
            force_max_dim=args.force_max_dim,
            quiet=args.quiet,
        )

        logger.info("-" * 70)
        logger.info(f"Validation & Staging Summary:")
        logger.info(f"  * Total Validated Pairs : {stats['total_samples']}")
        logger.info(f"  * Resized (Exceeded Cap): {stats['resized_samples']}")
        logger.info(f"  * Unaltered Dimension   : {stats['unaltered_samples']}")
        logger.info("-" * 70)

        if args.dry_run:
            logger.info("[DRY-RUN ENABLED] Skipping final zip generation. All validation checks passed!")
            return 0

        final_zip = create_submission_zip(staging_path, args.output_zip)
        zip_size_mb = final_zip.stat().st_size / (1024.0 * 1024.0)

        print("\n" + "=" * 70)
        print("          SUBMISSION ARCHIVE GENERATION SUCCESSFUL")
        print("=" * 70)
        print(f"Archive Location : {final_zip.resolve()}")
        print(f"Archive Size     : {zip_size_mb:.2f} MB")
        print(f"Restored Images  : {stats['total_samples']} files inside 'restored/'")
        print(f"Seg Masks        : {stats['total_samples']} files inside 'seg_masks/'")
        print(f"Mask Pixel Rules : Strictly 0..4 (Single-Channel PNG)")
        print(f"Status           : READY FOR TRACK 1 LEADERBOARD UPLOAD")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Packaging aborted due to error: {e}")
        return 1
    finally:
        # Cleanup temporary staging folder if we allocated it
        if not is_custom_staging and staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
