"""
tools/eda_stats.py
------------------
Exploratory Data Analysis (EDA) and Dataset Statistics Script for EURS Challenge
(Embedded Underwater Restoration and Segmentation @ MaCVi / WACV 2027).

Role & Ownership: Tooling & Logging (Arin)

Functionality:
- Recursively scans the dataset mask directory for ground-truth PNG segmentation masks.
- Validates pixel integrity across the 5 target classes:
    0: Background
    1: Coral
    2: Fish
    3: Elasmobranch (Sharks & Rays)
    4: Other
- Computes per-class total pixel counts, global frequency distributions, and percentage share.
- Calculates inverse class frequency weights using the ENet formula:
    w_c = 1.0 / ln(c_0 + frequency_c)
  where c_0 defaults to 1.02.
- Weight Clamping Guard: Clamps normalized weights to --max-weight (default: 10.0) to prevent
  gradient explosions during mixed-precision (bfloat16/fp16) training on rare classes.
- PyTorch Output Format: Prints resulting weights directly formatted as a copy-ready PyTorch tensor:
    torch.tensor([...], dtype=torch.float32)
- Inspects image resolutions, aspect ratios, and identifies any corrupted or out-of-range masks.
- Optionally exports summary statistics to JSON or CSV for downstream training consumption.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda_stats")

# Challenge Class Definition
CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Coral",
    2: "Fish",
    3: "Elasmobranch (Sharks/Rays)",
    4: "Other",
}
NUM_CLASSES: int = 5


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 5-class pixel distributions, aspect ratios, and inverse loss weights for EURS challenge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data/masks"),
        help="Path to directory containing ground-truth PNG segmentation masks.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional path to directory containing corresponding images to verify dimension parity.",
    )
    parser.add_argument(
        "--c0",
        type=float,
        default=1.02,
        help="Smoothing constant for logarithmic inverse class frequency weighting (w = 1 / log(c0 + freq)).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=10.0,
        help="Maximum allowable loss weight ceiling to prevent gradient explosions in mixed-precision (bfloat16) training.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional destination path to export computed statistics and weights as a JSON file.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional destination path to export per-image pixel counts as a CSV file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of mask files analyzed (useful for quick dry runs).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed file-by-file warnings and info logs.",
    )
    return parser.parse_args(argv)


def analyze_dataset_masks(
    mask_dir: Path,
    images_dir: Optional[Path] = None,
    c0: float = 1.02,
    max_weight: float = 10.0,
    max_files: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Analyzes segmentation masks in mask_dir, calculating pixel distributions,
    class imbalance weights, and dimension characteristics.
    """
    if c0 <= 1.0:
        raise ValueError(f"Parameter c0 must be strictly > 1.0 (got {c0}) to prevent division by zero.")

    if max_weight <= 0:
        raise ValueError(f"Parameter max_weight must be strictly positive (got {max_weight}).")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir.resolve()}")

    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {mask_dir.resolve()}")

    # Find all PNG/BMP/TIFF masks (case-insensitive)
    valid_mask_exts = {".png", ".bmp", ".tif", ".tiff"}
    mask_files = sorted(
        [
            p
            for p in mask_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_mask_exts
        ]
    )

    if not mask_files:
        raise ValueError(f"No mask image files found in {mask_dir.resolve()}")

    if max_files is not None and max_files > 0:
        mask_files = mask_files[:max_files]

    total_masks = len(mask_files)
    if not quiet:
        logger.info(f"Discovered {total_masks} mask files to analyze in {mask_dir}")

    # Accumulators
    class_pixel_counts = Counter({cls_id: 0 for cls_id in range(NUM_CLASSES)})
    images_with_class = Counter({cls_id: 0 for cls_id in range(NUM_CLASSES)})
    invalid_pixel_counter = Counter()
    dimensions_counter = Counter()
    aspect_ratios_counter = Counter()
    per_image_records = []
    dimension_mismatches = []
    missing_paired_images = 0

    for idx, mask_path in enumerate(mask_files, start=1):
        try:
            with Image.open(mask_path) as img:
                w, h = img.size
                dimensions_counter[(w, h)] += 1
                aspect_ratio = round(w / h, 4) if h > 0 else 0.0
                aspect_ratios_counter[aspect_ratio] += 1

                # Check paired image dimension if images_dir is supplied
                if images_dir is not None:
                    matching_img_path = images_dir / mask_path.name
                    if not matching_img_path.exists():
                        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                            alt = images_dir / (mask_path.stem + ext)
                            if alt.exists():
                                matching_img_path = alt
                                break
                    if matching_img_path.exists():
                        with Image.open(matching_img_path) as ref_img:
                            if ref_img.size != (w, h):
                                dimension_mismatches.append(
                                    (mask_path.name, (w, h), ref_img.size)
                                )
                    else:
                        missing_paired_images += 1

                # Convert to numpy array
                mask_arr = np.array(img)

                # Ensure 2D mask
                if mask_arr.ndim == 3:
                    if not quiet:
                        logger.warning(
                            f"Mask {mask_path.name} has {mask_arr.shape[2]} channels; taking first channel."
                        )
                    mask_arr = mask_arr[:, :, 0]

        except Exception as e:
            logger.error(f"Failed to read mask {mask_path}: {e}")
            continue

        # Count pixels per class
        unique_vals, counts = np.unique(mask_arr, return_counts=True)
        img_counts = {int(cls_id): 0 for cls_id in range(NUM_CLASSES)}

        for val, count in zip(unique_vals, counts):
            val_int = int(val)
            if 0 <= val_int < NUM_CLASSES:
                class_pixel_counts[val_int] += int(count)
                img_counts[val_int] = int(count)
            else:
                invalid_pixel_counter[val_int] += int(count)
                if not quiet:
                    logger.warning(
                        f"Out-of-range class index {val_int} found in {mask_path.name} ({count} pixels)!"
                    )

        for cls_id in range(NUM_CLASSES):
            if img_counts[cls_id] > 0:
                images_with_class[cls_id] += 1

        per_image_records.append(
            {
                "filename": mask_path.name,
                "relative_path": str(mask_path.relative_to(mask_dir)),
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                **{f"class_{c}_pixels": img_counts[c] for c in range(NUM_CLASSES)},
            }
        )

        if not quiet and idx % 200 == 0:
            logger.info(f"Processed {idx}/{total_masks} masks...")

    total_valid_pixels = sum(class_pixel_counts.values())
    if total_valid_pixels == 0:
        raise RuntimeError("No valid pixels found across any analyzed masks!")

    valid_masks_count = len(per_image_records)

    # Calculate frequencies and inverse weights
    frequencies: Dict[int, float] = {}
    percentage_distribution: Dict[int, float] = {}
    raw_enet_weights: Dict[int, float] = {}
    clamped_enet_weights: Dict[int, float] = {}
    was_clamped_flag: Dict[int, bool] = {}
    inv_log_weights: Dict[int, float] = {}
    median_freq_weights: Dict[int, float] = {}

    for c in range(NUM_CLASSES):
        count = class_pixel_counts[c]
        freq = count / total_valid_pixels
        frequencies[c] = freq
        percentage_distribution[c] = freq * 100.0
        # ENet formulation: w_c = 1 / ln(c_0 + freq_c)
        raw_w = 1.0 / math.log(c0 + freq)
        raw_enet_weights[c] = raw_w
        clamped_w = min(raw_w, float(max_weight))
        clamped_enet_weights[c] = clamped_w
        was_clamped_flag[c] = (raw_w > max_weight)
        inv_log_weights[c] = clamped_w

    # Normalized weights so that the sum of weights equals NUM_CLASSES (standard PyTorch practice)
    weight_sum = sum(inv_log_weights.values())
    raw_normalized_weights = {
        c: (inv_log_weights[c] / weight_sum) * NUM_CLASSES for c in range(NUM_CLASSES)
    }

    # Weight Clamping Guard: Also clamp normalized weights if they exceed max_weight
    clamped_normalized_weights = {
        c: min(float(raw_normalized_weights[c]), float(max_weight)) for c in range(NUM_CLASSES)
    }

    # Median frequency weighting: w_c = median(freq) / freq_c (capped to avoid div by zero)
    non_zero_freqs = [f for f in frequencies.values() if f > 0]
    med_freq = float(np.median(non_zero_freqs)) if non_zero_freqs else 1.0
    for c in range(NUM_CLASSES):
        f = frequencies[c]
        raw_med = (med_freq / f) if f > 0 else 0.0
        median_freq_weights[c] = min(raw_med, float(max_weight))

    pytorch_tensor_snippet = (
        f"torch.tensor({[round(clamped_normalized_weights[c], 4) for c in range(NUM_CLASSES)]}, dtype=torch.float32)"
    )

    stats_summary = {
        "dataset_summary": {
            "total_masks_scanned": total_masks,
            "total_masks_analyzed": valid_masks_count,
            "successfully_analyzed_masks": valid_masks_count,
            "total_pixels_counted": total_valid_pixels,
            "smoothing_constant_c0": c0,
            "max_weight_ceiling": max_weight,
            "invalid_pixel_values_encountered": dict(invalid_pixel_counter),
            "dimension_mismatches_with_images": len(dimension_mismatches),
            "missing_paired_images": missing_paired_images,
            "common_resolutions": [
                {"width": dim[0], "height": dim[1], "count": count}
                for dim, count in dimensions_counter.most_common(5)
            ],
            "common_aspect_ratios": [
                {"aspect_ratio": ar, "count": count}
                for ar, count in aspect_ratios_counter.most_common(5)
            ],
        },
        "class_statistics": [
            {
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "pixel_count": class_pixel_counts[c],
                "percentage": round(percentage_distribution[c], 4),
                "image_presence_count": images_with_class[c],
                "image_presence_percentage": round(
                    (images_with_class[c] / max(1, valid_masks_count)) * 100.0, 2
                ),
                "raw_inv_log_weight": round(raw_enet_weights[c], 5),
                "unclamped_loss_weight": round(raw_normalized_weights[c], 5),
                "normalized_loss_weight": round(clamped_normalized_weights[c], 5),
                "median_freq_weight": round(median_freq_weights[c], 5),
                "was_clamped": was_clamped_flag[c] or (raw_normalized_weights[c] > max_weight),
            }
            for c in range(NUM_CLASSES)
        ],
        "pytorch_weight_tensor": [
            round(clamped_normalized_weights[c], 4) for c in range(NUM_CLASSES)
        ],
        "pytorch_weight_snippet": pytorch_tensor_snippet,
    }

    return stats_summary, per_image_records


def print_eda_report(stats: Dict, c0: float, max_weight: float) -> None:
    """Prints a structured, human-readable terminal report."""
    ds = stats["dataset_summary"]
    classes = stats["class_statistics"]

    print("\n" + "=" * 80)
    print("           EURS CHALLENGE: DATASET SEGMENTATION EDA REPORT")
    print("=" * 80)
    print(f"Total Masks Scanned      : {ds['total_masks_scanned']}")
    print(f"Successfully Analyzed    : {ds['successfully_analyzed_masks']}")
    print(f"Total Pixels Counted     : {ds['total_pixels_counted']:,}")
    print(f"Smoothing Constant (c0)  : {c0}")
    print(f"Max Weight Cap           : {max_weight:.1f} (Mixed-Precision Guard)")

    print("\nTop Dimensions (Width x Height):")
    for res in ds["common_resolutions"]:
        pct = (res["count"] / max(1, ds["successfully_analyzed_masks"])) * 100
        print(f"  - {res['width']:4d} x {res['height']:4d} : {res['count']:4d} masks ({pct:5.1f}%)")

    print("\nTop Aspect Ratios (W / H):")
    for ar in ds["common_aspect_ratios"]:
        print(f"  - Aspect Ratio {ar['aspect_ratio']:0.4f} : {ar['count']:4d} masks")

    if ds["invalid_pixel_values_encountered"]:
        print("\n[WARNING] Found Out-Of-Range Mask Pixel Values:")
        for val, count in ds["invalid_pixel_values_encountered"].items():
            print(f"  * Class Index {val}: {count:,} pixels (Expected only 0..4)")
    else:
        print("\nPixel Integrity Check    : [PASSED] All mask pixels strictly in range [0..4].")

    print("\n" + "-" * 80)
    print(
        f"{'ID':<3} | {'Class Name':<26} | {'Pixel Count':<14} | {'Share (%)':<9} | {'Presence':<9} | {'Loss Weight':<10}"
    )
    print("-" * 80)
    for c in classes:
        clamped_flag = " [CLAMPED]" if c["was_clamped"] else ""
        print(
            f"{c['class_id']:<3} | "
            f"{c['class_name']:<26} | "
            f"{c['pixel_count']:>14,d} | "
            f"{c['percentage']:>8.3f}% | "
            f"{c['image_presence_percentage']:>7.1f}% | "
            f"{c['normalized_loss_weight']:>10.4f}{clamped_flag}"
        )
    print("-" * 80)

    py_tensor_str = stats["pytorch_weight_snippet"]
    print("\n>>> PyTorch Ready Loss Weight Snippet (losses/multi_task_loss.py):")
    print(f"{py_tensor_str}")
    print("\nExample usage in model training:")
    print(f"CLASS_WEIGHTS = {py_tensor_str}")
    print("criterion_ce = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))\n")
    print("=" * 80 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        stats, records = analyze_dataset_masks(
            mask_dir=args.mask_dir,
            images_dir=args.images_dir,
            c0=args.c0,
            max_weight=args.max_weight,
            max_files=args.max_files,
            quiet=args.quiet,
        )
    except Exception as e:
        logger.error(f"EDA Analysis failed: {e}")
        return 1

    print_eda_report(stats, args.c0, args.max_weight)

    # Save to JSON if requested
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Exported statistics JSON to: {args.save_json.resolve()}")

    # Save to CSV if requested
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv

        if records:
            fieldnames = list(records[0].keys())
            with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Exported per-image CSV to: {args.save_csv.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
