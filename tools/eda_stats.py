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
  where c_0 defaults to 1.02, as well as normalized and median-frequency weights.
- Generates a ready-to-use PyTorch tensor snippet for losses/multi_task_loss.py.
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
    max_files: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Analyzes segmentation masks in mask_dir, calculating pixel distributions,
    class imbalance weights, and dimension characteristics.
    """
    if c0 <= 1.0:
        raise ValueError(f"Parameter c0 must be strictly > 1.0 (got {c0}) to prevent division by zero.")

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
                        # Try with alternate common extensions
                        for ext in [".png", ".jpg", ".jpeg"]:
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

    # Calculate frequencies and inverse weights
    frequencies: Dict[int, float] = {}
    percentage_distribution: Dict[int, float] = {}
    inv_log_weights: Dict[int, float] = {}
    median_freq_weights: Dict[int, float] = {}

    for c in range(NUM_CLASSES):
        count = class_pixel_counts[c]
        freq = count / total_valid_pixels
        frequencies[c] = freq
        percentage_distribution[c] = freq * 100.0
        # ENet formulation: w_c = 1 / ln(c_0 + freq_c)
        inv_log_weights[c] = 1.0 / math.log(c0 + freq)

    # Normalized weights so that the sum of weights equals NUM_CLASSES (standard PyTorch practice)
    weight_sum = sum(inv_log_weights.values())
    normalized_weights = {
        c: (inv_log_weights[c] / weight_sum) * NUM_CLASSES for c in range(NUM_CLASSES)
    }

    # Median frequency weighting: w_c = median(freq) / freq_c (capped to avoid div by zero)
    non_zero_freqs = [f for f in frequencies.values() if f > 0]
    med_freq = float(np.median(non_zero_freqs)) if non_zero_freqs else 1.0
    for c in range(NUM_CLASSES):
        f = frequencies[c]
        median_freq_weights[c] = (med_freq / f) if f > 0 else 0.0

    stats_summary = {
        "dataset_summary": {
            "total_masks_analyzed": total_masks,
            "total_pixels_counted": total_valid_pixels,
            "invalid_pixel_values_encountered": dict(invalid_pixel_counter),
            "dimension_mismatches_with_images": len(dimension_mismatches),
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
                    (images_with_class[c] / max(1, len(per_image_records))) * 100.0, 2
                ),
                "raw_inv_log_weight": round(inv_log_weights[c], 5),
                "normalized_loss_weight": round(normalized_weights[c], 5),
                "median_freq_weight": round(median_freq_weights[c], 5),
            }
            for c in range(NUM_CLASSES)
        ],
        "pytorch_weight_tensor": [
            round(normalized_weights[c], 4) for c in range(NUM_CLASSES)
        ],
    }

    return stats_summary, per_image_records


def print_eda_report(stats: Dict, c0: float) -> None:
    """Prints a structured, human-readable terminal report."""
    ds = stats["dataset_summary"]
    classes = stats["class_statistics"]

    print("\n" + "=" * 80)
    print("           EURS CHALLENGE: DATASET SEGMENTATION EDA REPORT")
    print("=" * 80)
    print(f"Total Masks Scanned      : {ds['total_masks_analyzed']}")
    print(f"Total Pixels Counted     : {ds['total_pixels_counted']:,}")
    print(f"Smoothing Constant (c0)  : {c0}")

    print("\nTop Dimensions (Width x Height):")
    for res in ds["common_resolutions"]:
        pct = (res["count"] / ds["total_masks_analyzed"]) * 100
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
        print(
            f"{c['class_id']:<3} | "
            f"{c['class_name']:<26} | "
            f"{c['pixel_count']:>14,d} | "
            f"{c['percentage']:>8.3f}% | "
            f"{c['image_presence_percentage']:>7.1f}% | "
            f"{c['normalized_loss_weight']:>10.4f}"
        )
    print("-" * 80)

    py_weights = stats["pytorch_weight_tensor"]
    print("\n>>> PyTorch Ready Loss Weight Snippet (losses/multi_task_loss.py):")
    print(
        f"CLASS_WEIGHTS = torch.tensor({py_weights}, dtype=torch.float32)"
    )
    print(
        "criterion_ce = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))\n"
    )
    print("=" * 80 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        stats, records = analyze_dataset_masks(
            mask_dir=args.mask_dir,
            images_dir=args.images_dir,
            c0=args.c0,
            max_files=args.max_files,
            quiet=args.quiet,
        )
    except Exception as e:
        logger.error(f"EDA Analysis failed: {e}")
        return 1

    print_eda_report(stats, args.c0)

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
