"""
tools/visualizer.py
-------------------
Visualization and Tri-Panel Comparison Tool for EURS Challenge
(Embedded Underwater Restoration and Segmentation @ MaCVi / WACV 2027).

Role & Ownership: Tooling & Logging (Arin)

Functionality:
- Generates high-resolution 3-panel comparison figures:
    [Degraded Input Image | Clean Ground-Truth Image | Ground-Truth Segmentation Mask (color-mapped)]
- Supports single sample inspection (--index or --sample-id) and batch processing (--num-samples).
- Applies a distinct, high-contrast 5-class underwater color palette with custom legend:
    0: Background (Deep Navy)
    1: Coral (Vibrant Coral/Orange)
    2: Fish (Golden Yellow)
    3: Elasmobranch / Sharks & Rays (Bright Turquoise/Cyan)
    4: Other (Electric Purple)
- Supports optional overlay blending (--overlay) and prediction comparison (--pred-restored-dir / --pred-mask-dir).
- Saves all publication-ready comparison figures into experiments/plots/ for team and faculty review.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive Agg backend by default for headless/server safety
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualizer")

# 5-Class Metadata and Color Palette (RGB in 0..1 range)
CLASS_METADATA: Dict[int, Dict] = {
    0: {"name": "Background", "color": [0.06, 0.09, 0.16], "hex": "#0f172a"},      # Deep Navy
    1: {"name": "Coral", "color": [0.98, 0.45, 0.09], "hex": "#f97316"},           # Coral Orange
    2: {"name": "Fish", "color": [0.92, 0.70, 0.03], "hex": "#eab308"},            # Golden Yellow
    3: {"name": "Elasmobranch", "color": [0.02, 0.71, 0.83], "hex": "#06b6d4"},    # Bright Cyan/Turquoise
    4: {"name": "Other", "color": [0.66, 0.33, 0.97], "hex": "#a855f7"},           # Purple
}
NUM_CLASSES = 5


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Maps 2D single-channel mask (values 0..4) into an RGB array [H, W, 3] in range [0, 1].
    """
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.float32)

    for cls_id in range(NUM_CLASSES):
        match = (mask == cls_id)
        if np.any(match):
            colored[match] = CLASS_METADATA[cls_id]["color"]

    # Any invalid values mapped to warning red
    invalid = (mask < 0) | (mask >= NUM_CLASSES)
    if np.any(invalid):
        colored[invalid] = [1.0, 0.0, 0.0]

    return colored


def find_paired_file(target_stem: str, directory: Path) -> Optional[Path]:
    """Finds a matching image file in directory by stem, checking common extensions."""
    for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".BMP"]:
        candidate = directory / f"{target_stem}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def discover_common_stems(
    degraded_dir: Path,
    clean_dir: Path,
    mask_dir: Path,
) -> List[str]:
    """Discovers file stems present across degraded, clean, and mask directories."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    
    degraded_stems = {p.stem for p in degraded_dir.glob("*") if p.suffix.lower() in extensions}
    clean_stems = {p.stem for p in clean_dir.glob("*") if p.suffix.lower() in extensions}
    mask_stems = {p.stem for p in mask_dir.glob("*") if p.suffix.lower() in extensions}

    common = sorted(list(degraded_stems & clean_stems & mask_stems))
    return common


def render_tri_panel(
    degraded_img: np.ndarray,
    clean_img: np.ndarray,
    mask_arr: np.ndarray,
    sample_id: str,
    output_path: Path,
    overlay: bool = False,
    pred_restored: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
) -> Path:
    """
    Generates and saves the comparison figure using Matplotlib.
    Defaults to 3 panels: [Degraded Input | Clean Ground-Truth | GT Mask].
    If predictions are passed, renders an extended 5-panel layout.
    """
    mask_rgb = colorize_mask(mask_arr)
    unique_classes = [c for c in range(NUM_CLASSES) if np.any(mask_arr == c)]

    num_panels = 3
    if pred_restored is not None and pred_mask is not None:
        num_panels = 5

    # Matplotlib dark styling
    plt.style.use("dark_background")
    fig_width = 6.2 * num_panels
    fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, 6.5), dpi=150)

    if num_panels == 1:
        axes = [axes]

    h, w = degraded_img.shape[:2]

    # Panel 1: Degraded Input
    axes[0].imshow(degraded_img)
    axes[0].set_title(f"[1] Degraded Input Image\n({w} x {h})", fontsize=11, fontweight="bold", color="#38bdf8", pad=8)
    axes[0].axis("off")

    # Panel 2: Clean Ground-Truth
    axes[1].imshow(clean_img)
    axes[1].set_title(f"[2] Clean Ground-Truth Image\n({w} x {h})", fontsize=11, fontweight="bold", color="#34d399", pad=8)
    axes[1].axis("off")

    # Panel 3: Ground-Truth Segmentation Mask (or overlay)
    if overlay:
        blended = (0.55 * clean_img + 0.45 * mask_rgb).clip(0.0, 1.0)
        axes[2].imshow(blended)
        axes[2].set_title("[3] GT Mask (Overlay 45%)\nClasses 0..4", fontsize=11, fontweight="bold", color="#fbbf24", pad=8)
    else:
        axes[2].imshow(mask_rgb)
        axes[2].set_title("[3] GT Segmentation Mask\n(5-Class Palette)", fontsize=11, fontweight="bold", color="#fbbf24", pad=8)
    axes[2].axis("off")

    # Optional Panels 4 & 5: Model Predictions
    if num_panels == 5:
        axes[3].imshow(pred_restored)
        axes[3].set_title("[4] Model Restored Output\n(Predicted RGB)", fontsize=11, fontweight="bold", color="#a855f7", pad=8)
        axes[3].axis("off")

        pred_mask_rgb = colorize_mask(pred_mask)
        axes[4].imshow(pred_mask_rgb)
        axes[4].set_title("[5] Predicted Seg Mask\n(Argmax Logits)", fontsize=11, fontweight="bold", color="#f43f5e", pad=8)
        axes[4].axis("off")

    # Build Class Legend
    legend_patches = []
    for cls_id in range(NUM_CLASSES):
        meta = CLASS_METADATA[cls_id]
        is_present = cls_id in unique_classes
        label_text = f"{cls_id}: {meta['name']}" + (" *" if is_present else "")
        patch = mpatches.Patch(
            facecolor=meta["color"],
            edgecolor="#ffffff" if is_present else "#475569",
            linewidth=1.2 if is_present else 0.5,
            label=label_text,
        )
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        facecolor="#141e33",
        edgecolor="#223254",
        fontsize=9,
        title="* Denotes class present in this ground-truth sample",
        title_fontsize=9,
    )

    present_names = [CLASS_METADATA[c]["name"] for c in unique_classes if c != 0]
    present_str = ", ".join(present_names) if present_names else "Background only"
    fig.suptitle(
        f"EURS Tri-Panel Inspection | Sample ID: {sample_id} | Detected Targets: [{present_str}]",
        fontsize=13,
        fontweight="bold",
        color="#f1f5f9",
        y=0.98,
    )

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="#090d16", edgecolor="none")
    plt.close(fig)

    return output_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3-panel comparison figures [Degraded | Clean | GT Mask] for EURS challenge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--degraded-dir",
        type=Path,
        default=Path("data/degraded"),
        help="Path to directory containing degraded underwater images.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/clean"),
        help="Path to directory containing clean ground-truth images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data/masks"),
        help="Path to directory containing ground-truth segmentation masks (values 0..4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="Destination directory to save generated comparison plots.",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        default=None,
        help="0-based index of a specific sample to visualize.",
    )
    parser.add_argument(
        "--sample-id",
        "-s",
        type=str,
        default=None,
        help="Specific filename stem or sample ID to visualize (e.g., 'sample_0012').",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1,
        help="Number of samples to visualize in batch mode (used if neither --index nor --sample-id is specified).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly pick samples instead of taking sequential indices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random selection.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay segmentation mask onto the clean ground-truth image in panel 3.",
    )
    parser.add_argument(
        "--pred-restored-dir",
        type=Path,
        default=None,
        help="Optional directory containing predicted restored images (enables 5-panel mode).",
    )
    parser.add_argument(
        "--pred-mask-dir",
        type=Path,
        default=None,
        help="Optional directory containing predicted segmentation masks (enables 5-panel mode).",
    )
    return parser.parse_args(argv)


def load_image_normalized(path: Path) -> np.ndarray:
    """Loads an RGB image into float32 array in [0.0, 1.0]."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    """Loads a single-channel segmentation mask into integer array."""
    with Image.open(path) as img:
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.int64)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    for name, p in [
        ("Degraded", args.degraded_dir),
        ("Clean", args.clean_dir),
        ("Mask", args.mask_dir),
    ]:
        if not p.exists():
            logger.error(f"{name} directory does not exist: {p.resolve()}")
            return 1

    common_stems = discover_common_stems(
        args.degraded_dir,
        args.clean_dir,
        args.mask_dir,
    )

    if not common_stems:
        logger.error(
            f"No matching image/mask pairs found with common stems across:\n"
            f"  - {args.degraded_dir}\n"
            f"  - {args.clean_dir}\n"
            f"  - {args.mask_dir}"
        )
        return 1

    logger.info(f"Discovered {len(common_stems)} matching sample triplets.")

    # Determine targets to visualize
    targets: List[str] = []
    if args.sample_id is not None:
        target_stem = Path(args.sample_id).stem
        if target_stem not in common_stems:
            logger.error(f"Sample ID '{args.sample_id}' not found among matching dataset samples!")
            return 1
        targets = [target_stem]
    elif args.index is not None:
        if args.index < 0 or args.index >= len(common_stems):
            logger.error(
                f"Requested index {args.index} out of range [0..{len(common_stems) - 1}]."
            )
            return 1
        targets = [common_stems[args.index]]
    else:
        num = min(args.num_samples, len(common_stems))
        if args.random:
            rng = random.Random(args.seed)
            targets = rng.sample(common_stems, num)
        else:
            targets = common_stems[:num]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    for stem in targets:
        deg_path = find_paired_file(stem, args.degraded_dir)
        clean_path = find_paired_file(stem, args.clean_dir)
        mask_path = find_paired_file(stem, args.mask_dir)

        if deg_path is None or clean_path is None or mask_path is None:
            logger.warning(f"Skipping {stem}: one or more paired files could not be resolved.")
            continue

        try:
            deg_img = load_image_normalized(deg_path)
            clean_img = load_image_normalized(clean_path)
            mask_arr = load_mask(mask_path)
        except Exception as e:
            logger.error(f"Error reading sample {stem}: {e}")
            continue

        # Optional prediction loading
        pred_restored = None
        pred_mask = None
        if args.pred_restored_dir is not None and args.pred_mask_dir is not None:
            pred_res_path = find_paired_file(stem, args.pred_restored_dir)
            pred_m_path = find_paired_file(stem, args.pred_mask_dir)
            if pred_res_path and pred_m_path:
                pred_restored = load_image_normalized(pred_res_path)
                pred_mask = load_mask(pred_m_path)

        out_file = args.output_dir / f"compare_{stem}.png"
        saved = render_tri_panel(
            degraded_img=deg_img,
            clean_img=clean_img,
            mask_arr=mask_arr,
            sample_id=stem,
            output_path=out_file,
            overlay=args.overlay,
            pred_restored=pred_restored,
            pred_mask=pred_mask,
        )
        generated_plots.append(saved)
        logger.info(f"Generated comparison plot -> {saved.resolve()}")

    logger.info(f"Done! Successfully generated {len(generated_plots)} plot(s) in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
