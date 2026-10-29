"""
EURS Challenge Tooling Package
(Embedded Underwater Restoration and Segmentation @ MaCVi / WACV 2027)
"""

from tools.eda_stats import CLASS_NAMES, NUM_CLASSES, analyze_dataset_masks
from tools.make_submission import (
    VALID_CLASS_IDS,
    calculate_target_dims,
    create_submission_zip,
    validate_and_stage_files,
    validate_mask,
)
from tools.visualizer import CLASS_METADATA, colorize_mask, render_tri_panel

__all__ = [
    "analyze_dataset_masks",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "colorize_mask",
    "render_tri_panel",
    "CLASS_METADATA",
    "calculate_target_dims",
    "validate_mask",
    "validate_and_stage_files",
    "create_submission_zip",
    "VALID_CLASS_IDS",
]
