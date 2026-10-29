"""
Evaluation package for SAT-EURS.
"""

from evaluation.metrics import (
    compute_segmentation_metrics,
    compute_restoration_metrics,
    compute_psnr,
    compute_ssim,
    compute_sharpness_ratio,
    MetricAccumulator,
)
from evaluation.experiment_config import (
    PreprocessingConfig,
    ExperimentConfig,
    get_controlled_candidate_config,
)
from evaluation.pipeline import evaluate_model, run_validation_evaluation

__all__ = [
    "compute_segmentation_metrics",
    "compute_restoration_metrics",
    "compute_psnr",
    "compute_ssim",
    "compute_sharpness_ratio",
    "MetricAccumulator",
    "PreprocessingConfig",
    "ExperimentConfig",
    "get_controlled_candidate_config",
    "evaluate_model",
    "run_validation_evaluation",
]
