"""
Training module for EURS-Net V2.
Includes:
- TrainingConfig and candidate experiment presets (A, B, C, D)
- EURSTrainer with reproducible multi-task training, valid_mask handling,
  structured epoch logging, NaN/Inf detection, checkpointing, and evaluation.
"""

from training.config import TrainingConfig, get_candidate_training_config, get_all_candidate_configs
from training.trainer import EURSTrainer

__all__ = [
    "TrainingConfig",
    "get_candidate_training_config",
    "get_all_candidate_configs",
    "EURSTrainer"
]
