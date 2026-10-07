"""
Training Configuration Specification for SAT-EURS Controlled Resolution Study.

Ensures that the 4 experimental candidates:
  Candidate A: 768x768 direct resize
  Candidate B: 1024x1024 direct resize
  Candidate C: 768x768 aspect-preserving letterbox
  Candidate D: 1024x1024 aspect-preserving letterbox
modify ONLY the preprocessing resolution and mode, while holding all training,
optimization, loss, regularization, seeding, and evaluation parameters strictly identical.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List


from utils.paths import get_dataset_root, get_splits_path, get_stats_path


@dataclass
class TrainingConfig:
    """
    Configuration for EURS-Net V2 controlled training experiments.
    """
    candidate_id: str = "A"
    experiment_name: str = "A_768_direct"
    dataset_root: str = field(default_factory=get_dataset_root)
    splits_file: str = field(default_factory=get_splits_path)
    stats_file: str = field(default_factory=get_stats_path)
    tier: str = "tier1"
    output_dir: str = "experiments/resolution_study"

    # Preprocessing variable (the ONLY variable between candidates)
    resolution: int = 768
    mode: str = "direct"  # "direct" or "letterbox"

    # Controlled Invariant Training Contract (held strictly identical across A, B, C, D)
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    device: str = "cuda"

    # Multi-task loss weights (official competition proportion 2:1)
    weight_seg: float = 1.0
    weight_rest: float = 0.5
    use_class_weights: bool = True

    # Optimization stability & efficiency
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    use_augmentation: bool = True  # Synchronized horizontal & vertical flips

    # Checkpointing & Validation
    eval_every_epoch: int = 1
    checkpoint_metric: str = "val_loss"  # "val_loss"
    checkpoint_metric_mode: str = "min"  # "min" for val_loss

    def __post_init__(self):
        if self.resolution % 32 != 0:
            raise ValueError(f"Resolution must be divisible by 32, got {self.resolution}")
        if self.mode not in ("direct", "letterbox"):
            raise ValueError(f"Mode must be 'direct' or 'letterbox', got '{self.mode}'")
        if self.checkpoint_metric_mode not in ("min", "max"):
            raise ValueError(f"checkpoint_metric_mode must be 'min' or 'max', got '{self.checkpoint_metric_mode}'")

    @property
    def experiment_dir(self) -> str:
        return os.path.join(self.output_dir, self.experiment_name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_candidate_training_config(candidate_id: str, **overrides) -> TrainingConfig:
    """
    Returns the exact configuration for the four controlled candidates.
      A: 768x768 direct resize
      B: 1024x1024 direct resize
      C: 768x768 aspect-preserving letterbox
      D: 1024x1024 aspect-preserving letterbox
    """
    cid = candidate_id.upper().strip()
    presets = {
        "A": {
            "candidate_id": "A",
            "experiment_name": "A_768_direct",
            "resolution": 768,
            "mode": "direct"
        },
        "B": {
            "candidate_id": "B",
            "experiment_name": "B_1024_direct",
            "resolution": 1024,
            "mode": "direct"
        },
        "C": {
            "candidate_id": "C",
            "experiment_name": "C_768_letterbox",
            "resolution": 768,
            "mode": "letterbox"
        },
        "D": {
            "candidate_id": "D",
            "experiment_name": "D_1024_letterbox",
            "resolution": 1024,
            "mode": "letterbox"
        }
    }

    if cid not in presets:
        raise ValueError(f"Unknown candidate '{candidate_id}'. Choose from 'A', 'B', 'C', 'D'.")

    cfg_dict = presets[cid].copy()
    cfg_dict.update(overrides)
    return TrainingConfig(**cfg_dict)


def get_all_candidate_configs(**overrides) -> Dict[str, TrainingConfig]:
    """Returns a dictionary mapping candidate keys ('A', 'B', 'C', 'D') to their TrainingConfig."""
    return {cid: get_candidate_training_config(cid, **overrides) for cid in ["A", "B", "C", "D"]}
