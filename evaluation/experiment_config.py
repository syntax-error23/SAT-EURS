"""
Experiment Configuration Specification for SAT-EURS.
Guarantees that controlled resolution experiments (A, B, C, D) modify ONLY
the preprocessing configuration (resolution and mode) while holding all other
variables strictly identical.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple


@dataclass
class PreprocessingConfig:
    """
    Configuration for input spatial preprocessing.
    """
    resolution: int = 1024
    mode: str = "direct"  # "direct" or "letterbox"

    def __post_init__(self):
        if self.resolution % 32 != 0:
            raise ValueError(f"Resolution must be divisible by 32, got {self.resolution}")
        if self.mode not in ("direct", "letterbox"):
            raise ValueError(f"Mode must be 'direct' or 'letterbox', got '{self.mode}'")


from utils.paths import get_dataset_root, get_splits_path


@dataclass
class ExperimentConfig:
    """
    Complete configuration for controlled EURS experiments.
    """
    name: str = "exp_baseline"
    dataset_root: str = field(default_factory=get_dataset_root)
    splits_file: str = field(default_factory=get_splits_path)
    tier: str = "tier1"

    # Preprocessing variable
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Invariant Training Parameters (held strictly identical across comparisons)
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    device: str = "cuda"

    # Loss weights (held strictly identical across comparisons)
    weight_seg: float = 1.0
    weight_rest: float = 0.5


def get_controlled_candidate_config(candidate_id: str) -> ExperimentConfig:
    """
    Returns the exact configuration for the 4 candidate preprocessing strategies.
      Candidate A: 768x768 direct resize
      Candidate B: 1024x1024 direct resize
      Candidate C: 768x768 aspect-preserving letterbox
      Candidate D: 1024x1024 aspect-preserving letterbox
    """
    cid = candidate_id.upper().strip()
    configs = {
        "A": ExperimentConfig(name="exp_768_direct", preprocessing=PreprocessingConfig(resolution=768, mode="direct")),
        "B": ExperimentConfig(name="exp_1024_direct", preprocessing=PreprocessingConfig(resolution=1024, mode="direct")),
        "C": ExperimentConfig(name="exp_768_letterbox", preprocessing=PreprocessingConfig(resolution=768, mode="letterbox")),
        "D": ExperimentConfig(name="exp_1024_letterbox", preprocessing=PreprocessingConfig(resolution=1024, mode="letterbox")),
    }
    if cid not in configs:
        raise ValueError(f"Unknown candidate '{candidate_id}'. Choose from 'A', 'B', 'C', 'D'.")
    return configs[cid]
