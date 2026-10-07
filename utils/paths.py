"""
Portable, cross-platform path resolution for SAT-EURS.
Resolves dataset roots, manifests, and project directories across Linux, Windows,
and cloud/GPU environments without hardcoding user-specific paths.
"""

import os
from pathlib import Path
from typing import Optional

# Repository root: two levels up from utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Candidate dataset root paths to check in order of priority:
_ENV_VAR = "EURS_DATASET_ROOT"
_CANDIDATE_PATHS = [
    Path(os.environ[_ENV_VAR]) if _ENV_VAR in os.environ else None,
    Path("/home/shubk/Datasets/eurs"),
    Path.home() / "Datasets" / "eurs",
    PROJECT_ROOT / "data" / "eurs",
    PROJECT_ROOT / "datasets" / "eurs",
]


def get_dataset_root(fallback_ok: bool = True) -> str:
    """
    Resolves the dataset root directory portably.
    Checks EURS_DATASET_ROOT env var first, then system candidate paths.
    """
    # 1. Environment variable override
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        p = Path(env_path).resolve()
        if p.is_dir():
            return str(p)
        elif fallback_ok:
            return str(p)

    # 2. Probe candidate paths
    for candidate in _CANDIDATE_PATHS:
        if candidate is not None and candidate.is_dir():
            return str(candidate.resolve())

    # 3. Default fallback
    default_p = Path.home() / "Datasets" / "eurs"
    return str(default_p)


DATA_DIR = Path(get_dataset_root())


def get_splits_path(splits_file: Optional[str] = None) -> str:
    """Returns absolute path to splits.json."""
    if splits_file:
        p = Path(splits_file)
        if p.is_absolute():
            return str(p)
        return str((PROJECT_ROOT / p).resolve())
    return str((PROJECT_ROOT / "data" / "splits.json").resolve())


def get_stats_path(stats_file: Optional[str] = None) -> str:
    """Returns absolute path to dataset_stats.json."""
    if stats_file:
        p = Path(stats_file)
        if p.is_absolute():
            return str(p)
        return str((PROJECT_ROOT / p).resolve())
    return str((PROJECT_ROOT / "data" / "dataset_stats.json").resolve())
