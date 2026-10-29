"""
Centralized, Cross-Platform Path Management Authority for SAT-EURS.
Single source of truth for all repository paths across Windows, Linux, and macOS.
Resolves dataset roots, manifests, and project directories portably.
"""

import os
from pathlib import Path
from typing import Optional

# Repository root is anchored two levels up from this file (MACVI/utils/paths.py -> MACVI/)
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT

# Canonical Project Directories
DATA_DIR = REPO_ROOT / "data"
RAW_CLEAN_DIR = DATA_DIR / "clean"
RAW_MASKS_DIR = DATA_DIR / "masks"
RAW_DEGRADED_DIR = DATA_DIR / "degraded"
CACHE_1024_DIR = DATA_DIR / "cache_1024"

SPLITS_DIR = REPO_ROOT / "splits"
FROZEN_MANIFEST_PATH = SPLITS_DIR / "manifest_full.json"
FROZEN_MANIFEST_HASH = SPLITS_DIR / "manifest_full.json.sha256"

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
AUDITS_DIR = REPO_ROOT / "audits"
DEPLOY_DIR = REPO_ROOT / "deploy"
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_DIR = REPO_ROOT / "models"
TESTS_DIR = REPO_ROOT / "tests"
TOOLS_DIR = REPO_ROOT / "tools"
REPORTS_DIR = REPO_ROOT / "reports"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Candidate dataset root paths to probe in priority order:
_ENV_VAR = "EURS_DATASET_ROOT"
_CANDIDATE_PATHS = [
    Path(os.environ[_ENV_VAR]) if _ENV_VAR in os.environ else None,
    DATA_DIR,
    DATA_DIR / "mini_dev",
    Path.home() / "Datasets" / "eurs",
    Path("/home/shubk/Datasets/eurs"),
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
        if p.is_dir() or fallback_ok:
            return str(p)

    # 2. Probe candidate paths
    for candidate in _CANDIDATE_PATHS:
        if candidate is not None and candidate.is_dir():
            return str(candidate.resolve())

    # 3. Default fallback
    return str(DATA_DIR)


def get_splits_path(splits_file: Optional[str] = None) -> str:
    """Returns absolute path to splits.json."""
    if splits_file:
        p = Path(splits_file)
        if p.is_absolute():
            return str(p)
        return str((PROJECT_ROOT / p).resolve())
    return str((DATA_DIR / "splits.json").resolve())


def get_stats_path(stats_file: Optional[str] = None) -> str:
    """Returns absolute path to dataset_stats.json."""
    if stats_file:
        p = Path(stats_file)
        if p.is_absolute():
            return str(p)
        return str((PROJECT_ROOT / p).resolve())
    return str((DATA_DIR / "dataset_stats.json").resolve())
