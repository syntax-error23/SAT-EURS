"""
Centralized Path Management Authority
Single source of truth for all repository paths across Windows, Linux, and macOS.
Eliminates any hardcoded drive letters or absolute platform paths.
"""

import os
from pathlib import Path

# Repository root is anchored two levels up from this file (MACVI/utils/paths.py -> MACVI/)
REPO_ROOT = Path(__file__).resolve().parents[1]

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
