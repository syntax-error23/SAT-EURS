"""
Comprehensive Checkpointing & State Restoration Utility
Saves and restores:
- Model parameters
- Optimizer state & momentum buffers
- LR Scheduler state
- AMP GradScaler state (if present)
- PyTorch, NumPy, and Python standard RNG states
- Epoch, Global Step, Best Metric Score, and Custom Metadata
Ensures exact CPU deterministic continuation across training interruptions.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import random
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_score: float = -1.0,
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    rng_states = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }
    if torch.cuda.is_available():
        rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "rng_states": rng_states,
        "extra_state": extra_state or {}
    }

    # Save to temp file first then atomic rename to prevent corrupted checkpoints
    temp_filepath = filepath + ".tmp"
    torch.save(checkpoint, temp_filepath)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(temp_filepath, filepath)

    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
    map_location: str = "cpu"
) -> Dict[str, Any]:
    assert os.path.exists(filepath), f"Checkpoint not found: {filepath}"
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions that do not support weights_only argument
        checkpoint = torch.load(filepath, map_location=map_location)

    # 1. Restore Model Weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Restore Optimizer State
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Restore Scheduler State
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 4. Restore Scaler State
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 5. Restore RNG States
    if restore_rng and "rng_states" in checkpoint:
        rngs = checkpoint["rng_states"]
        if "torch_cpu" in rngs:
            torch.set_rng_state(rngs["torch_cpu"])
        if "numpy" in rngs:
            np.random.set_state(rngs["numpy"])
        if "python" in rngs:
            random.setstate(rngs["python"])
        if torch.cuda.is_available() and "torch_cuda" in rngs and rngs["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rngs["torch_cuda"])

    return checkpoint
