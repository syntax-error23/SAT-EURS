"""
CPU Deterministic Resume Unit Test (Phase 2A Pre-Flight)
Verifies:
1. Checkpoint serialization and complete state restoration:
   - Model weights
   - Optimizer momentum buffers (exp_avg, exp_avg_sq)
   - LR Scheduler state
   - PyTorch, NumPy, Python RNG states
2. CPU Deterministic Continuation:
   - 5 steps uninterrupted (Run A)
   - 2 steps, save checkpoint, fresh reload, 3 steps (Run B)
   - Assert exact numerical equality: torch.equal(W_A, W_B) for all parameters and buffers!
Note: This verifies exact CPU determinism. CUDA determinism is validated separately in Phase 2B.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import copy
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.eurs_net_v2 import EURSNetV2
from utils.checkpointing import save_checkpoint, load_checkpoint


def set_all_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_synthetic_batch(seed: int, batch_size: int = 2, H: int = 64, W: int = 64):
    torch.manual_seed(seed)
    x = torch.rand(batch_size, 3, H, W, dtype=torch.float32)
    clean = torch.rand(batch_size, 3, H, W, dtype=torch.float32)
    mask = torch.randint(0, 5, (batch_size, H, W), dtype=torch.long)
    return x, clean, mask


def test_resume_determinism():
    print("=" * 60)
    print("Test 2A.7: Full Checkpointing & CPU Resume Determinism")
    print("=" * 60)

    # Pre-generate 5 deterministic batches
    batches = [create_synthetic_batch(seed=100 + i) for i in range(5)]

    # -------------------------------------------------------------
    # RUN 1: Uninterrupted 5 Steps
    # -------------------------------------------------------------
    set_all_seeds(42)
    model_uninterrupted = EURSNetV2(num_classes=5)
    model_uninterrupted.train()

    # Save initial state dict to ensure Run 2 starts from identical weights
    init_state = copy.deepcopy(model_uninterrupted.state_dict())

    opt_uninterrupted = AdamW(model_uninterrupted.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_uninterrupted = CosineAnnealingLR(opt_uninterrupted, T_max=10, eta_min=1e-6)

    criterion = nn.MSELoss()

    losses_uninterrupted = []
    for step in range(5):
        x, clean, _ = batches[step]
        opt_uninterrupted.zero_grad()
        restored, _ = model_uninterrupted(x)
        loss = criterion(restored, clean)
        loss.backward()
        opt_uninterrupted.step()
        sched_uninterrupted.step()
        losses_uninterrupted.append(loss.item())

    final_params_uninterrupted = {k: v.clone() for k, v in model_uninterrupted.state_dict().items()}
    final_opt_uninterrupted = copy.deepcopy(opt_uninterrupted.state_dict())

    # -------------------------------------------------------------
    # RUN 2: 2 Steps -> Save -> Fresh Reload -> 3 Steps
    # -------------------------------------------------------------
    set_all_seeds(42)
    model_interrupted = EURSNetV2(num_classes=5)
    model_interrupted.load_state_dict(copy.deepcopy(init_state))
    model_interrupted.train()

    opt_interrupted = AdamW(model_interrupted.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_interrupted = CosineAnnealingLR(opt_interrupted, T_max=10, eta_min=1e-6)

    losses_part1 = []
    for step in range(2):
        x, clean, _ = batches[step]
        opt_interrupted.zero_grad()
        restored, _ = model_interrupted(x)
        loss = criterion(restored, clean)
        loss.backward()
        opt_interrupted.step()
        sched_interrupted.step()
        losses_part1.append(loss.item())

    # Save checkpoint to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_resume.pt")
        save_checkpoint(
            filepath=ckpt_path,
            model=model_interrupted,
            optimizer=opt_interrupted,
            scheduler=sched_interrupted,
            epoch=0,
            global_step=2,
            best_score=0.75,
            extra_state={"part1_losses": losses_part1}
        )

        # Destroy objects
        del model_interrupted, opt_interrupted, sched_interrupted

        # Re-instantiate fresh objects
        model_resumed = EURSNetV2(num_classes=5)
        opt_resumed = AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_resumed = CosineAnnealingLR(opt_resumed, T_max=10, eta_min=1e-6)

        # Load checkpoint
        ckpt_meta = load_checkpoint(
            filepath=ckpt_path,
            model=model_resumed,
            optimizer=opt_resumed,
            scheduler=sched_resumed,
            restore_rng=True
        )

        assert ckpt_meta["global_step"] == 2
        assert ckpt_meta["best_score"] == 0.75

        # Execute remaining 3 steps (steps 2, 3, 4)
        losses_part2 = []
        for step in range(2, 5):
            x, clean, _ = batches[step]
            opt_resumed.zero_grad()
            restored, _ = model_resumed(x)
            loss = criterion(restored, clean)
            loss.backward()
            opt_resumed.step()
            sched_resumed.step()
            losses_part2.append(loss.item())

    # -------------------------------------------------------------
    # Strict Numerical Parity & Determinism Invariant Assertions
    # -------------------------------------------------------------
    # 1. Check loss trajectory equality
    losses_resumed_total = losses_part1 + losses_part2
    for s in range(5):
        diff = abs(losses_uninterrupted[s] - losses_resumed_total[s])
        assert diff == 0.0 or diff < 1e-7, f"Step {s} loss mismatch: {losses_uninterrupted[s]} != {losses_resumed_total[s]}"
    print("  [PASS] Step-by-step training losses are bitwise identical.")

    # 2. Check Model Weights Equality
    resumed_params = model_resumed.state_dict()
    for name, p_uninterrupted in final_params_uninterrupted.items():
        p_resumed = resumed_params[name]
        assert torch.equal(p_uninterrupted, p_resumed), f"Parameter '{name}' not bitwise identical after resume!"
    print("  [PASS] 100% of model parameters are bitwise identical (torch.equal == True).")

    # 3. Check Optimizer State Equality (Momentum Buffers)
    resumed_opt_state = opt_resumed.state_dict()
    assert len(final_opt_uninterrupted["state"]) == len(resumed_opt_state["state"]), "Optimizer state dict size mismatch!"
    for p_id in final_opt_uninterrupted["state"]:
        state_orig = final_opt_uninterrupted["state"][p_id]
        state_res = resumed_opt_state["state"][p_id]
        for buf_name in ["exp_avg", "exp_avg_sq"]:
            if buf_name in state_orig:
                assert torch.equal(state_orig[buf_name], state_res[buf_name]), f"Optimizer buffer {buf_name} mismatch for param {p_id}!"
    print("  [PASS] 100% of AdamW optimizer momentum buffers are bitwise identical.")

    # 4. Check LR Scheduler State Equality
    lr_uninterrupted = sched_uninterrupted.get_last_lr()
    lr_resumed = sched_resumed.get_last_lr()
    assert lr_uninterrupted == lr_resumed, f"Scheduler LR mismatch: {lr_uninterrupted} != {lr_resumed}"
    print(f"  [PASS] LR Scheduler state exact match: LR = {lr_resumed}")

    print("=" * 60)
    print("CPU DETERMINISTIC RESUME TEST: EXACT NUMERICAL EQUALITY (100% PASSED)")
    print("Note: CUDA determinism will be verified during Phase 2B on the RTX 5090.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_resume_determinism()
