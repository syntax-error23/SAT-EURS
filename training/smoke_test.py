"""
Tiny Smoke Test for EURS-Net V2 Training Harness.

Runs 1-2 batches only on:
1. Candidate A (768 Direct Resize)
2. Candidate C (768 Aspect-Preserving Letterbox)

Verifies:
- forward pass
- multitask loss computation with valid_mask handling
- backward pass
- optimizer step
- checkpoint save
- checkpoint reload and state verification
- validation evaluation
"""

import os
import shutil
import tempfile
import torch

from training.config import get_candidate_training_config
from training.trainer import EURSTrainer


def run_candidate_smoke_test(candidate_id: str, temp_dir: str):
    print(f"\n=======================================================")
    print(f"RUNNING SMOKE TEST: Candidate {candidate_id}")
    print(f"=======================================================")

    cfg = get_candidate_training_config(
        candidate_id,
        output_dir=temp_dir,
        epochs=1,
        device="cpu",
        num_workers=0,
        batch_size=2
    )

    trainer = EURSTrainer(cfg)

    # 1. Forward -> Loss -> Backward -> Optimizer Step
    print(f"[{candidate_id}] Step 1: Training 2 batches...")
    train_metrics = trainer.train_epoch(epoch=1, max_batches=2)
    print(f"[{candidate_id}] Train metrics: {train_metrics}")
    assert train_metrics["train_loss_total"] > 0.0, "Total loss must be positive!"

    # 2. Validation evaluation (1 batch)
    print(f"[{candidate_id}] Step 2: Validating 1 batch...")
    val_metrics = trainer.validate_epoch(epoch=1, max_batches=1)
    print(f"[{candidate_id}] Validation metrics: {val_metrics}")
    assert val_metrics["psnr"] > 0.0, "PSNR must be positive!"
    assert val_metrics["macro_iou"] >= 0.0, "Macro IoU must be >= 0.0!"

    # 3. Checkpoint save
    print(f"[{candidate_id}] Step 3: Saving checkpoint...")
    trainer.save_checkpoint(epoch=1, is_best=True, metrics=val_metrics)
    best_ckpt = os.path.join(temp_dir, cfg.experiment_name, "checkpoint_best.pth")
    assert os.path.isfile(best_ckpt), f"Missing checkpoint at {best_ckpt}"

    # 4. Checkpoint reload
    print(f"[{candidate_id}] Step 4: Reloading checkpoint into fresh trainer...")
    fresh_trainer = EURSTrainer(cfg)
    loaded = fresh_trainer.load_checkpoint(best_ckpt)
    assert loaded["epoch"] == 1
    assert loaded["seed"] == 42

    print(f"[{candidate_id}] SMOKE TEST PASSED SUCCESSFULLY!")
    return {
        "candidate": candidate_id,
        "train_loss": train_metrics["train_loss_total"],
        "val_loss": val_metrics["val_loss_total"],
        "psnr": val_metrics["psnr"],
        "macro_iou": val_metrics["macro_iou"]
    }


def main():
    temp_dir = tempfile.mkdtemp(prefix="eurs_smoke_test_")
    try:
        # Test Direct mode (Candidate A)
        res_a = run_candidate_smoke_test("A", temp_dir)

        # Test Letterbox mode (Candidate C)
        res_c = run_candidate_smoke_test("C", temp_dir)

        print("\n" + "=" * 55)
        print("ALL SMOKE TESTS PASSED!")
        print("Candidate A (Direct):", res_a)
        print("Candidate C (Letterbox):", res_c)
        print("=" * 55)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
