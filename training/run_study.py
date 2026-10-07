"""
Orchestration and comparative analysis for the EURS Resolution Study (Candidates A, B, C, D).

Provides:
- Comparative analysis across candidates:
    * 768 direct vs 1024 direct
    * 768 letterbox vs 1024 letterbox
    * direct vs letterbox at 768
    * direct vs letterbox at 1024
- Generation of the unified experiments/resolution_study/summary.json
- CLI entrypoint for individual or full-sequence execution.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

from training.config import get_candidate_training_config, get_all_candidate_configs, TrainingConfig
from training.trainer import EURSTrainer


def compute_relative_differences(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes relative metric comparisons between:
      1. 768 direct (A) vs 1024 direct (B)
      2. 768 letterbox (C) vs 1024 letterbox (D)
      3. direct vs letterbox at 768 (A vs C)
      4. direct vs letterbox at 1024 (B vs D)
    """
    comparisons = {}

    def _diff(name_1: str, name_2: str, key1: str, key2: str):
        if key1 not in summaries or key2 not in summaries:
            return None

        s1 = summaries[key1]
        s2 = summaries[key2]

        comparison_metrics = [
            "psnr", "ssim", "sharpness_ratio", "pixel_accuracy",
            "macro_iou", "iou_class_0", "iou_class_1", "iou_class_2",
            "iou_class_3", "iou_class_4", "composite_score", "profiled_gflops"
        ]

        diff_dict = {
            "baseline": name_1,
            "comparison": name_2,
            "metrics": {}
        }

        for m in comparison_metrics:
            val1 = s1.get(m, 0.0)
            val2 = s2.get(m, 0.0)
            delta = val2 - val1
            rel_pct = (delta / val1 * 100.0) if val1 != 0 else 0.0
            diff_dict["metrics"][m] = {
                f"{key1}_value": round(val1, 4),
                f"{key2}_value": round(val2, 4),
                "delta": round(delta, 4),
                "relative_pct": round(rel_pct, 2)
            }

        return diff_dict

    comparisons["768_direct_vs_1024_direct"] = _diff("A_768_direct", "B_1024_direct", "A", "B")
    comparisons["768_letterbox_vs_1024_letterbox"] = _diff("C_768_letterbox", "D_1024_letterbox", "C", "D")
    comparisons["768_direct_vs_768_letterbox"] = _diff("A_768_direct", "C_768_letterbox", "A", "C")
    comparisons["1024_direct_vs_1024_letterbox"] = _diff("B_1024_direct", "D_1024_letterbox", "B", "D")

    return comparisons


def generate_resolution_study_summary(
    candidate_summaries: Dict[str, Dict[str, Any]],
    output_path: str = "experiments/resolution_study/summary.json"
) -> Dict[str, Any]:
    """Generates the master machine-readable resolution study summary."""
    comparisons = compute_relative_differences(candidate_summaries)

    study_summary = {
        "metadata": {
            "study_name": "EURS-Net V2 Preprocessing Resolution Study",
            "candidates": ["A", "B", "C", "D"],
            "constraints": {
                "max_parameters": 1000000,
                "max_gflops": 12.0
            }
        },
        "candidates": candidate_summaries,
        "comparative_analysis": comparisons
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=2)

    return study_summary


def main():
    parser = argparse.ArgumentParser(description="EURS-Net V2 Resolution Study Runner")
    parser.add_argument(
        "--candidate",
        type=str,
        choices=["A", "B", "C", "D", "ALL"],
        default="A",
        help="Candidate experiment to run ('A', 'B', 'C', 'D', or 'ALL')"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test (1-2 batches per epoch, 1 epoch) without full dataset training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cuda' or 'cpu')"
    )
    args = parser.parse_args()

    candidate_ids = ["A", "B", "C", "D"] if args.candidate == "ALL" else [args.candidate]
    summaries = {}

    for cid in candidate_ids:
        overrides = {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.device is not None:
            overrides["device"] = args.device

        cfg = get_candidate_training_config(cid, **overrides)
        print(f"\n=======================================================")
        print(f"Starting Candidate {cid}: {cfg.experiment_name}")
        print(f"Resolution: {cfg.resolution}x{cfg.resolution} | Mode: {cfg.mode}")
        print(f"=======================================================")

        trainer = EURSTrainer(cfg)
        if args.smoke_test:
            print(f"[SMOKE TEST] Running 2 training batches and 1 validation batch...")
            train_metrics = trainer.train_epoch(1, max_batches=2)
            val_metrics = trainer.validate_epoch(1, max_batches=1)
            print(f"[SMOKE TEST COMPLETE] Train loss: {train_metrics['train_loss_total']}, Val loss: {val_metrics['val_loss_total']}")
        else:
            summary = trainer.run_training()
            summaries[cid] = summary
            print(f"Candidate {cid} completed. Summary: {summary}")

    if not args.smoke_test and len(summaries) > 0:
        summary_path = os.path.join("experiments", "resolution_study", "summary.json")
        generate_resolution_study_summary(summaries, summary_path)
        print(f"\nUnified study summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
