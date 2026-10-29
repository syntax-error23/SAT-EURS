"""
SAT-EURS — Official Candidate & Submission Manifest
Strict machine-readable ledger of all Track 1 and Track 2 candidates, official submissions,
provenance, hashes, and verification statuses.
"""

import json
import os
import hashlib

def get_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "summary": {
        "best_historical_score_track1": {
            "score": 0.6940,
            "submission_id": "18747",
            "phase": "Development",
            "method": "EURS-Apex-V16-SOTA"
        },
        "best_historical_score_track2": {
            "score": 0.3380,
            "submission_id": "18699",
            "phase": "Development",
            "method": "EURS-Net-V5-Calib"
        },
        "best_testing_track1": {
            "score": 0.6340,
            "submission_id": "19096",
            "phase": "Testing",
            "method": "V28-A Five-Fold Ablation",
            "artifact_path": "deploy/v28/submission_v28a_fivefold_ablation.zip",
            "backup_path": "deploy/protected_reference/submission_v28a_fivefold_ablation.zip",
            "sha256": "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"
        },
        "best_testing_track2": {
            "score": 0.2530,
            "submission_id": "19130",
            "phase": "Testing",
            "method": "EURS-LiteV3-Full20",
            "artifact_path": "deploy/eurs_litev3_production.onnx",
            "backup_path": "deploy/protected_reference/eurs_litev3_production.onnx",
            "sha256": "b39efc20918927ebf073ae46c4fcf51e83ee658e0f957f60fe32f753df93a5d5"
        },
        "best_research_reference_track2": {
            "score": 0.3030,
            "submission_id": "18751",
            "phase": "Development (Research Reference)",
            "method": "EXP-COMBINED-BEST",
            "artifact_path": "deploy/EXP-COMBINED-BEST.onnx",
            "backup_path": "deploy/protected_reference/EXP-COMBINED-BEST_18751.onnx",
            "sha256": "645eb785ba2bfb07a9ce4349b19b79151af627a88c58b466035395e48cbbcd52"
        }
    },
    "candidates": [
        {
            "track": 1,
            "candidate_id": "T1-V28A-CHAMPION",
            "human_readable_name": "V28-A Five-Fold Ablation (Champion Reference)",
            "submission_id": "19096",
            "official_phase": "Testing",
            "artifact_path": "deploy/v28/submission_v28a_fivefold_ablation.zip",
            "sha256": "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39",
            "official_composite": 0.6340,
            "official_metrics": {
                "Q_seg": 0.5702,
                "mIoU": 0.7040,
                "Macro_F1": 0.8100,
                "PSNR": 25.48,
                "SSIM": 0.8290,
                "Sharpness": 0.9730,
                "UIQM": 0.644,
                "UCIQE": 0.592,
                "CCF": 0.090
            },
            "checkpoint_provenance": {
                "fold_0": "checkpoints/v27/segmenter_v27_fold0_best.pt",
                "fold_1": "checkpoints/v18/fold_1/segmenter_best.pt",
                "fold_2": "checkpoints/v18/fold_2/segmenter_best.pt",
                "fold_3": "checkpoints/v18/fold_3/segmenter_best.pt",
                "fold_4": "checkpoints/v18/fold_4/segmenter_best.pt",
                "refiners": "checkpoints/v18/fold_{k}/refiner_best.pt + tbl19_20sc_scale_consistent.pt"
            },
            "offsets": [0.0, 0.0, 0.0, 0.10, 0.25],
            "tta": "Horizontal-Flip TTA",
            "submission_status": "SCORED",
            "is_highest_eligible_testing": True,
            "claim_type": "Verified from official leaderboard"
        },
        {
            "track": 1,
            "candidate_id": "T1-V32-APEX",
            "human_readable_name": "V32-Apex-Candidate",
            "submission_id": "19333",
            "official_phase": "Testing",
            "artifact_path": "deploy/v32/submission_v32_apex_candidate.zip",
            "sha256": "156bcff73f5e32fcbe5ccf333b75523b048ba6465daa782c35190205089be6e4",
            "official_composite": 0.6170,
            "official_metrics": {
                "Q_seg": 0.5410,
                "mIoU": 0.6810,
                "Macro_F1": 0.7940,
                "PSNR": 25.48,
                "SSIM": 0.8290,
                "Sharpness": 0.9730,
                "UIQM": 0.644,
                "UCIQE": 0.592,
                "CCF": 0.090
            },
            "checkpoint_provenance": {
                "fold_0": "checkpoints/v28/segmenter_v28_fold0_best.pt",
                "fold_1": "checkpoints/v18/fold_1/segmenter_best.pt",
                "fold_2": "checkpoints/v18/fold_2/segmenter_best.pt",
                "fold_3": "checkpoints/v18/fold_3/segmenter_best.pt",
                "fold_4": "checkpoints/v18/fold_4/segmenter_best.pt",
                "refiners": "Verified V28-A raw restorations (exact 25.48 dB parity)"
            },
            "offsets": [0.0, 0.0, 0.0, 0.10, 0.30],
            "tta": "Horizontal-Flip TTA",
            "submission_status": "SCORED",
            "is_highest_eligible_testing": False,
            "claim_type": "Verified from official leaderboard"
        },
        {
            "track": 2,
            "candidate_id": "T2-EXP18751-REF",
            "human_readable_name": "EXP-COMBINED-BEST (#18751 Dev Reference)",
            "submission_id": "18751",
            "official_phase": "Development (Research Reference)",
            "artifact_path": "deploy/EXP-COMBINED-BEST.onnx",
            "sha256": "645eb785ba2bfb07a9ce4349b19b79151af627a88c58b466035395e48cbbcd52",
            "official_composite": 0.3030,
            "official_metrics": {
                "Q_seg": 0.1630,
                "mIoU": 0.3480,
                "Macro_F1": 0.4670,
                "PSNR": 16.75,
                "SSIM": 0.7650,
                "GFLOPs": 3.5670,
                "Parameters_M": 0.9589
            },
            "specs": {
                "input_shape": [1, 3, 1024, 1024],
                "outputs": ["restored_image", "seg_logits"],
                "gflops": 3.5666,
                "params_m": 0.9589,
                "caps_passed": True
            },
            "submission_status": "SCORED_IN_DEV",
            "is_highest_eligible_testing": False,
            "claim_type": "Verified from official leaderboard & repo artifact"
        },
        {
            "track": 2,
            "candidate_id": "T2-LITEV3-PROD",
            "human_readable_name": "EURS-LiteV3-Full20 (#19130 Current Testing Best)",
            "submission_id": "19130",
            "official_phase": "Testing",
            "artifact_path": "deploy/eurs_litev3_production.onnx",
            "sha256": "b39efc20918927ebf073ae46c4fcf51e83ee658e0f957f60fe32f753df93a5d5",
            "official_composite": 0.2530,
            "official_metrics": {
                "Q_seg": 0.0980,
                "mIoU": 0.2620,
                "Macro_F1": 0.3750,
                "PSNR": 18.44,
                "SSIM": 0.7180,
                "GFLOPs": 1.6380,
                "Parameters_M": 0.4859
            },
            "specs": {
                "input_shape": [1, 3, 384, 768],
                "outputs": ["restored_image", "seg_logits"],
                "gflops": 1.6382,
                "params_m": 0.4859,
                "caps_passed": True
            },
            "submission_status": "SCORED",
            "is_highest_eligible_testing": True,
            "claim_type": "Verified from official leaderboard"
        },
        {
            "track": 2,
            "candidate_id": "T2-NETV5-CALIB",
            "human_readable_name": "EURSNet-V5-Calibrated (#19332)",
            "submission_id": "19332",
            "official_phase": "Testing",
            "artifact_path": "deploy/eurs_net_v5_calibrated_track2.onnx",
            "sha256": "ad7bb1b1083083aee9fbbe44ce8974089d6b93077d892c20df84a267dd5fd135",
            "official_composite": 0.2470,
            "official_metrics": {
                "Q_seg": 0.0830,
                "mIoU": 0.2430,
                "Macro_F1": 0.3440,
                "PSNR": 17.69,
                "SSIM": 0.7500,
                "GFLOPs": 9.0500,
                "Parameters_M": 0.7683
            },
            "specs": {
                "input_shape": [1, 3, 1024, 1024],
                "outputs": ["restored_image", "seg_logits"],
                "gflops": 9.0500,
                "params_m": 0.7683,
                "caps_passed": True
            },
            "submission_status": "SCORED",
            "is_highest_eligible_testing": False,
            "claim_type": "Verified from official leaderboard"
        }
    ]
}

os.makedirs("reports", exist_ok=True)
manifest_path = "reports/candidate_and_submission_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Candidate and submission manifest written to {manifest_path}")
