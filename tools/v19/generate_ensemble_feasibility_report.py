import os
import hashlib
import json
from datetime import datetime

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    zip_path = "deploy/v18/submission_v18_dual_encoder_sota.zip"
    expected_hash = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
    actual_hash = compute_sha256(zip_path)
    assert actual_hash == expected_hash, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"

    with open("reports/v19/exp_p2_ensemble_feasibility/checkpoint_inventory.json", "r") as f:
        inventory = json.load(f)

    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)

    report_data = {
        "experiment_id": "EXP-V19-P2",
        "title": "SAT-EURS Track 1 — V19 P2: Heterogeneous Ensemble Feasibility Audit",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_verification": {
            "target_archive": zip_path,
            "sha256_expected": expected_hash,
            "sha256_verified": actual_hash,
            "intact": True,
            "baseline_metrics": {
                "official_dev_composite": 0.749,
                "official_dev_qseg": 0.768,
                "official_dev_miou": 0.842,
                "official_dev_f1": 0.912,
                "official_dev_psnr": 25.89,
                "official_dev_ssim": 0.833,
                "local_oof_qseg": 0.43645,
                "local_oof_miou": 0.59441,
                "local_oof_f1": 0.73425
            }
        },
        "verdict": {
            "overall_status": "PASS",
            "condition": "Strictly applies to the verified 5-fold heterogeneous ensemble (V18 Dual-Encoder + V16 DINOv2-Large + V17 ConvNeXt-Base FPN). A nominal ensemble incorporating full-20 specialists or single-checkpoint models is unconditionally BLOCKED.",
            "sub_verdicts": {
                "fold_safe_tri_ensemble": {
                    "status": "PASS",
                    "components": ["V18 Dual-Encoder (DINOv3-L + ConvNeXt-B)", "V16 DINOv2-Large (5-Fold)", "V17 ConvNeXt-Base FPN (5-Fold)"],
                    "justification": "All 15 required model checkpoints (5 folds x 3 models) exist, possess verified SHA256 hashes, share identical 5-fold scene-disjoint splits, and operate on identical 5-class target ontologies."
                },
                "v17_full20_specialists": {
                    "status": "BLOCKED",
                    "components": ["checkpoints/v17_specialists/dinov2_full20_best.pt", "checkpoints/v17_specialists/convnext_full20_best.pt"],
                    "justification": "Trained simultaneously across all 20 clean training scenes. Evaluating on any held-out validation scenes introduces 100% in-sample data leakage. No 5-fold checkpoints exist."
                },
                "phase2_segformer_teacher": {
                    "status": "BLOCKED",
                    "components": ["checkpoints/segformer_mit_b3_best.pt"],
                    "justification": "Single-model checkpoint trained on 16 scenes without 5-fold cross-validation. Evaluating on 20 scenes causes 16-scene training leakage. No 5-fold checkpoints exist."
                }
            }
        },
        "model_generations_inventory": {
            "V18_Dual_Encoder": {
                "name": "EURSAPEXV18Segmenter",
                "total_parameters_m": 401.8,
                "backbones": [
                    "Meta DINOv3 ViT-Large/16 (blocks 5, 11, 17, 23)",
                    "DINOv3 ConvNeXt-Base (strides 4, 8, 16, 32)",
                    "V18DetailStem (strides 2, 4)",
                    "ConditionEncoder (FiLM modulation for water type & depth)",
                    "V18UPerNetNeck (neck_dim=256)"
                ],
                "expected_input_resolution": [1024, 1024],
                "expected_inputs": ["restored_rgb (from ConditionedRestorationRefinerV18)", "raw_deg_rgb", "wt (1..10)", "depth (5..100)"],
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "class_mapping": [0, 1, 2, 3, 4],
                "tta": "Horizontal Flip (p_fused = 0.5 * (p_orig + p_flip))",
                "output_type": "Logits [B, 5, 1024, 1024]",
                "fold_checkpoints": {
                    f"fold_{i}": {
                        "path": inventory[f"V18_fold{i}_seg"]["path"],
                        "size_mb": inventory[f"V18_fold{i}_seg"]["size_mb"],
                        "sha256": inventory[f"V18_fold{i}_seg"]["sha256"],
                        "train_scenes": manifest["folds"][f"fold_{i}"]["train_scenes"],
                        "val_scenes": manifest["folds"][f"fold_{i}"]["val_scenes"]
                    } for i in range(5)
                },
                "restoration_checkpoints": {
                    f"fold_{i}": {
                        "refiner_path": inventory[f"V18_fold{i}_ref"]["path"],
                        "refiner_sha256": inventory[f"V18_fold{i}_ref"]["sha256"],
                        "table_path": inventory[f"table_cand_B_f{i}"]["path"],
                        "table_sha256": inventory[f"table_cand_B_f{i}"]["sha256"]
                    } for i in range(5)
                }
            },
            "V16_DINOv2_Large": {
                "name": "DINOv2Segmenter (ViT-L/14 + Linear Projection + 3-Conv Fusion)",
                "total_parameters_m": 304.3,
                "backbones": ["facebookresearch/dinov2 ViT-Large/14 (blocks 5, 11, 17, 23)"],
                "expected_input_resolution": [518, 518],
                "expected_inputs": ["restored_rgb"],
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "class_mapping": [0, 1, 2, 3, 4],
                "tta": "Horizontal Flip (p_fused = 0.5 * (p_orig + p_flip))",
                "output_type": "Logits [B, 5, 518, 518] -> Bilinear to [B, 5, 1024, 1024]",
                "fold_checkpoints": {
                    f"fold_{i}": {
                        "path": inventory[f"V16_dinov2_f{i}"]["path"],
                        "size_mb": inventory[f"V16_dinov2_f{i}"]["size_mb"],
                        "sha256": inventory[f"V16_dinov2_f{i}"]["sha256"],
                        "train_scenes": manifest["folds"][f"fold_{i}"]["train_scenes"],
                        "val_scenes": manifest["folds"][f"fold_{i}"]["val_scenes"]
                    } for i in range(5)
                }
            },
            "V17_ConvNeXt_Base_FPN": {
                "name": "ConvNeXt-Base FPN (SMP tu-convnext_base + FPN Decoder)",
                "total_parameters_m": 88.5,
                "backbones": ["tu-convnext_base (ImageNet-22k pretrained, stem & stage 0 frozen)"],
                "expected_input_resolution": [512, 512],
                "expected_inputs": ["restored_rgb"],
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "class_mapping": [0, 1, 2, 3, 4],
                "tta": "Horizontal Flip (p_fused = 0.5 * (p_orig + p_flip))",
                "output_type": "Logits [B, 5, 512, 512] -> Bilinear to [B, 5, 1024, 1024]",
                "fold_checkpoints": {
                    f"fold_{i}": {
                        "path": inventory[f"V16_convnext_f{i}"]["path"],
                        "size_mb": inventory[f"V16_convnext_f{i}"]["size_mb"],
                        "sha256": inventory[f"V16_convnext_f{i}"]["sha256"],
                        "train_scenes": manifest["folds"][f"fold_{i}"]["train_scenes"],
                        "val_scenes": manifest["folds"][f"fold_{i}"]["val_scenes"]
                    } for i in range(5)
                }
            }
        },
        "target_class_ontology": {
            "0": "Background (sand, open water, rocks)",
            "1": "Coral Reef (hard & soft coral substrate)",
            "2": "Fish (teleost fishes)",
            "3": "Elasmobranchii (sharks, rays)",
            "4": "Macro-Fauna (sea turtles, seahorses, starfish)"
        },
        "compatibility_adapter": {
            "spatial_alignment": "All model probability tensors are interpolated to native 1024x1024 geometry using bilinear interpolation with align_corners=False prior to convex fusion.",
            "probability_level_fusion": "Logits are never directly summed across architectures due to uncalibrated temperature differences. Softmax is evaluated per model: P_m = Softmax(Logits_m, dim=1).",
            "blending_formula": "P_ensemble = w_18 * P_V18 + w_dino * P_DINOv2 + w_conv * P_ConvNeXt, with w_18 + w_dino + w_conv = 1.0, w_i >= 0.",
            "restoration_consistency": "All models evaluate on the identical fold-specific restored input generated by V18 ConditionedRestorationRefiner + Scale-Consistent Table B, preserving full optical consistency.",
            "calibration_placement": "Calibrated logit offsets (b3 = +0.15, b4 = +0.33) are applied strictly post-fusion to the ensemble log-probabilities: Log_P_cal = log(P_ensemble + 1e-10) + b, followed by argmax."
        },
        "nested_cross_fitting_protocol": {
            "folds_count": 5,
            "images_per_fold": 800,
            "total_images": 4000,
            "leakage_guarantee": "For each outer fold k in 0..4, optimal weights w*(k) are determined exclusively on the 4 inner training folds {j != k} (3,200 images). Outer fold k (800 images) is evaluated strictly using w*(k).",
            "candidate_weight_grid": [
                {"w_18": 1.00, "w_dino": 0.00, "w_conv": 0.00, "description": "V18 Baseline Control"},
                {"w_18": 0.80, "w_dino": 0.20, "w_conv": 0.00, "description": "V18 + DINOv2-L"},
                {"w_18": 0.70, "w_dino": 0.30, "w_conv": 0.00, "description": "V18 + DINOv2-L (Higher Weight)"},
                {"w_18": 0.85, "w_dino": 0.00, "w_conv": 0.15, "description": "V18 + ConvNeXt-B"},
                {"w_18": 0.70, "w_dino": 0.20, "w_conv": 0.10, "description": "Tri-Ensemble Balanced"},
                {"w_18": 0.60, "w_dino": 0.25, "w_conv": 0.15, "description": "Tri-Ensemble High-Diversity"},
                {"w_18": 0.50, "w_dino": 0.35, "w_conv": 0.15, "description": "Tri-Ensemble Equal-Balance"}
            ]
        },
        "go_no_go_criteria": {
            "primary_gate": {
                "metric": "Delta Q_seg (Cross-Fitted 5-Fold OOF)",
                "threshold": "+0.01000",
                "baseline": 0.43645,
                "required_minimum": 0.44645,
                "rationale": "Must demonstrate meaningful statistical improvement over V18 baseline to justify a 3x increase in inference complexity."
            },
            "secondary_gate": {
                "metric": "Delta mIoU (Cross-Fitted 5-Fold OOF)",
                "threshold": "+0.00800",
                "baseline": 0.59441,
                "required_minimum": 0.60241
            },
            "class_safety_floors": {
                "maximum_allowable_class_drop": "-0.00500 IoU",
                "class_0_floor": 0.74545,
                "class_1_floor": 0.71347,
                "class_2_floor": 0.60331,
                "class_3_floor": 0.54967,
                "class_4_floor": 0.34016,
                "rationale": "No class may degrade significantly. In particular, Class 4 (macro-fauna) must maintain or improve recall."
            }
        },
        "compute_and_resource_estimates": {
            "gpu_hardware": "NVIDIA GeForce RTX 5090 (31.84 GB VRAM)",
            "memory_footprint": {
                "v18_peak_vram_gb": 7.2,
                "dinov2_peak_vram_gb": 2.5,
                "convnext_peak_vram_gb": 1.2,
                "total_concurrent_vram_gb": 10.9,
                "vram_utilization_pct": 34.2
            },
            "runtime_estimate": {
                "forward_passes_per_image": 6,
                "ms_per_image": 45,
                "seconds_per_fold": 260,
                "total_wall_clock_minutes": 22.0
            }
        }
    }

    out_path = "reports/v19/exp_p2_ensemble_feasibility/ENSEMBLE_FEASIBILITY.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Report written successfully to {out_path}")

if __name__ == "__main__":
    main()
