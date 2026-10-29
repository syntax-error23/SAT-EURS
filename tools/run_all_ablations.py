"""
SAT-EURS Track 2 — Full Ablation Matrix Runner
Sequentially trains and pools out-of-sample evaluation for:
1. E1: Supervised baseline (Weighted CE + L1 Restoration)
2. E2: Standard Distillation (WCE + T=2.5 KL against cached DINOv2 targets)
3. E3: Class-Aware Distillation (WCE + Rare-Class-Weighted Distillation)
4. E4: Boundary-Band Loss Suite (SegmentationV7Loss + Distillation)

Saves summary results to:
- reports/track2_readiness/TRACK2_ABLATION_RESULTS.csv
- reports/track2_readiness/TRACK2_ABLATION_SUMMARY.json
"""

import os
import sys
import time
import json
import csv
import subprocess

experiments = [
    ("E1_supervised", "Supervised Baseline (WCE + Rest L1)"),
    ("E2_distill_kl", "Standard Distillation (WCE + KL Div T=2.5 + Rest L1)"),
    ("E3_class_aware_kd", "Class-Aware Distillation (WCE + Rare-Class Weighted KD)"),
    ("E4_boundary_v7", "Boundary-Band Suite (V7 Loss: WCE+Lovasz+SoftF1+Boundary + KD)")
]

epochs = 20
batch_size = 16
lr = 5e-4

summary_records = []

for exp_name, exp_desc in experiments:
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {exp_name} ({exp_desc})")
    print(f"{'='*70}\n")
    
    # 1. Train Fold 0
    ckpt_f0 = f"checkpoints/litev3/{exp_name}_f0/best_model.pt"
    if not os.path.exists(ckpt_f0):
        cmd_f0 = [
            sys.executable, "tools/train_litev3.py",
            "--exp", exp_name,
            "--fold", "0",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr)
        ]
        print(f"[Command] {' '.join(cmd_f0)}", flush=True)
        ret0 = subprocess.run(cmd_f0)
        if ret0.returncode != 0:
            print(f"[Error] Failed training {exp_name}_f0!")
            sys.exit(1)
    else:
        print(f"[Skip] Found existing {ckpt_f0}")
        
    # 2. Train Fold 1
    ckpt_f1 = f"checkpoints/litev3/{exp_name}_f1/best_model.pt"
    if not os.path.exists(ckpt_f1):
        cmd_f1 = [
            sys.executable, "tools/train_litev3.py",
            "--exp", exp_name,
            "--fold", "1",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr)
        ]
        print(f"[Command] {' '.join(cmd_f1)}", flush=True)
        ret1 = subprocess.run(cmd_f1)
        if ret1.returncode != 0:
            print(f"[Error] Failed training {exp_name}_f1!")
            sys.exit(1)
    else:
        print(f"[Skip] Found existing {ckpt_f1}")
        
    # 3. Pooled Out-of-Sample Evaluation
    eval_json = f"reports/track2_readiness/{exp_name}_eval.json"
    cmd_eval = [
        sys.executable, "tools/evaluate_litev3_pooled.py",
        "--ckpt_f0", ckpt_f0,
        "--ckpt_f1", ckpt_f1,
        "--exp_name", exp_name,
        "--out_json", eval_json
    ]
    print(f"[Command] {' '.join(cmd_eval)}", flush=True)
    ret_eval = subprocess.run(cmd_eval)
    if ret_eval.returncode != 0:
        print(f"[Error] Failed evaluating {exp_name}!")
        sys.exit(1)
        
    with open(eval_json) as fp:
        res = json.load(fp)
        res["description"] = exp_desc
        summary_records.append(res)
        
    print(f"\n[Completed {exp_name}] S_local_90 = {res['s_local_90']:.4f}, Q_seg = {res['q_seg']:.4f}, Q_FR = {res['q_fr']:.4f}\n")

# Save summary JSON
with open("reports/track2_readiness/TRACK2_ABLATION_SUMMARY.json", "w") as fp:
    json.dump(summary_records, fp, indent=2)

# Save summary CSV
csv_path = "reports/track2_readiness/TRACK2_ABLATION_RESULTS.csv"
fieldnames = [
    "Experiment", "Description", "S_local_90", "Q_seg", "Macro_F1", "Q_FR",
    "PSNR_dB", "SSIM", "Sharpness", "P_cov",
    "IoU_C0", "IoU_C1", "IoU_C2", "IoU_C3", "IoU_C4",
    "Rec_C0", "Rec_C1", "Rec_C2", "Rec_C3", "Rec_C4"
]

with open(csv_path, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for r in summary_records:
        writer.writerow({
            "Experiment": r["exp_name"],
            "Description": r["description"],
            "S_local_90": r["s_local_90"],
            "Q_seg": r["q_seg"],
            "Macro_F1": r["macro_f1"],
            "Q_FR": r["q_fr"],
            "PSNR_dB": r["psnr"],
            "SSIM": r["ssim"],
            "Sharpness": r["sharpness"],
            "P_cov": r["p_cov"],
            "IoU_C0": r["ious"][0],
            "IoU_C1": r["ious"][1],
            "IoU_C2": r["ious"][2],
            "IoU_C3": r["ious"][3],
            "IoU_C4": r["ious"][4],
            "Rec_C0": r["recalls"][0],
            "Rec_C1": r["recalls"][1],
            "Rec_C2": r["recalls"][2],
            "Rec_C3": r["recalls"][3],
            "Rec_C4": r["recalls"][4],
        })

print(f"\nSaved comprehensive ablation results to {csv_path} and reports/track2_readiness/TRACK2_ABLATION_SUMMARY.json!\n")
