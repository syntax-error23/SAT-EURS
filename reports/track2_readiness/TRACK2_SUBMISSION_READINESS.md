# SAT-EURS Track 2 — Submission Readiness Dossier & Technical Handoff

**Document Version:** 1.0.0  
**Verification Date:** 2026-10-24T04:11:00+05:30  
**Lead Authors:** Senior Computer Vision Researcher & Model Optimization Engineer  
**Competition:** MaCVi @ WACV 2027 Challenge — SAT-EURS Track 2 (Embedded Edge Deployment on Jetson AGX Orin)  
**Final Submission Deadline:** October 27, 2026, 23:59 AoE

---

## 1. Executive Summary & Recommendation

We have executed the full pre-architecture audit, root-cause diagnostics, model implementation, and empirical ablation matrix for **SAT-EURS Track 2**. 

Our primary production candidate, **`SAT-EURS-LiteV3`**, achieves:
- **Local Composite Score ($S_{\text{local\_90}}$):** **0.3956**
- **Verified Improvement over Baseline Fallback (#18699, 0.3380):** **$+0.0576$ points ($+17.0\%$ relative improvement)**
- **Total Parameters:** **0.4859 M** (Official Limit: $\le 1.0\text{ M}$ $\implies$ **51.4% Headroom**)
- **Compute Complexity:** **1.6275 GFLOPs** (Official Limit: $\le 12.0\text{ GFLOPs}$ $\implies$ **86.4% Headroom**)
- **Numerical Parity (PyTorch vs ONNX Runtime):** Max restoration error $2.35 \times 10^{-6}$, max logits error $4.24 \times 10^{-5}$
- **All 8 Official Verification Gates:** **STRICTLY PASSED**

```
========================================================================================
RECOMMENDATION:
1. Primary Scored Submission: deploy/eurs_litev3_track2_submission.zip
2. Insurance Fallback Preserved: deploy/eurs_net_v5_calibrated_track2.zip (Score 0.3380)
3. V26 Track 1 Baseline Preserved: deploy/v26/submission_v26_test_apex_sota.zip (Unmodified)
========================================================================================
```

---

## 2. Official 8-Gate Verification Matrix

| Gate | Criterion | Official Limit | LiteV3 Measured | Margin / Headroom | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Gate 1** | Parameter Budget | $\le 1,000,000$ (1.0 M) | **485,893** (0.4859 M) | **+51.4% Headroom** | **PASSED** |
| **Gate 2** | Compute Complexity | $\le 12.0\text{ GFLOPs}$ | **1.6275 GFLOPs** | **+86.4% Headroom** | **PASSED** |
| **Gate 3** | Operator Whitelist | Strict `ALWAYS_ALLOWED` | 0 non-whitelisted nodes | Standard Ops (Conv, DW, Relu, Softmax) | **PASSED** |
| **Gate 4** | Input Contract | Static $[1, 3, 384, 768]$ float32 | $[1, 3, 384, 768]$ in $[0, 1]$ | Exact Tensor Shape | **PASSED** |
| **Gate 5** | Output Contract | Restored $[1, 3, H, W]$ + Logits $[1, 5, H, W]$ | Restored in $[0, 1]$, 5 raw logits | Valid Dual Tensors | **PASSED** |
| **Gate 6** | Numerical Parity | PyTorch CPU vs ORT CPU $\le 10^{-4}$ | Restored: $2.35 \times 10^{-6}$, Logits: $4.24 \times 10^{-5}$ | Exact Parity ($<10^{-4}$) | **PASSED** |
| **Gate 7** | Prediction Coverage | $P_{\text{cov}} = 1.0$ (Non-zero recall all 5 classes) | Predicted count $> 0$ for all classes | Class 3 & 4 non-zero | **PASSED** |
| **Gate 8** | Baseline Superiority | Exceeds Fallback #18699 ($0.3380$) | **0.3956** ($+0.0576$ gain) | **+17.0% Score Gain** | **PASSED** |

---

## 3. Empirical Ablation Results & Comparative Analysis

All experiments were trained under identical conditions on NVIDIA RTX 5090 (20 epochs, AdamW $\text{lr}=5\times 10^{-4}$ with Cosine Annealing, static resolution $384\times 768$) and evaluated strictly out-of-sample across the 8 held-out scenes (320 images) pooling Fold 0 and Fold 1.

### 3.1 Aggregated Ablation Results (`reports/track2_readiness/TRACK2_ABLATION_RESULTS.csv`)

| Experiment ID | Configuration Description | Composite $S_{\text{local\_90}}$ | Seg $Q_{\text{seg}}$ | Macro-F1 | Restor $Q_{\text{FR}}$ | PSNR (dB) | SSIM | Sharpness | Coverage $P_{\text{cov}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fallback** | EURS-Net V5 Calibrated (#18699) | 0.3380 | 0.2800 | — | 0.4540 | 17.80 | 0.6120 | 0.7200 | 1.0 |
| **E1** | **Supervised Baseline (WCE + Rest L1)** | **0.3956** | **0.3955** | **0.5196** | **0.5274** | **18.92** | **0.6841** | 0.8717 | **1.0** |
| **E2** | Standard Distillation ($T=2.5$ KL + Rest L1) | 0.3592 | 0.3350 | 0.4202 | 0.5272 | 18.89 | 0.6794 | **0.8824** | 0.0 |
| **E3** | Class-Aware Distillation (Rare-Weighted KD) | 0.3728 | 0.3592 | 0.4829 | 0.5243 | 18.79 | 0.6777 | 0.8789 | **1.0** |
| **E4** | Boundary-Band Loss Suite (V7 Loss + KD) | 0.3631 | 0.3417 | 0.4656 | 0.5271 | 18.91 | 0.6794 | 0.8806 | **1.0** |

### 3.2 Per-Class Out-of-Sample IoU & Recall Breakdown

| Exp | Class 0 (Bg) IoU / Rec | Class 1 (Fish) IoU / Rec | Class 2 (Starfish) IoU / Rec | Class 3 (Shark) IoU / Rec | Class 4 (Seahorse) IoU / Rec |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | 0.6817 / 0.7921 | 0.6176 / 0.8124 | **0.4084 / 0.6781** | **0.2425 / 0.2652** | **0.0275 / 0.0301** |
| **E2** | **0.7182 / 0.8565** | **0.6355 / 0.7802** | 0.3198 / 0.5531 | 0.0000 / 0.0000 | 0.0016 / 0.0017 |
| **E3** | 0.6234 / 0.6980 | 0.5710 / 0.8355 | 0.3759 / 0.6809 | 0.2178 / 0.2338 | 0.0077 / 0.0098 |
| **E4** | 0.5931 / 0.6622 | 0.5518 / 0.8268 | 0.3782 / 0.6929 | 0.1640 / 0.2058 | 0.0213 / 0.0319 |

### 3.3 Core Findings & Scientific Insights
1. **The Absent-Class Evaluator Trap Resolved:** Single-fold evaluations previously gave a false $1.00$ IoU to absent classes under a $0/0$ evaluator defect, encouraging harmful $-3.0$ logit suppression. In LiteV3, zero artificial logit offsets are used. Real sharks/rays in `sharkFish` are detected with **0.7388 precision** and positive recall.
2. **Distillation Dynamics in Dense Rare-Class Segmentation:** Standard uniform Knowledge Distillation (E2) boosts majority class IoU (Background $+3.65\%$, Fish $+1.79\%$), but because background and fish constitute $>95\%$ of pixels, uniform spatial KL divergence washes out rare class representations. Weighting distillation by inverse class frequencies (E3) successfully restores Class 3 recall from $0.0\%$ to $23.38\%$.
3. **Optimality of E1:** E1 (Supervised Baseline with rare-class-weighted cross-entropy) achieves the highest composite score (**0.3956**), highest macro segmentation score (**0.3955**), and best balanced per-class recall across all five classes.

---

## 4. Submission Artifact Manifest & Checksums

All artifacts have been created, verified, and preserved with cryptographic checksums:

| Artifact Role | File Path | File Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Primary Submission ZIP** | `deploy/eurs_litev3_track2_submission.zip` | 1,829,775 bytes (1.75 MB) | `712e4f777985d6d1fe46b08d76143fda14ff4e958eecbc744b72b0519f6a332b` |
| **Standalone Production ONNX** | `deploy/eurs_litev3_track2.onnx` | 3,187,927 bytes (3.04 MB) | `ed939fc16c15c4d93e2fc9bd6c53164592b2a66d2fd5d3256ac1f642fbfcf03f` |
| **Primary Source Checkpoint** | `checkpoints/litev3/E1_supervised_f1/best_model.pt` | 6,094,868 bytes (5.81 MB) | `45baae60967520e5c9b9ec20a32509177a4a9009eb8b6348ef53e20ec4e26219` |
| **Insurance Fallback Candidate** | `deploy/eurs_net_v5_calibrated_track2.zip` | 2,752,190 bytes (2.62 MB) | `38c15076a18c0c3432ec59261c2eaf87143e41693cdf227b6f38855c15c5fecb` |
| **Track 1 V26 Production ZIP** | `deploy/v26/submission_v26_test_apex_sota.zip` | 266,742,019 bytes (254.4 MB) | `f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f` |

### ZIP Structure Verification
```
deploy/eurs_litev3_track2_submission.zip
└── model.onnx (Root level, Deflate compressed, 3,187,927 bytes uncompressed)
```

---

## 5. NVIDIA Jetson AGX Orin Deployment Specifications

### 5.1 TensorRT Compilation Command
To deploy `model.onnx` on NVIDIA Jetson AGX Orin with TensorRT FP16 acceleration:

```bash
# On Jetson AGX Orin (JetPack 5.x / 6.x with TensorRT 8.6+)
trtexec \
    --onnx=model.onnx \
    --saveEngine=eurs_litev3_fp16.engine \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw,fp16:chw \
    --warmup=200 \
    --iterations=1000
```

### 5.2 Latency & Memory Profile
- **Input Resolution:** $[1, 3, 384, 768]$ FP16
- **Desktop RTX 5090 Inference Latency:** **$1.82\text{ ms}$** ($\sim 549\text{ FPS}$)
- **Estimated Jetson AGX Orin Latency (FP16):** **$\sim 8.4\text{ ms}$** ($\sim 119\text{ FPS}$, well within the 30 FPS / 33.3 ms edge real-time requirement)
- **Peak Engine Memory Footprint:** **$< 35\text{ MB}$ VRAM**

---

## 6. Human Submission Checklist & Decision Protocol

Under our audit charter, **no automated submissions are made**. A designated human researcher must execute the submission via the official competition portal.

### Submission Step-by-Step Instructions:
1. **Log in to the competition portal** and navigate to Track 2 (Embedded Edge Deployment).
2. **Inspect testing-phase status:** Check whether submission `#18699` is marked Development or Testing, and verify remaining test-phase submission attempts.
3. **If a Testing-phase submission slot is available:**
   - Upload file: `deploy/eurs_litev3_track2_submission.zip` (SHA-256: `712e4f777985d6d1fe46b08d76143fda14ff4e958eecbc744b72b0519f6a332b`).
   - Description / Method: `SAT-EURS-LiteV3 (Truncated MobileNetV3-Large + Multiscale Decoder + RSUIGM-200, 0.486M params, 1.63 GFLOPs)`.
4. **If testing-phase submission is constrained or fails portal validation:**
   - Immediately fallback to: `deploy/eurs_net_v5_calibrated_track2.zip` (SHA-256: `38c15076a18c0c3432ec59261c2eaf87143e41693cdf227b6f38855c15c5fecb`), which is pre-verified on the leaderboard with score 0.3380.
5. **Log official submission ID and server-evaluated score** in the team log upon completion.
