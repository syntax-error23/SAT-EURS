# SAT-EURS Track 2 — Phase 0 Status & State Preservation Audit

**Document Version:** 1.0.0  
**Audit Timestamp:** 2026-10-24T02:35:00+05:30  
**Author:** Senior CV Researcher & Competition Auditor  
**Branch:** `track2-litev3-rescue` (Branched from `master` @ commit `cd677b2`)

---

## 1. Executive Summary & State Preservation

Before executing any modifications or training runs, a strict state preservation protocol was executed:
1. **Branch Isolation:** A dedicated branch `track2-litev3-rescue` was created and checked out from `cd677b2`. Production branches (`master`, `track1-v26-submission`) remain untouched.
2. **Read-Only Preservation of Production Artifacts:**
   - The official Track 1 V26 submission package `deploy/v26/submission_v26_test_apex_sota.zip` was verified with SHA-256 hash `f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f` and left unmodified.
   - All historical checkpoints (`checkpoints/EXP-PRETRAINED-30EP_best.pt`, `checkpoints/track1_screening/`, `models/weights/`) remain intact.
3. **No Unilateral Portal Submissions:** No automated submission to the competition portal will be executed. All artifacts are prepared in `deploy/` with full provenance, manifests, and verification logs for human sign-off.

---

## 2. Git Environment & Repository State

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Initial HEAD Commit** | `cd677b260bf139b4b0a701ff6603a11e4bfbe130` | `master` branch |
| **Active Rescue Branch** | `track2-litev3-rescue` | Created specifically for Track 2 rescue |
| **Active Workspace** | `c:\Projects\SAT-EURS` | Dedicated competition workspace |
| **Host System** | Intel Core Ultra 9 285K (24 cores) | 64 GB DDR5 RAM, Windows 11 OS |
| **Compute Acceleration** | NVIDIA GeForce RTX 5090 | 32.6 GB VRAM, Driver 572.70, CUDA 12.8 |
| **PyTorch Version** | 2.6.0+cu128 | ONNX 1.17.0, ONNX Runtime 1.20.1 |

---

## 3. Submission Portal Status Verification

> [!WARNING]
> **Competition Portal Status:** Inaccessible / **UNVERIFIED**.  
> The external submission portal cannot be accessed directly from this environment due to network isolation and credential requirements. Under our audit charter, the testing-phase status of submission `#18699` is explicitly marked **UNVERIFIED**. We do NOT infer the competition phase from filenames, local manifests, or score values.

---

## 4. Fallback Insurance Candidate Audit

To guarantee zero risk of missed deadlines, we audited all existing candidate packages and identified the strongest compliant submission fallback:

- **Candidate Identity:** `EURSEdgeSOTA_V5_Calibrated`
- **ZIP Artifact:** `deploy/eurs_net_v5_calibrated_track2.zip`
- **ONNX File:** `deploy/eurs_edgesota_v2_final.onnx`
- **Artifact SHA-256:** `38c15076a18c0c3432ec59261c2eaf87143e41693cdf227b6f38855c15c5fecb`
- **Parameter Count:** **768,304** (0.7683 M) — **Compliant** ($\le 1.0\text{M}$)
- **Compute Complexity:** **9.05 GFLOPs** at $[1, 3, 512, 1024]$ — **Compliant** ($\le 12.0\text{ GFLOPs}$)
- **Official Scored Leaderboard Result:** Submission `#18699` scored **0.3380**.
- **Assessment:** This package is already packaged, compliant with official parameter and FLOP constraints, and serves as an immutable insurance fallback while SAT-EURS-LiteV3 is trained and verified.

---

## 5. Official Competition Constraints (Track 2)

| Constraint | Limit | Verification Tool |
| :--- | :--- | :--- |
| **Parameter Budget** | $\le 1,000,000$ (1.0 M) | `deploy/check_submission_flops.py` |
| **Compute Budget** | $\le 12.0\text{ GFLOPs}$ | `deploy/check_submission_flops.py` (FVCore) |
| **Operator Whitelist** | Strict `ALWAYS_ALLOWED` | Native ONNX checker & operator parser |
| **Input Specification** | Static $[1, 3, H, W]$ float32 in $[0, 1]$ | Tensor dimension check |
| **Output Specification** | Restored $[1, 3, H, W]$ + Logits $[1, 5, H, W]$ | Tensor dimension check |
| **Packaging Contract** | Standalone ZIP containing root `model.onnx` | `zipfile.ZipFile.namelist()` |
| **Deadline** | **October 27, 2026, 23:59 AoE** | 4-day execution window |
