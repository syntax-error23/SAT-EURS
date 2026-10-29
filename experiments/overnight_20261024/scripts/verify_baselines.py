import os
import os.path as osp
import zipfile
import hashlib
import json

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

archives = {
    "V18": {
        "path": "deploy/v18/submission_v18_dual_encoder_sota.zip",
        "expected_sha": "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52",
        "score": 0.749,
        "phase": "Development Phase"
    },
    "V27": {
        "path": "deploy/v27/submission_v27_test_hybrid_sota.zip",
        "expected_sha": "722b106ae817e4efbe911a83497ffabe4aa42387f2aedee7e5ac85deb6668c7c",
        "score": 0.633,
        "phase": "Testing Phase Entry #19096"
    },
    "V28-A": {
        "path": "deploy/v28/submission_v28a_fivefold_ablation.zip",
        "expected_sha": "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39",
        "score": 0.634,
        "phase": "Testing Phase Official Entry"
    }
}

report_lines = [
    "# SAT-EURS Track 1 — Production Baseline & Archive Verification",
    "",
    "## 1. Cryptographic Hash Verification",
    "",
    "| Version | Archive Path | Size (MB) | Expected SHA-256 | Actual SHA-256 | Status | Official Score |",
    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
]

all_passed = True
details = {}

for name, meta in archives.items():
    p = meta["path"]
    assert osp.exists(p), f"Missing baseline archive: {p}"
    size_mb = osp.getsize(p) / (1024 * 1024)
    actual_sha = get_sha256(p)
    status = "EXACT MATCH" if actual_sha.lower() == meta["expected_sha"].lower() else "MISMATCH"
    if status != "EXACT MATCH":
        all_passed = False
    report_lines.append(f"| **{name}** | `{p}` | {size_mb:.2f} | `{meta['expected_sha'][:16]}...` | `{actual_sha[:16]}...` | **{status}** | **{meta['score']}** ({meta['phase']}) |")

    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
        restored = [n for n in names if n.startswith("restored/") and not n.endswith("/")]
        seg_masks = [n for n in names if n.startswith("seg_masks/") and not n.endswith("/")]
        top_dirs = sorted(list(set(n.split('/')[0] for n in names if '/' in n)))

    details[name] = {
        "path": p,
        "size_mb": round(size_mb, 2),
        "actual_sha": actual_sha,
        "total_files": len(names),
        "top_dirs": top_dirs,
        "restored_count": len(restored),
        "seg_masks_count": len(seg_masks)
    }

report_lines.extend([
    "",
    "## 2. Archive Structural Layout",
    "",
    "| Version | Total Files | Top-level Directories | `restored/` Count | `seg_masks/` Count | Compliance |",
    "| :--- | :---: | :---: | :---: | :---: | :---: |"
])

for name, d in details.items():
    compliant = (d["total_files"] == 2400 and d["restored_count"] == 1200 and d["seg_masks_count"] == 1200 and set(d["top_dirs"]) == {"restored", "seg_masks"})
    c_str = "100% COMPLIANT" if compliant else "NON-COMPLIANT"
    report_lines.append(f"| **{name}** | {d['total_files']} | `{', '.join(d['top_dirs'])}` | {d['restored_count']} | {d['seg_masks_count']} | **{c_str}** |")

report_lines.extend([
    "",
    "## 3. Official Metric Context (V28-A: Official Composite = 0.634)",
    "",
    "| Component / Metric | Official V27 (#19096) | Official V28-A (Verified) | Delta (V28-A vs V27) | Scoring Weight |",
    "| :--- | :---: | :---: | :---: | :---: |",
    "| **Composite Score ($S$)** | **0.633** | **0.634** | **+0.001** | **100%** |",
    "| **Segmentation Quality ($Q_{\\text{seg}}$)** | $0.568$ | **0.570** | $+0.002$ | $60\\%$ |",
    "| **Segmentation mIoU** | $0.700$ | **0.704** | $+0.004$ | — |",
    "| **Segmentation Macro F1** | $0.810$ | **0.810** | $+0.000$ | — |",
    "| **Restoration PSNR (dB)** | $25.48$ | $25.48$ | $+0.00$ (Byte-identical) | — |",
    "| **Restoration SSIM** | $0.829$ | $0.829$ | $+0.000$ (Byte-identical) | — |",
    "| **Restoration Sharpness Ratio** | $0.973$ | $0.973$ | $+0.000$ (Byte-identical) | — |",
    "| **No-Reference UIQM** | $0.644$ | $0.644$ | $+0.000$ (Byte-identical) | — |",
    "| **No-Reference UCIQE** | $0.592$ | $0.592$ | $+0.000$ (Byte-identical) | — |",
    "| **No-Reference CCF** | $0.090$ | $0.090$ | $+0.000$ (Byte-identical) | — |",
    "",
    "### Key Observation:",
    "- V28-A removed the Full-20 segmentation blend ($w=0.50 \\to 0.00$), which lifted official $Q_{\\text{seg}}$ from $0.568$ to $0.570$ (+0.002) and mIoU from $0.700$ to $0.704$ (+0.004), boosting composite to **0.634**.",
    "- However, +0.001 is a modest gain. To make progress toward the competitive frontier (current testing leaderboard reference ~0.753), we must explore both segmentation and restoration improvements systematically.",
    ""
])

with open("experiments/overnight_20261024/baseline.md", "w") as f:
    f.write("\n".join(report_lines))

print("Baseline verification completed successfully. All passed:", all_passed)
