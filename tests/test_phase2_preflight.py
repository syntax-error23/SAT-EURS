"""
Master Phase 2A Pre-Flight Gatekeeper
Runs all Phase 2A verification test suites locally on CPU before touching the RTX 5090:
1. ONNX Hardware Contract (Opset 18, static shapes, ORT numerical parity < 1e-4)
2. Manifest Integrity & Cryptographic SHA-256 Checksum
3. Dual Samplers (Standard vs SceneBalanced, fixed epoch step semantics)
4. Loss Stability (Mean-1.0 square-root smoothed weights, 1-pixel rare class gradient stability)
5. Cache Preprocessing Equivalence (Image max diff < 1e-3, mask bitwise identity)
6. Checkpoint Serialization & CPU Resume Determinism (torch.equal on all params and momentum buffers)
7. Track 1 Submission Packaging & Format Integrity
8. Official Metric Fixtures & 15-35 dB PSNR Normalization Window
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import subprocess
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


TEST_SUITES: List[Tuple[str, str, str]] = [
    ("WP 2A.1", "ONNX Track 2 Contract & Numerical Parity", "tests/test_onnx_contract.py"),
    ("WP 2A.2/3", "Manifest Cryptographic SHA-256 & Scene-Disjointness", "tools/verify_manifest.py"),
    ("WP 2A.4", "Dual Samplers & Fixed Epoch Step Semantics", "tests/test_samplers.py"),
    ("WP 2A.5", "Loss Configuration & Mean-1.0 Smoothed Weights", "tests/test_loss_stability_full.py"),
    ("WP 2A.6", "Preprocessing Cache & Equivalence Testing", "tests/test_cache_equivalence.py"),
    ("WP 2A.7", "Full Checkpointing & CPU Resume Determinism", "tests/test_resume_determinism.py"),
    ("WP 2A.8", "Track 1 Packaging & Resize Wrapper", "tests/test_submission_track1_format.py"),
    ("WP 2A.9", "Metric Fixtures & 15-35 dB PSNR Window", "tests/test_metric_fixtures.py")
]


def run_preflight_gatekeeper():
    start_total = time.time()
    results = []

    print("\n" + "=" * 70)
    print("  EURS @ MaCVi WACV 2027: MASTER PHASE 2A PRE-FLIGHT GATEKEEPER")
    print("  Executing Local Verification Fortress on Laptop (CPU)")
    print("=" * 70 + "\n")

    for code, desc, script in TEST_SUITES:
        print(f"--> Executing [{code}] {desc} ({script})...")
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"    [PASSED] in {elapsed:.2f}s\n")
            results.append((code, desc, "PASS", elapsed, None))
        else:
            print(f"    [FAILED] in {elapsed:.2f}s\n")
            error_preview = (res.stderr or res.stdout)[-800:]
            results.append((code, desc, "FAIL", elapsed, error_preview))

    total_time = time.time() - start_total

    print("=" * 70)
    print("                  PHASE 2A PRE-FLIGHT AUDIT SUMMARY")
    print("=" * 70)
    print(f"{'Code':<12} | {'Description':<42} | {'Status':<6} | {'Time':<6}")
    print("-" * 70)

    all_passed = True
    for code, desc, status, elapsed, err in results:
        status_str = f"\033[92m{status}\033[0m" if status == "PASS" else f"\033[91m{status}\033[0m"
        print(f"{code:<12} | {desc:<42} | {status:<6} | {elapsed:>5.2f}s")
        if status != "PASS":
            all_passed = False

    print("-" * 70)
    print(f"Total Pre-Flight Runtime: {total_time:.2f}s")
    print("=" * 70)

    if all_passed:
        print("\n  [VERDICT] 100% PHASE 2A TESTS PASSED.")
        print("  All defined Phase 1 & 2A invariants and automated verification tests pass.")
        print("  Zero known software defects remain.")
        print("  CODEBASE IS 100% PRE-FLIGHT VERIFIED FOR RTX 5090 EXECUTION.\n")
        return 0
    else:
        print("\n  [VERDICT] PRE-FLIGHT GATEKEEPER FAILED! Resolve failures above before GPU training.\n")
        for code, desc, status, elapsed, err in results:
            if status == "FAIL":
                print(f"--- FAILURE DETAILS FOR [{code}] {desc} ---")
                print(err)
                print("-" * 70)
        return 1


if __name__ == '__main__':
    exit_code = run_preflight_gatekeeper()
    sys.exit(exit_code)
