"""
Unified Test Runner for All Phase 1 Verification Suites
Executes all 11 unit & integration test suites and logs a summary report.
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.paths import REPO_ROOT

TEST_SUITES = [
    ("Stage 1", "Core Architecture & Budget Audit", "tests/test_architecture.py"),
    ("Stage 2", "Dataset Integrity & Pairing Invariants", "tests/test_dataset_integrity.py"),
    ("Stage 3", "Transform Coordinate Alignment", "tests/test_transform_alignment.py"),
    ("Stage 3", "Small-Object Retention & Label Integrity", "tests/test_small_object_preservation.py"),
    ("Stage 4", "Loss Contracts & Module-Level Gradients", "tests/test_losses_and_gradients.py"),
    ("Stage 5", "Metric Golden Fixtures & Guardrails", "tests/test_metric_fixtures.py"),
    ("Stage 6", "Checkpoint Numerical Parity Roundtrip", "tests/test_checkpoint_roundtrip.py"),
    ("Stage 6", "10-Sample Memorization & Overfit Test", "tests/test_overfit.py"),
    ("Stage 7", "Restoration Output Clamp & Gradients", "tests/test_output_clamp.py"),
    ("Stage 7", "Device Parity & Numerical Stability", "tests/test_device_parity.py"),
    ("Stage 8", "Learning Rate Scheduler Unit Test", "tests/test_lr_scheduler.py"),
    ("Stage 8", "Minimal FP32 Training Loop (1 Epoch)", "train_debug.py"),
]


def run_all_tests():
    print("\n" + "#" * 70)
    print("      EURS-NET V2: COMPLETE PHASE 1 MASTER VERIFICATION SUITE")
    print("#" * 70 + "\n")

    results = []
    total_start = time.time()

    from utils.paths import REPO_ROOT
    for stage, name, script in TEST_SUITES:
        script_path = os.path.join(str(REPO_ROOT), script)
        print(f"Running [{stage}] {name} ({script})...")
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        elapsed = time.time() - t0

        status = "PASS" if res.returncode == 0 else "FAIL"
        results.append((stage, name, script, status, elapsed))

        if res.returncode == 0:
            print(f"  --> [PASS] in {elapsed:.2f} s")
        else:
            print(f"  --> [FAIL] in {elapsed:.2f} s")
            print("Stderr:\n", res.stderr[-500:])
            print("Stdout:\n", res.stdout[-500:])

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("                     PHASE 1 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<10} | {'Test Suite':<40} | {'Status':<6} | {'Time':<6}")
    print("-" * 70)

    all_passed = True
    for stage, name, script, status, elapsed in results:
        status_str = f"\033[92m{status}\033[0m" if status == "PASS" else f"\033[91m{status}\033[0m"
        print(f"{stage:<10} | {name:<40} | {status:<6} | {elapsed:>5.2f}s")
        if status != "PASS":
            all_passed = False

    print("=" * 70)
    print(f"Total Execution Time: {total_time:.2f} s")
    if all_passed:
        print("ALL 12 TEST SUITES PASSED! PHASE 1 PIPELINE 100% VERIFIED.")
    else:
        print("SOME TESTS FAILED! Review logs above.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_tests()
