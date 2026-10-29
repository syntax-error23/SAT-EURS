"""
SAT-EURS Track 1 — Persistent, Resumable Experiment Controller
Manages continuous experiment execution, durable queue state, incremental logging,
and process recovery across the overnight research campaign.
"""

import os
import os.path as osp
import sys
import time
import json
import subprocess
import shutil
from typing import Dict, List, Any, Optional

WORKSPACE_ROOT = "c:\\Projects\\SAT-EURS"
CAMPAIGN_DIR = osp.join(WORKSPACE_ROOT, "experiments", "overnight_20261024")
CONTROLLER_DIR = osp.join(CAMPAIGN_DIR, "controller")
QUEUE_FILE = osp.join(CONTROLLER_DIR, "experiment_queue.json")
LOGS_DIR = osp.join(CAMPAIGN_DIR, "logs")
RESULTS_CSV = osp.join(CAMPAIGN_DIR, "results.csv")
RESULTS_JSONL = osp.join(CAMPAIGN_DIR, "results.jsonl")
RUNTIME_LOG = osp.join(CAMPAIGN_DIR, "runtime_log.txt")
CONTINUATION_REPORT = osp.join(CAMPAIGN_DIR, "CONTINUATION_REPORT.md")


def load_queue() -> Dict[str, Any]:
    if not osp.exists(QUEUE_FILE):
        raise FileNotFoundError(f"Missing queue file: {QUEUE_FILE}")
    with open(QUEUE_FILE, "r") as f:
        return json.load(f)


def save_queue(queue_data: Dict[str, Any]):
    os.makedirs(osp.dirname(QUEUE_FILE), exist_ok=True)
    temp_file = QUEUE_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(queue_data, f, indent=2)
    shutil.move(temp_file, QUEUE_FILE)


def append_log(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    entry = f"[{ts}] {message}\n"
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    print(entry.strip(), flush=True)


def update_results_csv(exp: Dict[str, Any]):
    line = f"{exp['exp_id']},{exp['track']},{exp['name']},\"{exp['description']}\",{exp['status']},\"{exp.get('summary_metrics', '')}\",{exp.get('composite_est', 0.0):.4f},{exp.get('delta', 0.0):+.4f},{exp.get('verdict', 'INCONCLUSIVE')},\"{exp.get('notes', '')}\"\n"
    with open(RESULTS_CSV, "a", encoding="utf-8") as f:
        f.write(line)


def update_results_jsonl(exp: Dict[str, Any]):
    record = {
        "experiment_id": exp["exp_id"],
        "track": exp["track"],
        "name": exp["name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": exp.get("runtime_s", 0.0),
        "status": exp["status"],
        "metrics": exp.get("metrics", {}),
        "verdict": exp.get("verdict", "INCONCLUSIVE"),
        "notes": exp.get("notes", "")
    }
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def execute_next_experiment() -> Optional[Dict[str, Any]]:
    """
    Finds the next PENDING experiment in the queue, executes it, captures logs,
    and updates the durable state.
    """
    queue_data = load_queue()
    experiments = queue_data.get("experiments", [])

    next_exp = None
    for exp in experiments:
        if exp["status"] == "PENDING":
            next_exp = exp
            break

    if not next_exp:
        print("[Controller] No pending experiments in queue.", flush=True)
        return None

    exp_id = next_exp["exp_id"]
    next_exp["status"] = "RUNNING"
    next_exp["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_queue(queue_data)

    log_file = osp.join(LOGS_DIR, f"{exp_id}.log")
    os.makedirs(LOGS_DIR, exist_ok=True)
    next_exp["log_file"] = log_file

    append_log(f"Starting {exp_id} ({next_exp['name']}): {next_exp['command']}")

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            next_exp["command"],
            shell=True,
            cwd=WORKSPACE_ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT
        )
        proc.wait()

    runtime_s = round(time.time() - t0, 2)
    next_exp["runtime_s"] = runtime_s
    next_exp["exit_code"] = proc.returncode

    if proc.returncode == 0:
        next_exp["status"] = "COMPLETED"
        # Parse output JSON if specified
        out_json = next_exp.get("output_json")
        if out_json and osp.exists(out_json):
            with open(out_json, "r") as jf:
                out_data = json.load(jf)
                next_exp["metrics"] = out_data.get("metrics", out_data)
                if "verdict" in out_data:
                    next_exp["verdict"] = out_data["verdict"]
        append_log(f"Completed {exp_id} in {runtime_s}s with exit code 0.")
    else:
        next_exp["status"] = "FAILED"
        append_log(f"FAILED {exp_id} with exit code {proc.returncode} in {runtime_s}s. Inspect {log_file}")

    save_queue(queue_data)
    update_results_csv(next_exp)
    update_results_jsonl(next_exp)

    return next_exp


if __name__ == "__main__":
    if "--all" in sys.argv or "--loop" in sys.argv:
        while True:
            res = execute_next_experiment()
            if res is None:
                break
    else:
        execute_next_experiment()
