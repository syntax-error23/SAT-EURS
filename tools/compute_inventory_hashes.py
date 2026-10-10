import os
import hashlib
import json
import time
from pathlib import Path

files_to_hash = [
    'checkpoints/run2_50ep/forensic_backup/best_model_preserved.pt',
    'checkpoints/run2_50ep/best_model.pt',
    'checkpoints/run2_50ep/forensic_backup/latest_ep37_preserved.pt',
    'checkpoints/run2_50ep/latest.pt',
    'checkpoints/latest_run1.pt',
    'checkpoints/best_model_run1.pt',
    'checkpoints/exp_bounded_recovery/bounded_recovery_ep5.pt',
    'models/eurs_net_v2.py',
    'models/decoders.py',
    'models/backbone.py',
    'models/common.py',
    'losses/restoration_loss.py',
    'losses/msssim.py',
    'losses/loss_config.py',
    'losses/multi_task_loss.py',
    'evaluate.py',
    'metrics/restoration_metrics.py',
    'metrics/segmentation_metrics.py',
    'metrics/composite_score.py',
    'metrics/nr_metric.py',
    'configs/baseline_v1.yaml',
    'train.py'
]

inventory = {}
for rel_path in files_to_hash:
    p = Path(rel_path)
    if p.exists():
        data = p.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        stat = p.stat()
        inventory[rel_path] = {
            'abs_path': str(p.resolve()),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 4),
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
            'sha256': sha256
        }
    else:
        inventory[rel_path] = {'exists': False}

out_path = 'audits/final_forensic_audit/file_inventory_and_hashes.json'
with open(out_path, 'w') as f:
    json.dump(inventory, f, indent=2)

print('Inventory written to:', out_path)
for k, v in inventory.items():
    if v.get('exists', True):
        sz = v['size_bytes']
        sh = v['sha256'][:16]
        mt = v['mtime']
        print(f"{k:58s} | {sz:10d} B | {sh}... | {mt}")
