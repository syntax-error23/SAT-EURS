import os
import json
import sys

sys.path.insert(0, ".")

from tools.v28.evaluate_fold_oof import evaluate_model_on_val_scenes

def main():
    ckpts = {
        0: 'checkpoints/v27/segmenter_v27_fold0_best.pt',
        1: 'checkpoints/v18/fold_1/segmenter_best.pt',
        2: 'checkpoints/v18/fold_2/segmenter_best.pt',
        3: 'checkpoints/v18/fold_3/segmenter_best.pt',
        4: 'checkpoints/v18/fold_4/segmenter_best.pt',
    }

    results = {}
    for fold, ckpt in ckpts.items():
        ref_ckpt = f'checkpoints/v18/fold_{fold}/refiner_best.pt'
        res = evaluate_model_on_val_scenes(
            segmenter_ckpt=ckpt,
            fold=fold,
            refiner_ckpt=ref_ckpt,
            sample_stride=5,  # 40 variants/scene * 4 scenes = 160 images per fold
            device_id=0
        )
        results[f'fold_{fold}'] = res

    out_file = 'reports/v28/phase2/fivefold_baseline_oof_report.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 80)
    print('=== 5-FOLD BASELINE OOF AGGREGATE SUMMARY (800 Images Across 20 Scenes) ===')
    print('=' * 80)
    q_segs = [results[f'fold_{f}']['global_metrics']['Q_seg'] for f in range(5)]
    mious = [results[f'fold_{f}']['global_metrics']['mIoU'] for f in range(5)]
    f1s = [results[f'fold_{f}']['global_metrics']['Macro_F1'] for f in range(5)]

    for f in range(5):
        val_sc = results[f'fold_{f}']['val_scenes']
        print(f'Fold {f} Val ({val_sc}): Q_seg={q_segs[f]:.4f} | mIoU={mious[f]:.4f} | F1={f1s[f]:.4f}')

    print('-' * 80)
    print(f'Aggregate 5-Fold Mean Q_seg:    {sum(q_segs)/5:.4f}')
    print(f'Aggregate 5-Fold Mean mIoU:     {sum(mious)/5:.4f}')
    print(f'Aggregate 5-Fold Mean Macro F1: {sum(f1s)/5:.4f}')
    print('=' * 80)

if __name__ == '__main__':
    main()
