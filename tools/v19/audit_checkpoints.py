import os
import glob
import hashlib
import json
import torch

def get_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest_path = 'splits/manifest_5fold.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print('--- MANIFEST 5-FOLD ---')
    folds_dict = manifest.get('folds', manifest)
    for k, v in folds_dict.items():
        if isinstance(v, dict):
            val = v.get("val_scenes", v.get("val"))
            train = v.get("train_scenes", v.get("train"))
            print(f'{k}: val_scenes={val}, train_count={len(train)}')

    checkpoints_to_check = {
        # V18
        'V18_fold0_seg': 'checkpoints/v18/fold_0/segmenter_best.pt',
        'V18_fold1_seg': 'checkpoints/v18/fold_1/segmenter_best.pt',
        'V18_fold2_seg': 'checkpoints/v18/fold_2/segmenter_best.pt',
        'V18_fold3_seg': 'checkpoints/v18/fold_3/segmenter_best.pt',
        'V18_fold4_seg': 'checkpoints/v18/fold_4/segmenter_best.pt',
        'V18_fold0_ref': 'checkpoints/v18/fold_0/refiner_best.pt',
        'V18_fold1_ref': 'checkpoints/v18/fold_1/refiner_best.pt',
        'V18_fold2_ref': 'checkpoints/v18/fold_2/refiner_best.pt',
        'V18_fold3_ref': 'checkpoints/v18/fold_3/refiner_best.pt',
        'V18_fold4_ref': 'checkpoints/v18/fold_4/refiner_best.pt',

        # V16 DINOv2
        'V16_dinov2_f0': 'checkpoints/v16_dinov2/fold_0/best_model.pt',
        'V16_dinov2_f1': 'checkpoints/v16_dinov2/fold_1/best_model.pt',
        'V16_dinov2_f2': 'checkpoints/v16_dinov2/fold_2/best_model.pt',
        'V16_dinov2_f3': 'checkpoints/v16_dinov2/fold_3/best_model.pt',
        'V16_dinov2_f4': 'checkpoints/v16_dinov2/fold_4/best_model.pt',
        'V16_dinov2_f4_cp': 'checkpoints/track1_screening/DINOV2_CP_F4/best_model.pt',

        # V16 ConvNeXt
        'V16_convnext_f0': 'checkpoints/v16_convnext/fold_0/best_model.pt',
        'V16_convnext_f1': 'checkpoints/v16_convnext/fold_1/best_model.pt',
        'V16_convnext_f2': 'checkpoints/v16_convnext/fold_2/best_model.pt',
        'V16_convnext_f3': 'checkpoints/v16_convnext/fold_3/best_model.pt',
        'V16_convnext_f4': 'checkpoints/v16_convnext/fold_4/best_model.pt',

        # V17 Specialists
        'V17_spec_convnext': 'checkpoints/v17_specialists/convnext_full20_best.pt',
        'V17_spec_dinov2': 'checkpoints/v17_specialists/dinov2_full20_best.pt',

        # V16 baseline / other
        'V16_segformer': 'checkpoints/segformer_mit_b3_best.pt',
        
        # Restoration tables
        'tbl19_20sc_clean': 'checkpoints/track1_screening/tbl19_20sc_clean.pt',
        'table_cand_B_f0': 'checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_0.pt',
        'table_cand_B_f1': 'checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_1.pt',
        'table_cand_B_f2': 'checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_2.pt',
        'table_cand_B_f3': 'checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_3.pt',
        'table_cand_B_f4': 'checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_4.pt',
    }

    print('\n--- CHECKPOINT EXISTENCE, SIZE & SHA256 ---')
    out_dict = {}
    for name, p in checkpoints_to_check.items():
        if os.path.exists(p):
            sha = get_hash(p)
            sz = os.path.getsize(p) / (1024*1024)
            out_dict[name] = {'path': p, 'exists': True, 'sha256': sha, 'size_mb': round(sz, 2)}
            print(f'{name:<20} | {sz:8.2f} MB | {p:<60} | SHA256: {sha}')
        else:
            out_dict[name] = {'path': p, 'exists': False}
            print(f'{name:<20} | MISSING | {p}')

    with open('reports/v19/exp_p2_ensemble_feasibility/checkpoint_inventory.json', 'w') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == '__main__':
    main()
