import os, json, cv2, numpy as np

masks_dir = 'data/masks'
scenes = sorted([os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.endswith('.png')])

scene_classes = {}
for sc in scenes:
    m = cv2.imread(os.path.join(masks_dir, f'{sc}.png'), cv2.IMREAD_GRAYSCALE)
    u = sorted(list(np.unique(m)))
    counts = np.bincount(m.ravel(), minlength=5)
    scene_classes[sc] = {
        'classes': u,
        'counts': [int(x) for x in counts[:5]]
    }
    print(f'{sc:12s}: classes={u} | pixels: Bg={counts[0]:8d}, Cor={counts[1]:8d}, Fish={counts[2]:7d}, Ray={counts[3]:7d}, Fauna={counts[4]:7d}')

with open('data/splits.json') as f:
    splits = json.load(f)

print('\n=== AUDIT OF 5-FOLD SCENE AND CLASS ALLOCATION (Tier 3) ===')
for fold_name, fold_data in splits['tier3'].items():
    tr_sc = fold_data['train']
    val_sc = fold_data['val']
    
    inter = set(tr_sc).intersection(set(val_sc))
    
    tr_classes = set()
    for s in tr_sc:
        tr_classes.update(scene_classes[s]['classes'])
        
    val_classes = set()
    for s in val_sc:
        val_classes.update(scene_classes[s]['classes'])
        
    print(f'Fold {fold_name}:')
    print(f'  Train ({len(tr_sc)} scenes): {tr_sc}')
    print(f'  Val   ({len(val_sc)} scenes): {val_sc}')
    print(f'  Leakage (intersection): {sorted(list(inter)) if inter else "NONE (CLEAN)"}')
    print(f'  Classes in Train: {sorted(list(tr_classes))}')
    print(f'  Classes in Val:   {sorted(list(val_classes))}')
    missing_in_val = set(range(5)) - val_classes
    if missing_in_val:
        print(f'  WARNING: Classes completely absent in validation: {sorted(list(missing_in_val))}')
