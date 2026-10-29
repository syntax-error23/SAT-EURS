import os
import os.path as osp
import sys
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    V18CanonicalPreprocessor,
)

CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]

def evaluate_scene_predictions(
    val_scenes: List[str],
    segmenter: torch.nn.Module,
    refiner: torch.nn.Module,
    preprocessor: V18CanonicalPreprocessor,
    device: torch.device,
    data_root: str = "data/cached_1024",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    use_flip_tta: bool = True,
    sample_stride: int = 5,
    min_component_size: int = 0,
    restoration_blend_raw: float = 0.0,
    unsharp_alpha: float = 0.10,
    unsharp_sigma: float = 2.0,
    eval_restoration: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluates a segmenter and refiner on a set of validation scenes.
    Returns:
        conf_mat: 5x5 confusion matrix
        restoration_stats: dict with PSNR, SSIM, Sharpness if eval_restoration is True
    """
    conf_mat = np.zeros((5, 5), dtype=np.int64)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    psnr_list = []
    ssim_list = []
    sharp_list = []

    for sc in val_scenes:
        mask_path = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gh, gw = gt_mask.shape[:2]

        clean_path = osp.join(data_root, "clean", f"{sc}.png")
        clean_img = cv2.imread(clean_path) if eval_restoration and osp.exists(clean_path) else None

        deg_scene_dir = osp.join(data_root, "degraded", sc)
        deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])
        sampled_files = deg_files[::sample_stride]

        for fname in sampled_files:
            stem = osp.splitext(fname)[0]
            parts = stem.split("_")
            wt = int(parts[1])
            depth = int(parts[2])

            img_path = osp.join(deg_scene_dir, fname)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            prep = preprocessor.prepare_input(bgr)
            raw_pad = prep["rgb_tensor"].to(device)
            meta = prep["meta"]
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.inference_mode():
                out_r = refiner(raw_pad.float(), wt_t, dp_t)
                restored_fused = out_r["refined_image"]

                if restoration_blend_raw > 0.0:
                    restored_fused = (1.0 - restoration_blend_raw) * restored_fused + restoration_blend_raw * raw_pad.float()

                # Segmentation inference
                if use_flip_tta:
                    raw_pad_flip = torch.flip(raw_pad, dims=[3])
                    restored_flip = torch.flip(restored_fused, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                        out_flip = segmenter(restored_flip, raw_pad_flip, wt_t, dp_t)
                        p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                        prob_fused = 0.5 * (p_orig + p_flip)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        prob_fused = F.softmax(out_orig["logits"].float(), dim=1)

                log_prob = torch.log(prob_fused + 1e-10) + cal_t
                mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            pred_mask = preprocessor.unpad_output(mask_pad, meta)
            if pred_mask.shape != (gh, gw):
                pred_mask = cv2.resize(pred_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)

            # Optional connected component filter for small noise islands
            if min_component_size > 0:
                for c in range(1, 5):  # only filter foreground classes
                    binary = (pred_mask == c).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    for lbl in range(1, num_labels):
                        area = stats[lbl, cv2.CC_STAT_AREA]
                        if area < min_component_size:
                            pred_mask[labels == lbl] = 0  # revert to background

            y_t = gt_mask.reshape(-1)
            y_p = pred_mask.reshape(-1)
            indices = 5 * y_t + y_p
            counts = np.bincount(indices, minlength=25).reshape(5, 5)
            conf_mat += counts

    return conf_mat


def compute_metrics_from_conf_mat(conf_mat: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(conf_mat).astype(np.float64)
    gt_px = conf_mat.sum(axis=1).astype(np.float64)
    pred_px = conf_mat.sum(axis=0).astype(np.float64)
    fp = pred_px - tp
    fn = gt_px - tp

    ious = []
    f1s = []
    precs = []
    recs = []
    coverage = 1.0

    for c in range(5):
        denom_iou = tp[c] + fp[c] + fn[c]
        denom_f1 = 2 * tp[c] + fp[c] + fn[c]
        denom_prec = tp[c] + fp[c]
        denom_rec = gt_px[c]

        iou = float(tp[c] / denom_iou) if denom_iou > 0 else (1.0 if gt_px[c] == 0 else 0.0)
        f1 = float((2 * tp[c]) / denom_f1) if denom_f1 > 0 else (1.0 if gt_px[c] == 0 else 0.0)
        prec = float(tp[c] / denom_prec) if denom_prec > 0 else 0.0
        rec = float(tp[c] / denom_rec) if denom_rec > 0 else 0.0

        ious.append(iou)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

        if gt_px[c] > 0 and tp[c] == 0:
            coverage = 0.0

    miou = float(np.mean(ious))
    macro_f1 = float(np.mean(f1s))
    q_seg = float(miou * macro_f1 * coverage)

    return {
        "Q_seg": round(q_seg, 5),
        "mIoU": round(miou, 5),
        "Macro_F1": round(macro_f1, 5),
        "coverage": coverage,
        "class_ious": [round(x, 5) for x in ious],
        "class_f1s": [round(x, 5) for x in f1s],
        "class_precisions": [round(x, 5) for x in precs],
        "class_recalls": [round(x, 5) for x in recs],
        "confusion_matrix": conf_mat.tolist()
    }


def run_full_20scene_oof_benchmark(
    segmenter_ckpts: Dict[int, str],
    refiner_ckpts: Dict[int, str],
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    use_flip_tta: bool = True,
    sample_stride: int = 5,
    min_component_size: int = 0,
    restoration_blend_raw: float = 0.0,
    device_id: int = 0
) -> Dict[str, Any]:
    """
    Executes a complete 20-scene true out-of-fold benchmark.
    Each of the 5 folds predicts strictly on its held-out validation scenes.
    Accumulates both per-fold and pooled 20-scene confusion matrices.
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    base_table = torch.load(table_path, map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    fold_metrics = {}
    pooled_conf_mat = np.zeros((5, 5), dtype=np.int64)
    t0 = time.time()

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        # Load fold refiner
        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[fold], map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        # Load fold segmenter
        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_obj = torch.load(segmenter_ckpts[fold], map_location=device)
        s_sd = s_obj["state_dict"] if "state_dict" in s_obj else s_obj
        segmenter.load_state_dict(s_sd)
        segmenter.eval()

        cmat = evaluate_scene_predictions(
            val_scenes=val_scenes,
            segmenter=segmenter,
            refiner=refiner,
            preprocessor=preprocessor,
            device=device,
            data_root=data_root,
            offset_vector=offset_vector,
            use_flip_tta=use_flip_tta,
            sample_stride=sample_stride,
            min_component_size=min_component_size,
            restoration_blend_raw=restoration_blend_raw
        )

        fold_m = compute_metrics_from_conf_mat(cmat)
        fold_metrics[f"fold_{fold}"] = {
            "val_scenes": val_scenes,
            "metrics": fold_m
        }
        pooled_conf_mat += cmat
        print(f"Fold {fold} complete ({val_scenes}): Q_seg={fold_m['Q_seg']:.4f}, mIoU={fold_m['mIoU']:.4f}, F1={fold_m['Macro_F1']:.4f}", flush=True)

    pooled_m = compute_metrics_from_conf_mat(pooled_conf_mat)
    total_time = time.time() - t0

    return {
        "pooled_metrics": pooled_m,
        "fold_metrics": fold_metrics,
        "runtime_s": round(total_time, 2),
        "settings": {
            "offset_vector": offset_vector,
            "use_flip_tta": use_flip_tta,
            "sample_stride": sample_stride,
            "min_component_size": min_component_size,
            "restoration_blend_raw": restoration_blend_raw
        }
    }

if __name__ == "__main__":
    print("Testing benchmark engine...")
