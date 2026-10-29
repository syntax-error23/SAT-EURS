"""
Production Training Script for EURS-Net V4 (Track 1 & Track 2 Unified Engine)
Target: MaCVi @ WACV 2027 Embedded Underwater Restoration & Segmentation (EURS) Challenge

Key V4 Capabilities & Fixes Integrated:
1. 100% Key-Compatible Warm-Start from V2 Champion (`best_controlled_checkpoint.pt`),
   preserving exact Step 0 outputs (`Composite = 0.6805` surrogate / `0.6596` @ NR=0.50).
2. Gradual 5-Stage Downscaling & Upscaling Pyramid (`EURSNetV4`) with:
   - Native-resolution Spatially-Varying Guided Affine Restoration (`affine_head_256`)
   - Native 1024x1024 Full-Resolution Edge & Detail Refiner (`fullres_refiner`)
   - Multi-Dilation Context Block (`context_64`, dilations 1, 2, 4) for Shark/Other
   - Detached Restoration-to-Segmentation Feature Guidance (`rest_adapter_256`)
   - Explicit 512x512 Auxiliary Morphological Boundary Supervision (`aux_boundary_head`)
3. Two-Stage Differential Learning Rate Schedule:
   - Stage 1 (Epochs 1..freeze_epochs): Warm up new V4 heads & restoration decoder while
     holding pre-trained backbone & V2 segmentation core frozen.
   - Stage 2 (Epochs freeze_epochs+1..epochs): Joint end-to-end fine-tuning with differential
     learning rates (0.25x backbone, 0.50x V2 decoders, 1.00x V4 new modules).
4. Epoch-Varying Augmentation Seeds (`train_dataset.set_epoch(epoch)`), Multi-Scale Crop,
   and 90-degree rotations.
5. Strict FP32 Loss Execution (`V4MultiTaskLoss`) preventing `bfloat16` MS-SSIM cancellation,
   black-pixel `ColorAngleLoss` singularity, and double class-weight `alpha` application.
6. Official MaCVi Evaluation & Guardrail Checkpointing:
   - Dataset-mean `PSNR_norm` and hard binary `coverage_penalty` (plus per-image & soft diagnostics)
   - Hard guardrails: `Shark IoU >= 0.1472`, `coverage_penalty == 1.0`, `clipping < 5.0%`
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH
from models.eurs_net_v4 import EURSNetV4
from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from data.copy_paste import RareClassCopyPasteAugmenter
from data.samplers import SceneBalancedSampler, StandardSampler
from losses.loss_config import compute_training_class_weights
from losses.v4_losses import V4MultiTaskLoss
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import (
    RestorationEvaluator,
    compute_psnr_and_norm,
    compute_ssim_single,
    compute_sharpness_ratio,
)
from metrics.nr_metric import NonReferenceQualityEvaluator
from metrics.composite_score import compute_composite_score
from utils.checkpointing import save_checkpoint, load_checkpoint


PROTECTED_CHECKPOINTS = {
    "checkpoints/run2_50ep/best_model.pt": "ceacd22546aaa4f8fbf0ef15c0641b568a0261c8ba62ca7ef342c40ac27e50ba",
    "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt": "444675aa4b6c25f686b8f0c30c9456b269b1c56d41c3d15d2bb4dd2e4229bf18",
}


def compute_file_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_protected_checkpoints() -> None:
    """Ensure baseline V1/V2 checkpoints are untouched on disk."""
    for rel_path, expected_sha in PROTECTED_CHECKPOINTS.items():
        if os.path.exists(rel_path):
            actual_sha = compute_file_sha256(rel_path)
            assert actual_sha == expected_sha, (
                f"Protected checkpoint hash mismatch for {rel_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )


def _compute_single_image_metrics(args: Tuple[np.ndarray, np.ndarray, str]) -> Dict[str, Any]:
    r_np, c_np, sid = args
    p_val, p_norm = compute_psnr_and_norm(r_np, c_np)
    s_val = compute_ssim_single(r_np, c_np)
    sh_val = compute_sharpness_ratio(r_np, c_np)

    cl_val = float(np.mean((r_np == 0.0) | (r_np == 1.0))) * 100.0
    cl_0 = float(np.mean(r_np == 0.0)) * 100.0
    cl_1 = float(np.mean(r_np == 1.0)) * 100.0
    cl_r = float(np.mean((r_np[:, :, 0] == 0.0) | (r_np[:, :, 0] == 1.0))) * 100.0
    cl_g = float(np.mean((r_np[:, :, 1] == 0.0) | (r_np[:, :, 1] == 1.0))) * 100.0
    cl_b = float(np.mean((r_np[:, :, 2] == 0.0) | (r_np[:, :, 2] == 1.0))) * 100.0

    return {
        'sid': sid,
        'psnr': p_val,
        'psnr_norm': p_norm,
        'ssim': s_val,
        'sharpness': sh_val,
        'clip_pct': cl_val,
        'clip_0': cl_0,
        'clip_1': cl_1,
        'clip_r': cl_r,
        'clip_g': cl_g,
        'clip_b': cl_b,
    }


def evaluate_v4_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> Dict[str, Any]:
    """
    Evaluate EURSNetV4 on the validation split (default: all 800 images) and return
    both Official MaCVi metrics (dataset-mean PSNR_norm, hard binary coverage_penalty)
    and diagnostic per-image / per-scene / per-channel clipping breakdowns.
    """
    model.eval()
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    class_names = ['Background', 'Coral', 'Fish', 'Shark', 'Other']

    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()
    nr_eval = NonReferenceQualityEvaluator()

    tasks = []
    with torch.no_grad():
        for b_idx, (deg, clean, mask, meta) in enumerate(loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            restored, seg_logits = model(deg)
            if restored.shape[2:] != clean.shape[2:]:
                restored = nn.functional.interpolate(restored, size=clean.shape[2:], mode='bilinear', align_corners=False)
            if seg_logits.shape[2:] != mask.shape[1:]:
                seg_logits = nn.functional.interpolate(seg_logits, size=mask.shape[1:], mode='bilinear', align_corners=False)

            seg_eval.update(seg_logits, mask)
            nr_eval.update(restored)

            b_size = deg.shape[0]
            rest_cpu = restored.permute(0, 2, 3, 1).cpu().numpy()
            clean_cpu = clean.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(b_size):
                tasks.append((rest_cpu[i], clean_cpu[i], meta['scene_id'][i]))

            if max_batches is not None and b_idx >= max_batches:
                break

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_compute_single_image_metrics, tasks))

    scene_data = {
        s: {'psnr': [], 'ssim': [], 'sharp': [], 'clip_pct': [], 'clip_r': [], 'clip_g': [], 'clip_b': []}
        for s in scenes
    }
    all_clip_pcts, all_clip_0, all_clip_1 = [], [], []
    all_clip_r, all_clip_g, all_clip_b = [], [], []

    for r in results:
        sid = r['sid']
        rest_eval.psnr_list.append(r['psnr'])
        rest_eval.psnr_norm_list.append(r['psnr_norm'])
        rest_eval.ssim_list.append(r['ssim'])
        rest_eval.sharpness_list.append(r['sharpness'])

        all_clip_pcts.append(r['clip_pct'])
        all_clip_0.append(r['clip_0'])
        all_clip_1.append(r['clip_1'])
        all_clip_r.append(r['clip_r'])
        all_clip_g.append(r['clip_g'])
        all_clip_b.append(r['clip_b'])

        if sid in scene_data:
            scene_data[sid]['psnr'].append(r['psnr'])
            scene_data[sid]['ssim'].append(r['ssim'])
            scene_data[sid]['sharp'].append(r['sharpness'])
            scene_data[sid]['clip_pct'].append(r['clip_pct'])
            scene_data[sid]['clip_r'].append(r['clip_r'])
            scene_data[sid]['clip_g'].append(r['clip_g'])
            scene_data[sid]['clip_b'].append(r['clip_b'])

    seg_res = seg_eval.compute()
    rest_res = rest_eval.compute()
    nr_res = nr_eval.compute()

    # Official MaCVi Composite (dataset-mean PSNR_norm, hard binary coverage_penalty)
    comp_official_surrogate = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], nr_res['Q_NR'])
    comp_official_std050 = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], 0.50)

    # Per-image PSNR_norm diagnostic Composite
    comp_per_img_surrogate = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest_per_image'], nr_res['Q_NR'])

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            'iou': float(seg_res['per_class_IoU'][idx]),
            'f1': float(seg_res['per_class_F1'][idx]),
            'precision': float(seg_res['per_class_precision'][idx]),
            'recall': float(seg_res['per_class_recall'][idx]),
        }

    per_scene = {}
    for s in scenes:
        count = len(scene_data[s]['psnr'])
        per_scene[s] = {
            'count': count,
            'psnr_mean': float(np.mean(scene_data[s]['psnr'])) if count > 0 else 0.0,
            'ssim_mean': float(np.mean(scene_data[s]['ssim'])) if count > 0 else 0.0,
            'sharp_mean': float(np.mean(scene_data[s]['sharp'])) if count > 0 else 0.0,
            'clip_pct_mean': float(np.mean(scene_data[s]['clip_pct'])) if count > 0 else 0.0,
            'clip_pct_max': float(np.max(scene_data[s]['clip_pct'])) if count > 0 else 0.0,
            'clip_r_mean': float(np.mean(scene_data[s]['clip_r'])) if count > 0 else 0.0,
            'clip_g_mean': float(np.mean(scene_data[s]['clip_g'])) if count > 0 else 0.0,
            'clip_b_mean': float(np.mean(scene_data[s]['clip_b'])) if count > 0 else 0.0,
            'clip_guardrail_pass': bool(np.mean(scene_data[s]['clip_pct']) < 5.0) if count > 0 else True,
        }

    aggregate_clip_pct = float(np.mean(all_clip_pcts)) if all_clip_pcts else 0.0

    return {
        'total_images': len(all_clip_pcts),
        'mIoU': seg_res['mIoU'],
        'macro_F1': seg_res['macro_F1'],
        'shark_iou': per_class['Shark']['iou'],
        'other_iou': per_class['Other']['iou'],
        'per_class': per_class,
        'coverage_penalty': seg_res['coverage_penalty'],
        'soft_coverage_penalty': seg_res['soft_coverage_penalty'],
        'Q_seg': seg_res['Q_seg'],
        'PSNR_dB': rest_res['PSNR_dB'],
        'PSNR_norm': rest_res['PSNR_norm'],
        'PSNR_norm_per_image': rest_res['PSNR_norm_per_image'],
        'SSIM': rest_res['SSIM'],
        'sharpness_ratio': rest_res['sharpness_ratio'],
        'Q_rest': rest_res['Q_rest'],
        'Q_rest_per_image': rest_res['Q_rest_per_image'],
        'clipping_aggregate_pct': round(aggregate_clip_pct, 4),
        'clipping_low_0_pct': round(float(np.mean(all_clip_0)), 4) if all_clip_0 else 0.0,
        'clipping_high_1_pct': round(float(np.mean(all_clip_1)), 4) if all_clip_1 else 0.0,
        'clipping_channels_pct': {
            'R': round(float(np.mean(all_clip_r)), 4) if all_clip_r else 0.0,
            'G': round(float(np.mean(all_clip_g)), 4) if all_clip_g else 0.0,
            'B': round(float(np.mean(all_clip_b)), 4) if all_clip_b else 0.0,
        },
        'clipping_pct_max': round(float(np.max(all_clip_pcts)), 4) if all_clip_pcts else 0.0,
        'clipping_guardrail_pass': bool(aggregate_clip_pct < 5.0),
        'per_scene': per_scene,
        'Q_NR_surrogate': nr_res['Q_NR'],
        'composite_official_surrogate': comp_official_surrogate['Composite_Score'],
        'composite_official_std050': comp_official_std050['Composite_Score'],
        'composite_per_image_surrogate': comp_per_img_surrogate['Composite_Score'],
    }


def is_better_v4_checkpoint(cand: Dict[str, Any], best: Dict[str, Any]) -> bool:
    """
    Guardrail-enforced checkpoint selection:
    1. coverage_penalty == 1.0 (all 5 classes detected)
    2. shark_iou >= 0.1472 (never regress below V2 champion Shark IoU)
    3. clipping_aggregate_pct < 5.0% and every scene clip_pct_mean < 5.0%
    4. Maximize Official MaCVi Composite Score (`composite_official_surrogate`)
    """
    if cand.get('coverage_penalty', 0.0) < 1.0:
        return False
    if cand.get('shark_iou', 0.0) < 0.1472:
        return False
    if cand.get('clipping_aggregate_pct', 100.0) >= 5.0:
        return False
    for _, s_info in cand.get('per_scene', {}).items():
        if s_info.get('clip_pct_mean', 100.0) >= 5.0:
            return False
    if best is None:
        return True

    s_cand = cand.get('composite_official_surrogate', 0.0)
    s_best = best.get('composite_official_surrogate', 0.0)
    if s_cand > s_best + 1e-4:
        return True
    if abs(s_cand - s_best) <= 1e-4:
        if cand.get('mIoU', 0.0) > best.get('mIoU', 0.0) + 1e-4:
            return True
    return False


def build_v4_optimizer(model: EURSNetV4, base_lr: float, weight_decay: float = 1e-4) -> optim.Optimizer:
    """
    Construct AdamW optimizer with 3 differential learning rate groups:
      Group 0 (backbone, fwb & imagenet_encoder): 0.35 * base_lr
      Group 1 (V2 restoration & seg core):        0.50 * base_lr
      Group 2 (V4 & Path-2 new heads, adapters, gates): 1.00 * base_lr
    """
    v4_new_prefixes = (
        "restoration_decoder.affine_head_256",
        "restoration_decoder.fullres_refiner",
        "segmentation_decoder.context_64",
        "segmentation_decoder.context_gate",
        "segmentation_decoder.rest_adapter_256",
        "segmentation_decoder.rest_gate",
        "segmentation_decoder.aux_boundary_head",
        "segmentation_decoder.mb_lat_32",
        "segmentation_decoder.mb_lat_64",
        "segmentation_decoder.mb_lat_128",
        "segmentation_decoder.mb_lat_256",
        "segmentation_decoder.mb_gate_32",
        "segmentation_decoder.mb_gate_64",
        "segmentation_decoder.mb_gate_128",
        "segmentation_decoder.mb_gate_256",
    )

    backbone_params = []
    v2_decoder_params = []
    v4_new_params = []

    for name, param in model.named_parameters():
        if name.startswith(("backbone.", "fwb.", "imagenet_encoder.")):
            backbone_params.append(param)
        elif name.startswith(v4_new_prefixes):
            v4_new_params.append(param)
        else:
            v2_decoder_params.append(param)

    param_groups = [
        {"name": "encoders", "params": backbone_params, "lr": base_lr * 0.35, "weight_decay": weight_decay},
        {"name": "v2_decoders", "params": v2_decoder_params, "lr": base_lr * 0.50, "weight_decay": weight_decay},
        {"name": "v4_new_heads", "params": v4_new_params, "lr": base_lr * 1.00, "weight_decay": weight_decay},
    ]
    return optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def set_stage_freeze_state(
    model: EURSNetV4,
    epoch: int,
    freeze_epochs: int,
    lock_restoration: bool = True,
) -> str:
    """
    Configure parameter requires_grad and BatchNorm running-stat protection per stage:
    - When `lock_restoration=True` (Path 2 default):
      `fwb`, `backbone`, and `restoration_decoder` are strictly frozen and placed in `.eval()`
      so restoration (`Q_rest = 0.4029`, `PSNR = 18.61 dB`, `Sharp = 0.1374`, `Clip = 1.94%`)
      is 100% invariant and immune to Copy-Paste augmentation.
    - Pretrained `BatchNorm2d` running statistics in `imagenet_encoder` and V2 `segmentation_decoder`
      core (`smooth_64`, `smooth_128`, `small_obj_neck`, `boundary_refine_512`, `dw_boundary_512`)
      are kept in `.eval()` mode so 4-image batches never corrupt pre-learned running_mean/running_var.
    """
    v4_seg_new_prefixes = (
        "segmentation_decoder.context_64",
        "segmentation_decoder.context_gate",
        "segmentation_decoder.rest_adapter_256",
        "segmentation_decoder.rest_gate",
        "segmentation_decoder.aux_boundary_head",
        "segmentation_decoder.mb_lat_32",
        "segmentation_decoder.mb_lat_64",
        "segmentation_decoder.mb_lat_128",
        "segmentation_decoder.mb_lat_256",
        "segmentation_decoder.mb_gate_32",
        "segmentation_decoder.mb_gate_64",
        "segmentation_decoder.mb_gate_128",
        "segmentation_decoder.mb_gate_256",
    )

    if epoch <= freeze_epochs:
        for name, param in model.named_parameters():
            if name.startswith("imagenet_encoder.") or name.startswith(v4_seg_new_prefixes):
                param.requires_grad = True
            elif (not lock_restoration) and name.startswith("restoration_decoder."):
                param.requires_grad = True
            else:
                param.requires_grad = False
        stage_str = (
            f"STAGE_1_WARMUP (epoch {epoch}/{freeze_epochs}: ImageNet encoder + V4/Path-2 adapters active; "
            f"V2 seg core & restoration locked)"
        )
    else:
        for name, param in model.named_parameters():
            if lock_restoration and name.startswith(("fwb.", "backbone.", "restoration_decoder.")):
                param.requires_grad = False
            else:
                param.requires_grad = True
        stage_str = (
            f"STAGE_2_PATH2_SEG (epoch {epoch}: ImageNet encoder + full Segmentation Decoder active; "
            f"restoration locked={lock_restoration})"
        )

    # Always lock running stats of frozen / pre-trained BatchNorm layers to prevent batch_size=4 drift
    model.fwb.eval()
    model.backbone.eval()
    if lock_restoration:
        model.restoration_decoder.eval()

    for m in model.imagenet_encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    for sub_name in ("smooth_64", "smooth_128", "small_obj_neck", "boundary_refine_512", "dw_boundary_512"):
        sub_mod = getattr(model.segmentation_decoder, sub_name, None)
        if sub_mod is not None:
            for m in sub_mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    return stage_str


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int = 2,
    total_epochs: int = 10,
    min_lr_ratio: float = 0.02,
) -> LambdaLR:
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train_v4(args: argparse.Namespace) -> Dict[str, Any]:
    verify_protected_checkpoints()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        if args.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            target_amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            target_amp_dtype = torch.float16
            use_scaler = True
    else:
        target_amp_dtype = torch.float32
        use_scaler = False

    eff_workers = 0 if device.type == 'cpu' else args.num_workers
    eff_persistent = False if eff_workers == 0 else (args.num_workers > 0)
    eff_pin_memory = (device.type == 'cuda')

    if device.type == 'cpu' and args.dry_run and not getattr(args, '_target_size_explicit', False):
        args.target_size = 256

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 75)
    print("             EURS-NET V4 (PATH 2 DUAL-ENCODER) TRAINING ENGINE")
    print("=" * 75)
    print(f"  Target Device:        {device}")
    print(f"  Batch Size:           {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"  DataLoader Workers:   {eff_workers}")
    print(f"  Sampler:              {args.sampler}")
    print(f"  Base Learning Rate:   {args.lr} (0.35x encoders, 0.50x V2 seg, 1.0x V4/Path-2 heads)")
    print(f"  Total Epochs:         {args.epochs} (Stage 1 Adapter Warmup: {args.freeze_epochs} epochs)")
    print(f"  Target Resolution:    {args.target_size}x{args.target_size}")
    print(f"  Copy-Paste Aug:       {args.use_copy_paste} (paste_prob={args.paste_prob}, rot90_prob=0.0, all_scenes={args.train_all_scenes})")
    print(f"  Lock Restoration:     {args.lock_restoration}")
    print(f"  Mixed Precision AMP:  {use_amp} (model dtype: {target_amp_dtype}, loss dtype: float32)")
    print(f"  Warm-Start Ckpt:      {args.warmstart}")
    print(f"  Checkpoint Dir:       {args.checkpoint_dir}")
    print("=" * 75 + "\n")

    # 1. Dataset & DataLoaders with Rare-Class Copy-Paste & Epoch-Varying Augmentation Seeds
    cp_aug = (
        RareClassCopyPasteAugmenter(
            data_dir=args.data_dir,
            manifest_path=args.manifest,
            paste_prob=args.paste_prob,
            max_pastes_per_image=2,
            base_seed=args.seed,
            include_all_scenes=args.train_all_scenes,
        )
        if (args.augment and args.use_copy_paste and not args.dry_run)
        else None
    )

    tx = (
        SynchronizedTransform(
            target_size=(args.target_size, args.target_size),
            scale_range=(0.65, 1.0),
            rot90_prob=0.0,
            base_seed=args.seed,
            copy_paste_augmenter=cp_aug,
        )
        if args.augment
        else None
    )

    train_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='all' if args.train_all_scenes else 'train',
        manifest_path=args.manifest,
        transform=tx,
        target_size=(args.target_size, args.target_size),
        base_seed=args.seed,
    )
    val_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='val',
        manifest_path=args.manifest,
        transform=None,
        target_size=(args.target_size, args.target_size),
        base_seed=args.seed,
    )

    if args.sampler == 'scene_balanced':
        train_sampler = SceneBalancedSampler(train_dataset, seed=args.seed)
    else:
        train_sampler = StandardSampler(train_dataset, shuffle=True, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=eff_workers,
        pin_memory=eff_pin_memory,
        persistent_workers=eff_persistent,
    )
    if args.dry_run and len(val_dataset) >= 800:
        dry_val_indices = [0, 50, 200, 250, 400, 450, 600, 650]
        val_subset = torch.utils.data.Subset(val_dataset, dry_val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=eff_workers,
            pin_memory=eff_pin_memory,
            persistent_workers=eff_persistent,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=eff_workers,
            pin_memory=eff_pin_memory,
            persistent_workers=eff_persistent,
        )

    # 2. Model Initialization & V4/V2 Champion Warm-Start
    model = EURSNetV4(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=True).to(device)

    best_metrics = None
    best_score = -1.0
    start_epoch = 1
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        optimizer = build_v4_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr_ratio=args.min_lr / args.lr,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(use_scaler and device.type == 'cuda'))
        ckpt_meta = load_checkpoint(
            filepath=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            map_location=str(device),
        )
        start_epoch = ckpt_meta.get("epoch", 0) + 1
        global_step = ckpt_meta.get("global_step", 0)
        best_score = ckpt_meta.get("best_score", -1.0)
        print(f"[RESUME] Resumed V4 checkpoint from {args.resume} at epoch {start_epoch}, best score {best_score:.4f}")
    else:
        if args.warmstart and os.path.exists(args.warmstart):
            ws_info = model.load_v2_champion_weights(args.warmstart, strict_v2_keys=True)
            print(
                f"[WARM-START] Loaded {ws_info['transferred_tensors']} tensors (0 skipped) from checkpoint: "
                f"{args.warmstart}"
            )
        optimizer = build_v4_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr_ratio=args.min_lr / args.lr,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(use_scaler and device.type == 'cuda'))

    # 3. Strict FP32 V4 Multi-Task Loss Objective with Damped 4.4x Class Weights
    alpha = torch.tensor([0.50, 0.70, 1.40, 2.20, 2.20], dtype=torch.float32, device=device)
    eff_weight_rest = 0.0 if args.lock_restoration else args.weight_rest

    loss_fn = V4MultiTaskLoss(
        alpha=alpha,
        weight_seg=args.weight_seg,
        weight_rest=eff_weight_rest,
        lovasz_start_epoch=args.lovasz_start_epoch,
        lovasz_max_weight=args.lovasz_max_weight,
        weight_boundary=args.weight_boundary,
    ).to(device)

    # Optional Epoch 0 baseline verification before opening ImageNet gates
    history: List[Dict[str, Any]] = []
    if args.eval_epoch0 and start_epoch == 1:
        max_val_batches = 2 if args.dry_run else None
        print("[Epoch 0] Evaluating warm-started EURSNetV4 baseline prior to gradient steps...")
        ep0_metrics = evaluate_v4_model(model, val_loader, device, max_batches=max_val_batches)
        best_metrics = ep0_metrics
        best_score = ep0_metrics['composite_official_surrogate']
        print(
            f"  [Epoch 0 Baseline] Official Composite: {ep0_metrics['composite_official_surrogate']:.4f} "
            f"(std0.50: {ep0_metrics['composite_official_std050']:.4f}) | "
            f"Q_seg: {ep0_metrics['Q_seg']:.4f} (mIoU: {ep0_metrics['mIoU']:.4f}, Shark: {ep0_metrics['shark_iou']:.4f}) | "
            f"Q_rest: {ep0_metrics['Q_rest']:.4f} (PSNR: {ep0_metrics['PSNR_dB']:.2f} dB, SSIM: {ep0_metrics['SSIM']:.4f}, "
            f"Sharp: {ep0_metrics['sharpness_ratio']:.4f}, Clip: {ep0_metrics['clipping_aggregate_pct']:.2f}%)"
        )
        history.append({"epoch": 0, "stage": "WARMSTART_BASELINE", "val_metrics": ep0_metrics})

    # Open ImageNet lateral gates slightly so both ImageNet encoder and lateral adapters receive gradients from Step 1
    model.enable_imagenet_gates(init_val=0.05)

    epochs_to_run = 1 if args.dry_run else args.epochs

    for epoch in range(start_epoch, epochs_to_run + 1):
        ep_t0 = time.time()
        train_dataset.set_epoch(epoch)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        loss_fn.set_epoch(epoch)

        model.train()
        stage_desc = set_stage_freeze_state(
            model,
            epoch=epoch,
            freeze_epochs=args.freeze_epochs,
            lock_restoration=args.lock_restoration,
        )
        print(f"\n>>> Epoch {epoch}/{epochs_to_run} | {stage_desc}")

        total_loss = 0.0
        steps_in_epoch = 0
        optimizer.zero_grad()

        for batch_idx, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            global_step += 1
            steps_in_epoch += 1
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=target_amp_dtype, enabled=use_amp):
                restored_rgb, seg_logits, boundary_logits_512 = model.forward_with_aux(deg)

            # Loss runs strictly in FP32 outside autocast
            loss, l_dict = loss_fn(
                restored_rgb.float(),
                seg_logits.float(),
                clean.float(),
                mask,
                boundary_logits=boundary_logits_512.float(),
            )
            loss_scaled = loss / args.grad_accum

            assert torch.isfinite(loss), f"Loss diverged to NaN/Inf at epoch {epoch}, batch {batch_idx}: {l_dict}"

            if use_scaler:
                scaler.scale(loss_scaled).backward()
                if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += float(loss.item())

            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) or (args.dry_run and batch_idx == 2):
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                print(
                    f"  [Ep {epoch:02d}/{epochs_to_run:02d}][Batch {batch_idx:03d}/{len(train_loader):03d}] "
                    f"LR(enc/v2/v4): {lrs[0]:.1e}/{lrs[1]:.1e}/{lrs[2]:.1e} | "
                    f"Loss: {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f}, "
                    f"Bnd: {l_dict['loss_boundary']:.4f}, Lovasz: {l_dict['loss_lovasz']:.4f})"
                )

            if args.dry_run and batch_idx >= 2:
                if batch_idx % args.grad_accum != 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                        optimizer.step()
                    optimizer.zero_grad()
                break

        scheduler.step()
        avg_train_loss = total_loss / max(1, steps_in_epoch)

        # 4. Validation & Guardrail Checkpointing
        max_val_batches = 2 if args.dry_run else None
        val_metrics = evaluate_v4_model(model, val_loader, device, max_batches=max_val_batches)
        ep_elapsed = time.time() - ep_t0

        print("-" * 75)
        print(
            f"[Epoch {epoch:02d} Summary] Time: {ep_elapsed:.1f}s | Train Loss: {avg_train_loss:.4f}\n"
            f"  Official Composite (Surrogate NR): {val_metrics['composite_official_surrogate']:.4f} | "
            f"Official Composite (NR=0.50): {val_metrics['composite_official_std050']:.4f}\n"
            f"  Segmentation: Q_seg={val_metrics['Q_seg']:.4f} | mIoU={val_metrics['mIoU']:.4f} | "
            f"macro_F1={val_metrics['macro_F1']:.4f} | CovPen={val_metrics['coverage_penalty']:.1f}\n"
            f"    Per-Class IoU: Bg={val_metrics['per_class']['Background']['iou']:.4f}, "
            f"Coral={val_metrics['per_class']['Coral']['iou']:.4f}, "
            f"Fish={val_metrics['per_class']['Fish']['iou']:.4f}, "
            f"Shark={val_metrics['per_class']['Shark']['iou']:.4f}, "
            f"Other={val_metrics['per_class']['Other']['iou']:.4f}\n"
            f"  Restoration:  Q_rest={val_metrics['Q_rest']:.4f} (per-img: {val_metrics['Q_rest_per_image']:.4f}) | "
            f"PSNR={val_metrics['PSNR_dB']:.2f} dB (norm: {val_metrics['PSNR_norm']:.4f}) | "
            f"SSIM={val_metrics['SSIM']:.4f} | Sharp={val_metrics['sharpness_ratio']:.4f}\n"
            f"  Clipping:     Agg={val_metrics['clipping_aggregate_pct']:.3f}% "
            f"(R={val_metrics['clipping_channels_pct']['R']:.2f}%, "
            f"G={val_metrics['clipping_channels_pct']['G']:.2f}%, "
            f"B={val_metrics['clipping_channels_pct']['B']:.2f}%) | "
            f"Q_NR={val_metrics['Q_NR_surrogate']:.4f}"
        )
        print("-" * 75)

        latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_v4_checkpoint.pt")
        save_checkpoint(
            filepath=latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            extra_state=vars(args),
        )

        is_best = is_better_v4_checkpoint(val_metrics, best_metrics)
        if is_best:
            best_metrics = val_metrics
            best_score = val_metrics['composite_official_surrogate']
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_v4_checkpoint.pt")
            save_checkpoint(
                filepath=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                extra_state=vars(args),
            )
            print(f"  [BEST V4 CHECKPOINT SAVED] Epoch {epoch} -> {best_ckpt_path} (Composite: {best_score:.4f})")

        history.append({
            "epoch": epoch,
            "stage": stage_desc,
            "train_loss": avg_train_loss,
            "is_best": is_best,
            "val_metrics": val_metrics,
        })

        log_json_path = os.path.join(args.log_dir, "train_v4_history.json")
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    verify_protected_checkpoints()
    return {
        "best_score": best_score,
        "best_metrics": best_metrics,
        "history": history,
    }


def parse_v4_args(argv=None) -> argparse.Namespace:
    default_data_dir = "data/cached_1024" if os.path.exists("data/cached_1024") else str(DATA_DIR)
    default_warmstart = (
        "checkpoints/v4/best_v4_checkpoint.pt"
        if os.path.exists("checkpoints/v4/best_v4_checkpoint.pt")
        else "experiments/exp_controlled_restoration_v2/checkpoints/best_controlled_checkpoint.pt"
    )
    parser = argparse.ArgumentParser(description="EURS-Net V4 (Path 2 Dual-Encoder) Production Training Script")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--manifest", type=str, default=str(FROZEN_MANIFEST_PATH))
    parser.add_argument(
        "--warmstart",
        type=str,
        default=default_warmstart,
        help="Path to V4 or V2 Champion checkpoint for 100%% key-compatible warm-start",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to V4 checkpoint to resume training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v4")
    parser.add_argument("--log_dir", type=str, default="logs/v4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze_epochs", type=int, default=1, help="Stage 1 warmup epochs with frozen V2 seg core")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampler", type=str, default="scene_balanced", choices=["scene_balanced", "standard"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.add_argument("--use_copy_paste", action="store_true", default=True)
    parser.add_argument("--no_copy_paste", action="store_false", dest="use_copy_paste")
    parser.add_argument("--paste_prob", type=float, default=0.75)
    parser.add_argument("--train_all_scenes", action="store_true", default=False, help="Train on all 20 scenes for final competition submission")
    parser.add_argument("--lock_restoration", action="store_true", default=True)
    parser.add_argument("--unlock_restoration", action="store_false", dest="lock_restoration")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--weight_seg", type=float, default=1.0)
    parser.add_argument("--weight_rest", type=float, default=0.6)
    parser.add_argument("--lovasz_start_epoch", type=int, default=2)
    parser.add_argument("--lovasz_max_weight", type=float, default=0.25)
    parser.add_argument("--weight_boundary", type=float, default=0.15)
    parser.add_argument("--eval_epoch0", action="store_true", default=True)
    parser.add_argument("--no_eval_epoch0", action="store_false", dest="eval_epoch0")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2-batch smoke test without full training")

    args = parser.parse_args(argv)
    args._target_size_explicit = any(a.startswith("--target_size") for a in (argv or sys.argv[1:]))
    return args


if __name__ == "__main__":
    cli_args = parse_v4_args()
    train_v4(cli_args)
