"""
Synchronized Spatial Augmentations for EURS-Net V2
Implements parameter-generator API ensuring 100% geometric synchrony
across degraded, clean, and mask triplets.
"""

from typing import Tuple, Dict, Any, Optional
import random
import cv2
import numpy as np


class SynchronizedTransform:
    """
    Synchronized Spatial Transformation Pipeline:
    - Order: HorizontalFlip -> VerticalFlip/Rot90 -> Multi-Scale RandomCrop -> Resize to Target Size
    - Generates spatial parameters once per sample using an explicit epoch-varying seed/generator.
    - Applies identical coordinates to degraded, clean, and mask.
    - Images use cv2.INTER_LINEAR (bilinear).
    - Masks use cv2.INTER_NEAREST (preserving discrete integer class IDs {0, 1, 2, 3, 4}).
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        flip_prob: float = 0.5,
        crop_size: Optional[Tuple[int, int]] = (512, 512),
        crop_prob: float = 0.5,
        scale_range: Optional[Tuple[float, float]] = (0.5, 1.0),
        rot90_prob: float = 0.0,
        base_seed: int = 42,
        copy_paste_augmenter: Optional[Any] = None,
    ):
        self.target_size = target_size
        self.flip_prob = flip_prob
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.scale_range = scale_range
        self.rot90_prob = rot90_prob
        self.base_seed = base_seed
        self.copy_paste_augmenter = copy_paste_augmenter
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch so per-sample seeds vary deterministically across epochs."""
        self.epoch = int(epoch)
        if self.copy_paste_augmenter is not None and hasattr(self.copy_paste_augmenter, 'set_epoch'):
            self.copy_paste_augmenter.set_epoch(self.epoch)

    def generate_spatial_params(
        self,
        img_h: int,
        img_w: int,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """Generate deterministic transformation parameters for a sample."""
        rand = rng if rng is not None else random

        do_flip = rand.random() < self.flip_prob
        rot_k = rand.choice([1, 2, 3]) if (self.rot90_prob > 0.0 and rand.random() < self.rot90_prob) else 0
        do_crop = (rand.random() < self.crop_prob) if self.crop_size is not None else False

        eff_h, eff_w = (img_w, img_h) if (rot_k % 2 == 1) else (img_h, img_w)

        crop_box = None
        if do_crop and self.crop_size is not None:
            if self.scale_range is not None:
                s_min, s_max = self.scale_range
                scale = rand.uniform(s_min, s_max)
                ch = max(64, min(eff_h, int(round(eff_h * scale))))
                cw = max(64, min(eff_w, int(round(eff_w * scale))))
            else:
                ch, cw = self.crop_size
            if eff_h >= ch and eff_w >= cw:
                top = rand.randint(0, eff_h - ch)
                left = rand.randint(0, eff_w - cw)
                crop_box = (top, left, ch, cw)

        return {
            'do_flip': do_flip,
            'rot_k': rot_k,
            'do_crop': do_crop,
            'crop_box': crop_box,
            'target_size': self.target_size
        }

    def apply_spatial_params(
        self,
        img: np.ndarray,
        params: Dict[str, Any],
        is_mask: bool = False
    ) -> np.ndarray:
        """Apply generated spatial parameters to a single tensor/array."""
        out = img.copy()
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        # 1. Horizontal Flip
        if params['do_flip']:
            out = cv2.flip(out, 1)

        # 2. Discrete 90-degree rotation (exact pixel grid, no interpolation blur)
        rot_k = params.get('rot_k', 0)
        if rot_k > 0:
            out = np.ascontiguousarray(np.rot90(out, k=rot_k))

        # 3. Random Crop
        if params['do_crop'] and params['crop_box'] is not None:
            top, left, ch, cw = params['crop_box']
            out = out[top:top + ch, left:left + cw]

        # 4. Final Resize to target_size
        tw, th = params['target_size']
        if (out.shape[1], out.shape[0]) != (tw, th):
            out = cv2.resize(out, (tw, th), interpolation=interp)

        return out

    def __call__(
        self,
        degraded: np.ndarray,
        clean: np.ndarray,
        mask: np.ndarray,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_epoch = self.epoch if epoch is None else int(epoch)
        if self.copy_paste_augmenter is not None:
            degraded, clean, mask = self.copy_paste_augmenter(
                degraded, clean, mask, seed=seed, epoch=active_epoch
            )
        if seed is not None:
            effective_seed = int(self.base_seed) + active_epoch * 100003 + int(seed) * 1009
            rng = random.Random(effective_seed)
        else:
            rng = None
        h, w = degraded.shape[:2]
        params = self.generate_spatial_params(h, w, rng=rng)

        deg_aug = self.apply_spatial_params(degraded, params, is_mask=False)
        clean_aug = self.apply_spatial_params(clean, params, is_mask=False)
        mask_aug = self.apply_spatial_params(mask, params, is_mask=True)

        return deg_aug, clean_aug, mask_aug

