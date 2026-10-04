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
    - Order: HorizontalFlip -> RandomCrop -> Resize to Target Size
    - Generates spatial parameters once per sample using an optional explicit seed/generator.
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
    ):
        self.target_size = target_size
        self.flip_prob = flip_prob
        self.crop_size = crop_size
        self.crop_prob = crop_prob

    def generate_spatial_params(
        self,
        img_h: int,
        img_w: int,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Any]:
        """Generate deterministic transformation parameters for a sample."""
        rand = rng if rng is not None else random

        do_flip = rand.random() < self.flip_prob
        do_crop = (rand.random() < self.crop_prob) if self.crop_size is not None else False

        crop_box = None
        if do_crop and self.crop_size is not None:
            ch, cw = self.crop_size
            if img_h >= ch and img_w >= cw:
                top = rand.randint(0, img_h - ch)
                left = rand.randint(0, img_w - cw)
                crop_box = (top, left, ch, cw)

        return {
            'do_flip': do_flip,
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

        # 2. Random Crop
        if params['do_crop'] and params['crop_box'] is not None:
            top, left, ch, cw = params['crop_box']
            out = out[top:top + ch, left:left + cw]

        # 3. Final Resize to target_size
        tw, th = params['target_size']
        if (out.shape[1], out.shape[0]) != (tw, th):
            out = cv2.resize(out, (tw, th), interpolation=interp)

        return out

    def __call__(
        self,
        degraded: np.ndarray,
        clean: np.ndarray,
        mask: np.ndarray,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = random.Random(seed) if seed is not None else None
        h, w = degraded.shape[:2]
        params = self.generate_spatial_params(h, w, rng=rng)

        deg_aug = self.apply_spatial_params(degraded, params, is_mask=False)
        clean_aug = self.apply_spatial_params(clean, params, is_mask=False)
        mask_aug = self.apply_spatial_params(mask, params, is_mask=True)

        return deg_aug, clean_aug, mask_aug
