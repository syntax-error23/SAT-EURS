"""
Rare-Class Object Copy-Paste (Instance CutMix) Augmentation for EURS-Net V4 (Path 2)
Strictly extracts foreground object instances ONLY from the 16 frozen training scenes:
  - Class 3 (Elasmobranchii / Shark & Ray): extracted from `ray1`
  - Class 4 (Other / benthic & marine fauna): extracted from `starfish2` and `turtles1`
  - Class 2 (Fish / Actinopterygii): extracted from training fish scenes

Solves the two fatal scene-level co-occurrence bottlenecks discovered in confusion-matrix analysis:
  1. In the 16 training scenes, Class 4 (`starfish2`, `turtles1`) NEVER co-occurs with Class 1 (`Coral`),
     causing the network to misclassify 53% of `seahorse1` (Class 4 next to Coral) as Coral and 19% as Fish,
     while misclassifying 243K pixels of `sharkFish` (blue water + Coral) as Class 4.
  2. In the 16 training scenes, Class 3 (`ray1`) only appears on flat grey sand, causing the network
     to miss 65% of the swimming shark in `sharkFish`.

By pasting scale-, aspect-, and color-adapted Class 3 and Class 4 instances onto the 13 Coral+Fish
training scenes, every scene layout `{0, 1, 2, 3, 4}` co-occurs during training.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


class RareClassCopyPasteAugmenter:
    """
    Extracts and caches foreground object instances from the training split and pastes them
    onto training samples with target-scene color/turbidity adaptation and aspect diversity.
    """
    def __init__(
        self,
        data_dir: str = "data/cached_1024",
        manifest_path: str = "splits/manifest_full.json",
        paste_prob: float = 0.75,
        max_pastes_per_image: int = 2,
        base_seed: int = 42,
        include_all_scenes: bool = False,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.paste_prob = paste_prob
        self.max_pastes_per_image = max_pastes_per_image
        self.base_seed = base_seed
        self.include_all_scenes = include_all_scenes
        self.epoch = 0

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        val_scenes = set(manifest["val_scenes"])
        assert set(manifest["train_scenes"]).isdisjoint(val_scenes), "FATAL: Copy-Paste train/val leakage!"
        if include_all_scenes:
            self.train_scenes = sorted(list(set(manifest["train_scenes"]) | val_scenes))
        else:
            # Strictly use ONLY train_scenes (zero validation leakage)
            self.train_scenes = sorted(list(manifest["train_scenes"]))

        self.clean_dir = os.path.join(self.data_dir, "clean")
        self.masks_dir = os.path.join(self.data_dir, "masks")
        self.degraded_dir = os.path.join(self.data_dir, "degraded")

        # Instance bank indexed by class_id: {2: [...], 3: [...], 4: [...]}
        self.instance_bank: Dict[int, List[Dict[str, np.ndarray]]] = {2: [], 3: [], 4: []}
        self._build_instance_bank()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_instance_bank(self) -> None:
        """
        Extract foreground connected components and full-class masks from the 16 training scenes.
        Stores clean RGB crop, a representative degraded RGB crop, and binary instance mask.
        """
        for scene_id in self.train_scenes:
            mask_path = os.path.join(self.masks_dir, f"{scene_id}.png")
            clean_path = os.path.join(self.clean_dir, f"{scene_id}.png")
            scene_deg_dir = os.path.join(self.degraded_dir, scene_id)
            if not (os.path.exists(mask_path) and os.path.exists(clean_path) and os.path.exists(scene_deg_dir)):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            if mask is None or clean_bgr is None:
                continue
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

            deg_files = sorted([f for f in os.listdir(scene_deg_dir) if f.lower().endswith((".png", ".jpg"))])
            if not deg_files:
                continue
            # Sample up to 5 degraded lighting variants per rare scene for rich appearance diversity
            sample_deg_files = deg_files[::max(1, len(deg_files) // 5)][:5]
            deg_rgbs = []
            for df in sample_deg_files:
                d_bgr = cv2.imread(os.path.join(scene_deg_dir, df), cv2.IMREAD_COLOR)
                if d_bgr is not None:
                    d_rgb = cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB)
                    if d_rgb.shape[:2] != mask.shape[:2]:
                        d_rgb = cv2.resize(d_rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                    deg_rgbs.append(d_rgb)

            if clean_rgb.shape[:2] != mask.shape[:2]:
                clean_rgb = cv2.resize(clean_rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

            for target_cls in (3, 4, 2):
                cls_bin = (mask == target_cls).astype(np.uint8)
                total_area = int(cls_bin.sum())
                if total_area < 400:
                    continue

                # For rare classes 3 and 4, extract both individual connected components AND full union
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_bin, connectivity=8)
                extracted_boxes: List[Tuple[int, int, int, int, np.ndarray]] = []

                for lbl in range(1, num_labels):
                    x, y, w, h, area = stats[lbl]
                    if area < 500:
                        continue
                    comp_mask = (labels[y:y+h, x:x+w] == lbl).astype(np.uint8)
                    extracted_boxes.append((x, y, w, h, comp_mask))

                # Also add the full bounding box of the class for classes 3 and 4
                if target_cls in (3, 4):
                    ys, xs = np.where(cls_bin > 0)
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    extracted_boxes.append((x0, y0, x1 - x0, y1 - y0, cls_bin[y0:y1, x0:x1]))

                # For class 2 (Fish), keep only the largest 2 components per scene to keep bank compact
                if target_cls == 2:
                    extracted_boxes = sorted(extracted_boxes, key=lambda b: int(b[4].sum()), reverse=True)[:1]

                for (x, y, w, h, comp_mask) in extracted_boxes:
                    c_crop = clean_rgb[y:y+h, x:x+w].copy()
                    # For classes 3 & 4, store multiple degraded variants; for class 2, store 1
                    variants = deg_rgbs if target_cls in (3, 4) else deg_rgbs[:1]
                    for d_rgb in variants:
                        d_crop = d_rgb[y:y+h, x:x+w].copy()
                        self.instance_bank[target_cls].append({
                            "scene_id": scene_id,
                            "class_id": target_cls,
                            "clean_crop": c_crop,
                            "deg_crop": d_crop,
                            "mask_crop": comp_mask,
                        })

        print(
            f"[RareClassCopyPaste] Initialized strictly from 16 train scenes: "
            f"Class 3 (Shark/Ray)={len(self.instance_bank[3])} instances, "
            f"Class 4 (Other)={len(self.instance_bank[4])} instances, "
            f"Class 2 (Fish)={len(self.instance_bank[2])} instances."
        )

    @staticmethod
    def _adapt_color_to_target(
        obj_deg: np.ndarray,
        obj_clean: np.ndarray,
        obj_mask: np.ndarray,
        target_deg_patch: np.ndarray,
        rng: random.Random,
    ) -> np.ndarray:
        """
        Blend the pasted degraded object's color statistics toward the target scene's local
        underwater ambient light so the model learns object morphology/texture rather than
        scene-specific lighting seams.
        """
        valid = obj_mask > 0
        if not np.any(valid):
            return obj_deg

        out = obj_deg.astype(np.float32)
        tgt = target_deg_patch.astype(np.float32)

        # Compute per-channel mean/std of target patch and object
        tgt_mean = tgt.mean(axis=(0, 1), keepdims=True)
        tgt_std = tgt.std(axis=(0, 1), keepdims=True) + 1e-5

        obj_pixels = out[valid]
        obj_mean = obj_pixels.mean(axis=0, keepdims=True).reshape(1, 1, 3)
        obj_std = obj_pixels.std(axis=0, keepdims=True).reshape(1, 1, 3) + 1e-5

        # Normalized color transfer with randomized blend factor in [0.35, 0.75]
        blend = rng.uniform(0.35, 0.75)
        transferred = (out - obj_mean) * (tgt_std / obj_std).clip(0.5, 2.0) + tgt_mean
        adapted = (1.0 - blend) * out + blend * transferred
        return np.clip(adapted, 0.0, 255.0).astype(np.uint8)

    def __call__(
        self,
        degraded: np.ndarray,
        clean: np.ndarray,
        mask: np.ndarray,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_epoch = self.epoch if epoch is None else int(epoch)
        eff_seed = self.base_seed + active_epoch * 100003 + (0 if seed is None else int(seed) * 1009) + 777
        rng = random.Random(eff_seed)

        if rng.random() > self.paste_prob:
            return degraded, clean, mask

        H, W = mask.shape[:2]
        out_deg = degraded.copy()
        out_clean = clean.copy()
        out_mask = mask.copy()

        num_pastes = rng.randint(1, self.max_pastes_per_image)
        for _ in range(num_pastes):
            # Prioritize rare classes: 45% Class 4 (Other), 40% Class 3 (Shark/Ray), 15% Class 2 (Fish)
            r = rng.random()
            if r < 0.45 and len(self.instance_bank[4]) > 0:
                cls_id = 4
            elif r < 0.85 and len(self.instance_bank[3]) > 0:
                cls_id = 3
            elif len(self.instance_bank[2]) > 0:
                cls_id = 2
            else:
                continue

            inst = rng.choice(self.instance_bank[cls_id])
            c_crop = inst["clean_crop"].copy()
            d_crop = inst["deg_crop"].copy()
            m_crop = inst["mask_crop"].copy()

            # Random horizontal flip
            if rng.random() < 0.5:
                c_crop = cv2.flip(c_crop, 1)
                d_crop = cv2.flip(d_crop, 1)
                m_crop = cv2.flip(m_crop, 1)

            # Mild rotation (-25 to +25 deg) for natural swimming/perching pose diversity
            angle = rng.uniform(-25.0, 25.0)
            ih, iw = m_crop.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((iw / 2.0, ih / 2.0), angle, 1.0)
            c_crop = cv2.warpAffine(c_crop, rot_mat, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            d_crop = cv2.warpAffine(d_crop, rot_mat, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            m_crop = cv2.warpAffine(m_crop, rot_mat, (iw, ih), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Target size distribution matched to validation foreground object scales:
            # - Class 4 (Other, e.g., seahorse1 ~318K px): macro foreground height in [240, 680], vertical bias
            # - Class 3 (Shark, e.g., sharkFish ~48K px): medium-large width in [220, 560], horizontal bias
            # - Class 2 (Fish): width in [120, 380]
            if cls_id == 4:
                new_h = rng.randint(240, min(680, int(H * 0.72)))
                aspect_hw = rng.uniform(0.85, 1.65)  # vertical/upright bias matching seahorse1
                new_w = max(140, min(int(W * 0.70), int(new_h / aspect_hw)))
            elif cls_id == 3:
                new_w = rng.randint(220, min(580, int(W * 0.65)))
                aspect_wh = rng.uniform(1.10, 2.10)  # horizontal swimming bias matching sharkFish
                new_h = max(110, min(int(H * 0.55), int(new_w / aspect_wh)))
            else:
                new_w = rng.randint(120, min(380, int(W * 0.45)))
                new_h = max(90, min(int(H * 0.45), int(new_w * rng.uniform(0.65, 1.25))))

            c_resized = cv2.resize(c_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            d_resized = cv2.resize(d_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            m_resized = cv2.resize(m_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if int(m_resized.sum()) < 200:
                continue

            # Choose placement location (y0, x0)
            max_y = max(0, H - new_h)
            max_x = max(0, W - new_w)
            y0 = rng.randint(0, max_y) if max_y > 0 else 0
            x0 = rng.randint(0, max_x) if max_x > 0 else 0

            target_deg_roi = out_deg[y0:y0+new_h, x0:x0+new_w]
            target_clean_roi = out_clean[y0:y0+new_h, x0:x0+new_w]
            target_mask_roi = out_mask[y0:y0+new_h, x0:x0+new_w]

            # Adapt degraded object color toward target scene's local water/reef region
            d_adapted = self._adapt_color_to_target(d_resized, c_resized, m_resized, target_deg_roi, rng)

            # Soft 3x3 Gaussian alpha feathering for RGB, strict hard binary mask for segmentation target
            alpha = cv2.GaussianBlur(m_resized.astype(np.float32), (5, 5), 1.2)[..., None]
            out_deg[y0:y0+new_h, x0:x0+new_w] = np.clip(
                alpha * d_adapted.astype(np.float32) + (1.0 - alpha) * target_deg_roi.astype(np.float32),
                0.0,
                255.0,
            ).astype(np.uint8)
            out_clean[y0:y0+new_h, x0:x0+new_w] = np.clip(
                alpha * c_resized.astype(np.float32) + (1.0 - alpha) * target_clean_roi.astype(np.float32),
                0.0,
                255.0,
            ).astype(np.uint8)

            binary_cond = m_resized > 0
            target_mask_roi[binary_cond] = cls_id
            out_mask[y0:y0+new_h, x0:x0+new_w] = target_mask_roi

        return out_deg, out_clean, out_mask
