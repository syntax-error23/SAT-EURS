"""
EURS Multi-Task Dataset Implementation for SAT-EURS.

Loads synchronized triplets of:
  - degraded RGB image  [3, H, W] float32 in [0.0, 1.0]
  - clean GT RGB image  [3, H, W] float32 in [0.0, 1.0]
  - segmentation mask   [H, W]    int64   in {0, 1, 2, 3, 4}

Guarantees:
  - Exact scene-level pairing based on locked splits.
  - Zero scene leakage between train and validation.
  - Fails loudly with descriptive RuntimeErrors if any GT image, GT mask,
    or degraded directory is missing.
  - Applies synchronized spatial transforms identically to all 3 elements.
  - Strictly deterministic sample indexing.
"""

import os
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = "/home/shubk/Datasets/eurs"
DEFAULT_SPLITS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "splits.json"
)


class EURSDataset(Dataset):
    """
    PyTorch Dataset for the EURS Underwater Restoration & Segmentation Challenge.

    Args:
        root_dir (str): Root path to extracted EURS dataset (e.g., '/home/shubk/Datasets/eurs').
        split (str): Split name ('train' or 'val').
        tier (str): Validation protocol tier ('tier1', 'tier2_run1', 'tier2_run2', etc.). Default: 'tier1'.
        splits_file (str): Path to locked splits JSON file. Default: 'data/splits.json'.
        transform (Callable, optional): Synchronized transform callable accepting
            (degraded, clean, mask) and returning (degraded, clean, mask).
        subsplit (str, optional): For Tier 2/3, specific run or fold key (e.g. 'run1', 'fold1').
    """
    def __init__(
        self,
        root_dir: str = DEFAULT_DATASET_ROOT,
        split: str = "train",
        tier: str = "tier1",
        splits_file: str = DEFAULT_SPLITS_FILE,
        transform: Optional[Callable] = None,
        subsplit: Optional[str] = None
    ):
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.split = split.lower().strip()
        self.tier = tier.lower().strip()
        self.splits_file = os.path.abspath(splits_file)
        self.transform = transform
        self.subsplit = subsplit.lower().strip() if subsplit else None

        if self.split not in ("train", "val"):
            raise ValueError(f"Split must be either 'train' or 'val', got '{split}'")

        # 1. Validate root directory structure
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"EURS dataset root directory not found: {self.root_dir}")

        train_dir = os.path.join(self.root_dir, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Missing 'train' directory under dataset root: {train_dir}")

        self.degraded_dir = os.path.join(train_dir, "degraded")
        self.gt_images_dir = os.path.join(train_dir, "gt_images")
        self.gt_masks_dir = os.path.join(train_dir, "gt_masks")

        for d_name, d_path in [
            ("degraded", self.degraded_dir),
            ("gt_images", self.gt_images_dir),
            ("gt_masks", self.gt_masks_dir),
        ]:
            if not os.path.isdir(d_path):
                raise FileNotFoundError(f"Missing expected dataset subdirectory: {d_path}")

        # 2. Load and validate splits manifest
        if not os.path.isfile(self.splits_file):
            raise FileNotFoundError(f"Splits manifest file not found: {self.splits_file}")

        with open(self.splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        self.scene_ids = self._resolve_scene_ids(splits_data)

        # 3. Build sample registry with strict validation
        self.samples: List[Dict[str, str]] = self._build_and_validate_index()

    def _resolve_scene_ids(self, splits_data: Dict[str, Any]) -> List[str]:
        """Resolves target scene IDs from splits_data according to tier and split."""
        if self.tier == "tier1":
            if "tier1" not in splits_data:
                raise KeyError(f"'tier1' not found in splits file {self.splits_file}")
            tier_dict = splits_data["tier1"]
            if self.split not in tier_dict:
                raise KeyError(f"'{self.split}' not found under tier1 in {self.splits_file}")
            scenes = tier_dict[self.split]
        elif self.tier.startswith("tier2"):
            # e.g., tier="tier2", subsplit="run1" or tier="tier2_run1"
            run_key = self.subsplit or (self.tier.split("_", 1)[1] if "_" in self.tier else "run1")
            tier_dict = splits_data.get("tier2", {}).get(run_key, {})
            if self.split not in tier_dict:
                raise KeyError(f"'{self.split}' not found under tier2/{run_key} in {self.splits_file}")
            scenes = tier_dict[self.split]
        elif self.tier.startswith("tier3"):
            fold_key = self.subsplit or (self.tier.split("_", 1)[1] if "_" in self.tier else "fold1")
            tier_dict = splits_data.get("tier3", {}).get(fold_key, {})
            if self.split not in tier_dict:
                raise KeyError(f"'{self.split}' not found under tier3/{fold_key} in {self.splits_file}")
            scenes = tier_dict[self.split]
        else:
            raise ValueError(f"Unsupported tier: '{self.tier}'. Supported: 'tier1', 'tier2', 'tier3'.")

        if not isinstance(scenes, list) or len(scenes) == 0:
            raise ValueError(f"Resolved scene list for {self.tier}/{self.split} is empty or invalid.")

        # Ensure no duplicates
        if len(scenes) != len(set(scenes)):
            raise ValueError(f"Duplicate scene IDs detected in split {self.split}: {scenes}")

        return sorted(scenes)

    def _build_and_validate_index(self) -> List[Dict[str, str]]:
        """
        Scans filesystem for all samples in target scenes.
        Fails loudly if any GT image, GT mask, or degraded variant is missing.
        """
        samples = []

        for scene_id in self.scene_ids:
            # Verify GT clean image existence
            gt_image_path = os.path.join(self.gt_images_dir, f"{scene_id}.png")
            if not os.path.isfile(gt_image_path):
                raise RuntimeError(
                    f"Missing GT clean image for scene '{scene_id}': expected file at {gt_image_path}"
                )

            # Verify GT mask existence
            gt_mask_path = os.path.join(self.gt_masks_dir, f"{scene_id}.png")
            if not os.path.isfile(gt_mask_path):
                raise RuntimeError(
                    f"Missing GT mask for scene '{scene_id}': expected file at {gt_mask_path}"
                )

            # Verify degraded directory existence
            deg_scene_dir = os.path.join(self.degraded_dir, scene_id)
            if not os.path.isdir(deg_scene_dir):
                raise RuntimeError(
                    f"Missing degraded directory for scene '{scene_id}': expected directory at {deg_scene_dir}"
                )

            # Gather and sort all degraded PNGs deterministically
            deg_filenames = [
                f for f in os.listdir(deg_scene_dir)
                if f.endswith(".png") and not f.startswith(".")
            ]

            if len(deg_filenames) == 0:
                raise RuntimeError(
                    f"No degraded PNG images found for scene '{scene_id}' in {deg_scene_dir}"
                )

            deg_filenames.sort()

            for fname in deg_filenames:
                # Basic filename consistency check: must match {scene_id}_...
                if not fname.startswith(f"{scene_id}_"):
                    raise RuntimeError(
                        f"Malformed degraded image filename '{fname}' in scene '{scene_id}' folder; "
                        f"expected prefix '{scene_id}_'"
                    )

                samples.append({
                    "degraded_path": os.path.join(deg_scene_dir, fname),
                    "clean_path": gt_image_path,
                    "mask_path": gt_mask_path,
                    "scene_id": scene_id,
                    "filename": fname
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.samples)}")

        record = self.samples[idx]

        # 1. Load images with PIL
        try:
            deg_img = Image.open(record["degraded_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable degraded image '{record['degraded_path']}': {e}") from e

        try:
            clean_img = Image.open(record["clean_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable clean image '{record['clean_path']}': {e}") from e

        try:
            mask_img = Image.open(record["mask_path"])
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable mask image '{record['mask_path']}': {e}") from e

        # Spatial consistency check
        if not (deg_img.size == clean_img.size == mask_img.size):
            raise RuntimeError(
                f"Dimension mismatch in sample '{record['filename']}': "
                f"degraded={deg_img.size}, clean={clean_img.size}, mask={mask_img.size}"
            )

        # 2. Convert to PyTorch tensors
        # Images: [C, H, W], float32 in [0.0, 1.0]
        deg_np = np.array(deg_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        clean_np = np.array(clean_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        
        # Mask: [H, W], int64
        mask_np = np.array(mask_img, dtype=np.int64)

        deg_tensor = torch.from_numpy(deg_np)
        clean_tensor = torch.from_numpy(clean_np)
        mask_tensor = torch.from_numpy(mask_np)

        # 3. Apply synchronized spatial transforms
        valid_mask = None
        if self.transform is not None:
            deg_tensor, clean_tensor, mask_tensor = self.transform(
                deg_tensor, clean_tensor, mask_tensor
            )
            if hasattr(self.transform, "last_metadata") and self.transform.last_metadata is not None:
                valid_mask = self.transform.last_metadata.valid_mask
            elif hasattr(self.transform, "transforms"):
                for t in self.transform.transforms:
                    if hasattr(t, "last_metadata") and t.last_metadata is not None:
                        valid_mask = t.last_metadata.valid_mask
                        break

        sample = {
            "degraded": deg_tensor,
            "clean": clean_tensor,
            "mask": mask_tensor,
            "scene_id": record["scene_id"],
            "filename": record["filename"]
        }
        if valid_mask is not None:
            sample["valid_mask"] = valid_mask

        return sample
