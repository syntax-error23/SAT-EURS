"""
EURS Multi-Task Dataset Implementation for SAT-EURS.

Loads synchronized triplets of:
  - degraded RGB image  [3, H, W] float32 in [0.0, 1.0]
  - clean GT RGB image  [3, H, W] float32 in [0.0, 1.0]
  - segmentation mask   [H, W]    int64   in {0, 1, 2, 3, 4}

Guarantees:
  - Exact scene-level pairing based on locked splits / manifest.
  - Zero scene leakage between train and validation.
  - Fails loudly with descriptive RuntimeErrors if any GT image, GT mask,
    or degraded directory is missing.
  - Applies synchronized spatial transforms identically across inputs.
  - Strictly deterministic sample indexing.
  - Dual access contract:
      1. Dict-like indexing: sample["degraded"], sample["clean"], sample["mask"], sample["scene_id"]
      2. Tuple unpacking: (deg, clean, mask, meta) = sample
      3. DataLoader collation supports both dict and tuple batch conventions.
"""

import os
import json
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map, collate

from utils.paths import get_dataset_root, get_splits_path, FROZEN_MANIFEST_PATH

DEFAULT_DATASET_ROOT = get_dataset_root()
DEFAULT_SPLITS_FILE = get_splits_path()


def find_dataset_subdirs(base_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Finds clean (or gt_images), masks (or gt_masks), and degraded subdirectories
    supporting official competition archive conventions, subdirectories, and mini_dev fallbacks.
    """
    search_dirs = [
        base_dir,
        os.path.join(base_dir, 'train'),
        os.path.join(base_dir, 'mini_dev'),
        os.path.join(os.path.dirname(base_dir), 'mini_dev') if os.path.dirname(base_dir) else base_dir,
    ]

    clean_dir, masks_dir, degraded_dir = None, None, None

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        if clean_dir is None:
            for cand in ['clean', 'gt_images']:
                p = os.path.join(d, cand)
                if os.path.isdir(p):
                    clean_dir = p
                    break
        if masks_dir is None:
            for cand in ['masks', 'gt_masks']:
                p = os.path.join(d, cand)
                if os.path.isdir(p):
                    masks_dir = p
                    break
        if degraded_dir is None:
            for cand in ['degraded']:
                p = os.path.join(d, cand)
                if os.path.isdir(p):
                    degraded_dir = p
                    break

    return clean_dir, masks_dir, degraded_dir


class DualSampleItem:
    """
    Sample container returned by EURSDataset.
    Supports both dict-key access:
        sample["degraded"], sample["clean"], sample["mask"], sample["scene_id"]
    and 4-tuple unpacking:
        deg, clean, mask, meta = sample
    """
    def __init__(self, data_dict: Dict[str, Any]):
        self._dict = data_dict

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            meta = {
                'scene_id': self._dict.get('scene_id', ''),
                'filename': self._dict.get('filename', ''),
                'clean_file': self._dict.get('clean_file', f"{self._dict.get('scene_id', '')}.png"),
                'mask_file': self._dict.get('mask_file', f"{self._dict.get('scene_id', '')}.png"),
                'degraded_file': self._dict.get('filename', ''),
            }
            tuple_elements = [
                self._dict['degraded'],
                self._dict['clean'],
                self._dict['mask'],
                meta
            ]
            return tuple_elements[key]
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __iter__(self):
        meta = {
            'scene_id': self._dict.get('scene_id', ''),
            'filename': self._dict.get('filename', ''),
            'clean_file': self._dict.get('clean_file', f"{self._dict.get('scene_id', '')}.png"),
            'mask_file': self._dict.get('mask_file', f"{self._dict.get('scene_id', '')}.png"),
            'degraded_file': self._dict.get('filename', ''),
        }
        yield self._dict['degraded']
        yield self._dict['clean']
        yield self._dict['mask']
        yield meta

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __repr__(self) -> str:
        return f"DualSampleItem(keys={list(self._dict.keys())})"


class DualBatchItem:
    """
    Collated batch container returned by DataLoader for EURSDataset.
    Supports both dict-key access:
        batch["degraded"], batch["clean"], batch["mask"], batch["valid_mask"]
    and 4-tuple unpacking:
        deg, clean, mask, meta = batch
    """
    def __init__(self, collated_dict: Dict[str, Any]):
        self._dict = collated_dict

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            meta = {
                'scene_id': self._dict.get('scene_id', []),
                'filename': self._dict.get('filename', []),
                'clean_file': self._dict.get('clean_file', []),
                'mask_file': self._dict.get('mask_file', []),
                'degraded_file': self._dict.get('filename', []),
            }
            tuple_elements = [
                self._dict['degraded'],
                self._dict['clean'],
                self._dict['mask'],
                meta
            ]
            return tuple_elements[key]
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __iter__(self):
        meta = {
            'scene_id': self._dict.get('scene_id', []),
            'filename': self._dict.get('filename', []),
            'clean_file': self._dict.get('clean_file', []),
            'mask_file': self._dict.get('mask_file', []),
            'degraded_file': self._dict.get('filename', []),
        }
        yield self._dict['degraded']
        yield self._dict['clean']
        yield self._dict['mask']
        yield meta

    def get(self, key: str, default: Any = None) -> Any:
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __repr__(self) -> str:
        return f"DualBatchItem(keys={list(self._dict.keys())})"


def _collate_dual_sample(batch: List[DualSampleItem], *, collate_fn_map=None) -> DualBatchItem:
    dicts = [item._dict if isinstance(item, DualSampleItem) else item for item in batch]
    collated = collate(dicts, collate_fn_map=collate_fn_map)
    return DualBatchItem(collated)


# Register custom collation in PyTorch collate map
default_collate_fn_map[DualSampleItem] = _collate_dual_sample


class EURSDataset(Dataset):
    """
    Harmonized EURS PyTorch Multi-Task Dataset.
    Supports both legacy Tanmay contracts (data_dir, manifest_path, target_size, tuple unpacking)
    and Shubham contracts (root_dir, splits_file, tier, dict access, valid_mask propagation).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        tier: str = "tier1",
        subsplit: Optional[str] = None,
        splits_file: Optional[str] = None,
        manifest_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = (1024, 1024),
        base_seed: int = 42,
    ):
        super().__init__()

        # Resolve dataset root directory
        resolved_root = root_dir or data_dir or DEFAULT_DATASET_ROOT
        self.root_dir = os.path.abspath(resolved_root)
        self.data_dir = self.root_dir
        self.split = split.lower().strip()
        self.tier = tier.lower().strip()
        self.subsplit = subsplit
        self.transform = transform
        self.target_size = target_size
        self.base_seed = base_seed
        self.epoch = 0
        if self.transform is not None and hasattr(self.transform, 'base_seed'):
            self.transform.base_seed = base_seed
        else:
            self.target_size = target_size

        if self.split not in ("train", "val", "all"):
            raise ValueError(f"Invalid split: '{self.split}'. Must be 'train', 'val', or 'all'.")

        # Resolve splits / manifest file path: check if default splits_file is available and valid for root_dir
        if splits_file:
            self.splits_file = splits_file
        elif manifest_path:
            self.splits_file = manifest_path
        else:
            # If root_dir is mini_dev or only contains 4 mini_dev scenes, prefer FROZEN_MANIFEST_PATH
            clean_dir_test, _, _ = find_dataset_subdirs(self.root_dir)
            if clean_dir_test and not os.path.exists(os.path.join(clean_dir_test, "fish11.png")) and os.path.exists(str(FROZEN_MANIFEST_PATH)):
                self.splits_file = str(FROZEN_MANIFEST_PATH)
            elif os.path.exists(DEFAULT_SPLITS_FILE):
                self.splits_file = str(DEFAULT_SPLITS_FILE)
            else:
                self.splits_file = str(FROZEN_MANIFEST_PATH)

        # 1. Resolve subdirectories
        clean_dir, masks_dir, degraded_dir = find_dataset_subdirs(self.root_dir)

        self.gt_images_dir = clean_dir or os.path.join(self.root_dir, "gt_images")
        self.gt_masks_dir = masks_dir or os.path.join(self.root_dir, "gt_masks")
        self.degraded_dir = degraded_dir or os.path.join(self.root_dir, "degraded")

        self.clean_dir = self.gt_images_dir
        self.masks_dir = self.gt_masks_dir

        if not os.path.isdir(self.degraded_dir):
            raise RuntimeError(f"Missing degraded directory: {self.degraded_dir}")

        # 2. Load splits manifest
        if not os.path.isfile(self.splits_file):
            raise FileNotFoundError(f"Splits manifest file not found: {self.splits_file}")

        with open(self.splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        self.scene_ids = self._resolve_scene_ids(splits_data)

        # 3. Build sample registry
        self.samples: List[Dict[str, str]] = self._build_and_validate_index()

        print(f"[EURSDataset] Initialized '{self.split}' split: {len(self.samples)} samples across {len(self.scene_ids)} scenes.")

    def _resolve_scene_ids(self, splits_data: Dict[str, Any]) -> List[str]:
        """Resolves target scene IDs from splits_data supporting both schema formats."""
        # Check for Shubham tier schema
        if "tier1" in splits_data or "tier2" in splits_data or "tier3" in splits_data:
            if self.tier == "tier1":
                tier_dict = splits_data["tier1"]
                if self.split == "all":
                    scenes = tier_dict.get("train", []) + tier_dict.get("val", [])
                else:
                    if self.split not in tier_dict:
                        raise KeyError(f"'{self.split}' not found under tier1 in {self.splits_file}")
                    scenes = tier_dict[self.split]
            elif self.tier.startswith("tier2"):
                run_key = self.subsplit or (self.tier.split("_", 1)[1] if "_" in self.tier else "run1")
                tier_dict = splits_data.get("tier2", {}).get(run_key, {})
                if self.split == "all":
                    scenes = tier_dict.get("train", []) + tier_dict.get("val", [])
                else:
                    if self.split not in tier_dict:
                        raise KeyError(f"'{self.split}' not found under tier2/{run_key} in {self.splits_file}")
                    scenes = tier_dict[self.split]
            elif self.tier.startswith("tier3"):
                fold_key = self.subsplit or (self.tier.split("_", 1)[1] if "_" in self.tier else "fold1")
                tier_dict = splits_data.get("tier3", {}).get(fold_key, {})
                if self.split == "all":
                    scenes = tier_dict.get("train", []) + tier_dict.get("val", [])
                else:
                    if self.split not in tier_dict:
                        raise KeyError(f"'{self.split}' not found under tier3/{fold_key} in {self.splits_file}")
                    scenes = tier_dict[self.split]
            else:
                raise ValueError(f"Unsupported tier: '{self.tier}'.")
        elif "train_scenes" in splits_data and "val_scenes" in splits_data:
            train_scenes = set(splits_data["train_scenes"])
            val_scenes = set(splits_data["val_scenes"])
            assert train_scenes.isdisjoint(val_scenes), "FATAL: Data leakage detected in manifest!"
            if self.split == "train":
                scenes = list(train_scenes)
            elif self.split == "val":
                scenes = list(val_scenes)
            else:
                scenes = list(train_scenes | val_scenes)
        else:
            raise KeyError(f"Unrecognized splits manifest format in {self.splits_file}")

        if not isinstance(scenes, list) or len(scenes) == 0:
            raise ValueError(f"Resolved scene list for {self.split} is empty or invalid.")

        if len(scenes) != len(set(scenes)):
            raise ValueError(f"Duplicate scene IDs detected in split {self.split}: {scenes}")

        return sorted(scenes)

    def _build_and_validate_index(self) -> List[Dict[str, str]]:
        samples = []

        for scene_id in self.scene_ids:
            # Locate clean ground truth image
            gt_image_path = os.path.join(self.gt_images_dir, f"{scene_id}.png")
            if not os.path.isfile(gt_image_path):
                for ext in (".jpg", ".jpeg"):
                    alt = os.path.join(self.gt_images_dir, f"{scene_id}{ext}")
                    if os.path.isfile(alt):
                        gt_image_path = alt
                        break
            if not os.path.isfile(gt_image_path):
                raise RuntimeError(
                    f"Missing GT clean image for scene '{scene_id}': expected file at {gt_image_path}"
                )

            # Locate ground truth mask
            gt_mask_path = os.path.join(self.gt_masks_dir, f"{scene_id}.png")
            if not os.path.isfile(gt_mask_path):
                for ext in (".jpg", ".jpeg"):
                    alt = os.path.join(self.gt_masks_dir, f"{scene_id}{ext}")
                    if os.path.isfile(alt):
                        gt_mask_path = alt
                        break
            if not os.path.isfile(gt_mask_path):
                raise RuntimeError(
                    f"Missing GT mask for scene '{scene_id}': expected file at {gt_mask_path}"
                )

            # Locate degraded images directory
            deg_scene_dir = os.path.join(self.degraded_dir, scene_id)
            if not os.path.isdir(deg_scene_dir):
                raise RuntimeError(
                    f"Missing degraded directory for scene '{scene_id}': expected directory at {deg_scene_dir}"
                )

            deg_filenames = [
                f for f in os.listdir(deg_scene_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith(".")
            ]

            if len(deg_filenames) == 0:
                raise RuntimeError(
                    f"No degraded PNG images found for scene '{scene_id}' in {deg_scene_dir}"
                )

            deg_filenames.sort()

            for fname in deg_filenames:
                if not fname.startswith(f"{scene_id}_") and not fname.startswith(scene_id):
                    raise RuntimeError(
                        f"Malformed degraded image filename '{fname}' in scene '{scene_id}' folder; "
                        f"expected prefix '{scene_id}'"
                    )

                samples.append({
                    "degraded_path": os.path.join(deg_scene_dir, fname),
                    "clean_path": gt_image_path,
                    "mask_path": gt_mask_path,
                    "scene_id": scene_id,
                    "filename": fname,
                    "clean_file": os.path.basename(gt_image_path),
                    "mask_file": os.path.basename(gt_mask_path),
                })

        return samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.transform is not None and hasattr(self.transform, 'set_epoch'):
            self.transform.set_epoch(self.epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DualSampleItem:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.samples)}")

        record = self.samples[idx]

        # 1. Load images using PIL / cv2
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

        if not (deg_img.size == clean_img.size == mask_img.size):
            raise RuntimeError(
                f"Dimension mismatch in sample '{record['filename']}': "
                f"degraded={deg_img.size}, clean={clean_img.size}, mask={mask_img.size}"
            )

        # 2. Check if legacy numpy SynchronizedTransform is passed
        from data.transforms import SynchronizedTransform
        is_numpy_transform = (
            self.transform is not None and
            type(self.transform) is SynchronizedTransform
        )

        valid_mask = None

        if is_numpy_transform or (self.target_size is not None and self.transform is None):
            deg_np = np.array(deg_img, dtype=np.uint8)
            clean_np = np.array(clean_img, dtype=np.uint8)
            mask_np = np.array(mask_img, dtype=np.uint8)

            if self.target_size is not None:
                tw, th = self.target_size
                if (deg_np.shape[1], deg_np.shape[0]) != (tw, th):
                    deg_np = cv2.resize(deg_np, (tw, th), interpolation=cv2.INTER_LINEAR)
                    clean_np = cv2.resize(clean_np, (tw, th), interpolation=cv2.INTER_LINEAR)
                    mask_np = cv2.resize(mask_np, (tw, th), interpolation=cv2.INTER_NEAREST)

            if is_numpy_transform:
                deg_np, clean_np, mask_np = self.transform(deg_np, clean_np, mask_np, seed=idx)

            deg_tensor = torch.from_numpy(deg_np.transpose(2, 0, 1)).float() / 255.0
            clean_tensor = torch.from_numpy(clean_np.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).long()
        else:
            deg_np = np.array(deg_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            clean_np = np.array(clean_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mask_np = np.array(mask_img, dtype=np.int64)

            deg_tensor = torch.from_numpy(deg_np)
            clean_tensor = torch.from_numpy(clean_np)
            mask_tensor = torch.from_numpy(mask_np)

            if self.target_size is not None and self.transform is None:
                tw, th = self.target_size
                if (deg_tensor.shape[2], deg_tensor.shape[1]) != (tw, th):
                    from data.transforms import SynchronizedResize
                    resize_op = SynchronizedResize(size=(th, tw))
                    deg_tensor, clean_tensor, mask_tensor = resize_op(deg_tensor, clean_tensor, mask_tensor)

            if self.transform is not None:
                orig_h, orig_w = deg_tensor.shape[-2], deg_tensor.shape[-1]
                deg_tensor, clean_tensor, mask_tensor = self.transform(
                    deg_tensor, clean_tensor, mask_tensor
                )
                if hasattr(self.transform, "compute_metadata"):
                    valid_mask = self.transform.compute_metadata(orig_h, orig_w).valid_mask
                elif hasattr(self.transform, "transforms"):
                    for t in self.transform.transforms:
                        if hasattr(t, "compute_metadata"):
                            valid_mask = t.compute_metadata(orig_h, orig_w).valid_mask
                            break
                        elif hasattr(t, "last_metadata") and t.last_metadata is not None:
                            valid_mask = t.last_metadata.valid_mask
                            break
                elif hasattr(self.transform, "last_metadata") and self.transform.last_metadata is not None:
                    valid_mask = self.transform.last_metadata.valid_mask

        sample_dict = {
            "degraded": deg_tensor,
            "clean": clean_tensor,
            "mask": mask_tensor,
            "scene_id": record["scene_id"],
            "filename": record["filename"],
            "clean_file": record["clean_file"],
            "mask_file": record["mask_file"],
        }
        if valid_mask is not None:
            sample_dict["valid_mask"] = valid_mask

        return DualSampleItem(sample_dict)
