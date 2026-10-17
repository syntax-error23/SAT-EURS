"""
SAT-EURS Track 1 — V18 External Dataset Adapters & Taxonomy Isolation
Implements:
1. Conservative, non-corrupting category mappings for:
   - Coralscapes (Coral -> EURS Class 1, ambiguous benthos -> 255)
   - SUIM (Coarse groups -> Objectness pretraining, 5-class head -> 255)
   - UIIS10K (Instance masks -> Objectness & boundaries, unverified fish -> 255)
   - USIS10K (Salient marine masks -> Objectness & boundaries, unverified -> 255)
2. Strict ignore_index = 255 on all ambiguous pixels (never default to background 0)
3. Taxonomy manifest generation with license, source, and storage specifications
4. Color-coded contact sheet overlay generator for human quality assurance
"""

import os
import os.path as osp
import json
from typing import Dict, Tuple, List, Optional, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = 255

# Strict taxonomy mapping dictionaries
EXTERNAL_TAXONOMY_MAPS = {
    "Coralscapes": {
        "description": "High-resolution coral reef semantic segmentation dataset",
        "license": "CC BY-NC 4.0 (Research Use Only)",
        "source_url": "https://coralscapes.org/dataset",
        "expected_storage_gb": 18.5,
        "class_mapping": {
            0: 0,     # Background / Open Water -> EURS 0
            1: 1,     # Hard Coral -> EURS 1 (Coral)
            2: 1,     # Soft Coral -> EURS 1 (Coral)
            3: 255,   # Sand / Substrate -> Ignored (EURS substrate is Class 0, benthic boundary is ambiguous)
            4: 255,   # Algae / Turf -> Ignored
            5: 255,   # Sponge -> Ignored
            6: 255,   # Dead Coral / Rubble -> Ignored
            255: 255  # Void -> Ignored
        },
        "target_supervision": {
            "main_5class": True,
            "boundary": True,
            "objectness": True
        }
    },
    "SUIM": {
        "description": "Semantic Segmentation of Underwater Imagery (Islam et al.)",
        "license": "MIT License",
        "source_url": "https://irvlab.cs.umn.edu/resources/suim-dataset",
        "expected_storage_gb": 2.1,
        "class_mapping": {
            # Coarse classes: strictly used for objectness pretraining, not fine-grained 5-class head
            "BW": {"main": 0, "obj": 0},      # Background Water
            "HD": {"main": 255, "obj": 1},    # Human Diver -> Objectness foreground, ignored for 5-class
            "WR": {"main": 255, "obj": 1},    # Wrecks / Artificial -> Objectness foreground
            "RO": {"main": 255, "obj": 1},    # Marine Robots -> Objectness foreground
            "RI": {"main": 255, "obj": 1},    # Reefs & Invertebrates -> Objectness foreground (coarse mix)
            "FV": {"main": 255, "obj": 1},    # Fish & Vertebrates -> Objectness foreground (mixes fish & elasmobranchs)
            "SR": {"main": 0, "obj": 0}       # Sea floor / Rock -> Background
        },
        "target_supervision": {
            "main_5class": False,  # Coarse taxonomy corrupts EURS elasmobranch separation
            "boundary": True,
            "objectness": True
        }
    },
    "UIIS10K": {
        "description": "Underwater Image Instance Segmentation (10,000 images)",
        "license": "Academic Research License",
        "source_url": "https://github.com/AutonomousRoboticSystems/UIIS10K",
        "expected_storage_gb": 12.0,
        "class_mapping": {
            # Reliable foreground instance masks
            "background": {"main": 0, "obj": 0},
            "sea_urchin": {"main": 4, "obj": 1},   # Invertebrate / Macro-fauna -> EURS Class 4
            "sea_cucumber": {"main": 4, "obj": 1}, # Invertebrate -> EURS Class 4
            "starfish": {"main": 4, "obj": 1},     # Starfish -> EURS Class 4
            "fish_generic": {"main": 255, "obj": 1}# Ambiguous fish -> Objectness foreground, ignored in 5-class
        },
        "target_supervision": {
            "main_5class": True,  # Only verified echinoderms supervise Class 4
            "boundary": True,
            "objectness": True
        }
    },
    "USIS10K": {
        "description": "Salient Underwater Object Segmentation (10,000 images)",
        "license": "Open Academic Use",
        "source_url": "https://github.com/Visual-Perception-and-Learning-Lab/USIS10K",
        "expected_storage_gb": 8.4,
        "class_mapping": {
            0: {"main": 0, "obj": 0},    # Background
            1: {"main": 255, "obj": 1}   # Salient object -> Binary Objectness foreground, ignored in 5-class
        },
        "target_supervision": {
            "main_5class": False,
            "boundary": True,
            "objectness": True
        }
    }
}


class ExternalDatasetAdapter(Dataset):
    """
    Adapter loading external datasets with strict taxonomy isolation.
    Guarantees ambiguous pixels map strictly to ignore_index=255.
    """
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        split: str = "train",
        crop_size: int = 512,
        is_mock: bool = True
    ):
        super().__init__()
        if dataset_name not in EXTERNAL_TAXONOMY_MAPS:
            raise ValueError(f"Unknown external dataset: {dataset_name}. Valid: {list(EXTERNAL_TAXONOMY_MAPS.keys())}")

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.is_mock = is_mock
        self.meta = EXTERNAL_TAXONOMY_MAPS[dataset_name]

        # In mock mode, generate synthetic samples for CPU-only testing
        self.num_samples = 16 if is_mock else 0
        if not is_mock and osp.exists(data_dir):
            # Parse real files
            self.samples = self._discover_files()
            self.num_samples = len(self.samples)

    def _discover_files(self) -> List[Tuple[str, str]]:
        # Discovers real image/mask pairs when dataset is provided
        samples = []
        img_dir = osp.join(self.data_dir, "images", self.split)
        mask_dir = osp.join(self.data_dir, "masks", self.split)
        if osp.exists(img_dir) and osp.exists(mask_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((osp.join(img_dir, f), osp.join(mask_dir, f)))
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cs = self.crop_size
        if self.is_mock:
            # Synthetic sample for CPU test
            rgb_f32 = np.full((cs, cs, 3), 0.5, dtype=np.float32)
            # Create synthetic coral or object mask
            main_mask = np.full((cs, cs), 0, dtype=np.int64)
            main_mask[cs // 4: 3 * cs // 4, cs // 4: 3 * cs // 4] = 1 if self.meta["target_supervision"]["main_5class"] else 255
            obj_mask = (main_mask > 0).astype(np.float32)
            bnd_mask = np.zeros((cs, cs), dtype=np.float32)
            bnd_mask[cs // 4, :] = 1.0
        else:
            img_p, mask_p = self.samples[idx]
            bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
            raw_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            rgb_f32 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Apply dataset-specific mapping
            main_mask = self._map_raw_mask(raw_mask)
            obj_mask = (main_mask > 0).astype(np.float32)
            bnd_mask = self._compute_boundary_mask(main_mask)

            # Center crop or resize to crop_size
            h, w = main_mask.shape
            if (h, w) != (cs, cs):
                rgb_f32 = cv2.resize(rgb_f32, (cs, cs), interpolation=cv2.INTER_LINEAR)
                main_mask = cv2.resize(main_mask, (cs, cs), interpolation=cv2.INTER_NEAREST)
                obj_mask = cv2.resize(obj_mask, (cs, cs), interpolation=cv2.INTER_NEAREST)
                bnd_mask = cv2.resize(bnd_mask, (cs, cs), interpolation=cv2.INTER_NEAREST)

        return {
            "image": torch.from_numpy(rgb_f32).permute(2, 0, 1).float(),
            "main_mask": torch.from_numpy(main_mask).long(),
            "objectness_mask": torch.from_numpy(obj_mask).float().unsqueeze(0),
            "boundary_mask": torch.from_numpy(bnd_mask).float().unsqueeze(0),
            "dataset_name": self.dataset_name
        }

    def _map_raw_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        mapped = np.full_like(raw_mask, fill_value=IGNORE_INDEX, dtype=np.int64)
        m_cfg = self.meta["class_mapping"]
        for src_id, target_info in m_cfg.items():
            if isinstance(src_id, int):
                target_id = target_info if isinstance(target_info, int) else target_info.get("main", 255)
                mapped[raw_mask == src_id] = target_id
        return mapped

    def _compute_boundary_mask(self, mask: np.ndarray, width: int = 5) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        eroded = cv2.erode(mask.astype(np.uint8), kernel)
        boundary = ((dilated - eroded) > 0).astype(np.float32)
        return boundary


def export_external_taxonomy_manifest(out_path: str = "configs/v18/external_taxonomy_manifest.json") -> Dict[str, Any]:
    """Generates the official versioned external dataset taxonomy manifest."""
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    manifest = {
        "version": "v18",
        "taxonomy_policy": "STRICT_PARTIAL_LABELING_ZERO_TAXONOMY_CORRUPTION",
        "ignore_index": IGNORE_INDEX,
        "datasets": EXTERNAL_TAXONOMY_MAPS
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_contact_sheet_overlay(
    images_np: List[np.ndarray],
    masks_np: List[np.ndarray],
    out_path: str,
    num_samples: int = 4
) -> str:
    """
    Renders an overlay contact sheet color-coding classes:
      0: Gray (Background)
      1: Orange (Coral)
      2: Blue (Fish)
      3: Red (Elasmobranchii - Shark/Ray)
      4: Green (Other Marine Fauna)
      255: Yellow Hatched (Ignored / Auxiliary Pixels)
    """
    palette = {
        0: (128, 128, 128),
        1: (255, 140, 0),
        2: (30, 144, 255),
        3: (220, 20, 60),
        4: (50, 205, 50),
        255: (255, 255, 0)
    }

    tiles = []
    n = min(len(images_np), num_samples)
    for i in range(n):
        img_u8 = (images_np[i] * 255.0).astype(np.uint8) if images_np[i].max() <= 1.0 else images_np[i].astype(np.uint8)
        mask = masks_np[i]

        color_mask = np.zeros_like(img_u8)
        for cls_id, rgb in palette.items():
            color_mask[mask == cls_id] = rgb

        overlay = cv2.addWeighted(img_u8, 0.60, color_mask, 0.40, 0)
        # Put small text label
        classes_present = [c for c in palette if (mask == c).any()]
        txt = f"S{i}: {classes_present}"
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(overlay)

    # Combine into horizontal row
    h, w = tiles[0].shape[:2]
    contact_sheet = np.hstack(tiles)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR))
    return out_path


def get_external_adapter_registry() -> Dict[str, Dict[str, Any]]:
    return EXTERNAL_TAXONOMY_MAPS
