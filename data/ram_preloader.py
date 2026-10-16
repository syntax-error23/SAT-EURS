"""
RAM Preloader Module for SAT-EURS Track 1
Preloads data/cached_1024 into host RAM once at startup (< 4.0 GB footprint).
Eliminates 100% of runtime disk I/O across all folds and models.
"""

import os
import os.path as osp
import time
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

_RAM_CACHE = {
    "clean": {},    # sc -> np.ndarray [1024, 1024, 3] RGB float32 / 255.0
    "masks": {},    # sc -> np.ndarray [1024, 1024] uint8
    "degraded": {}  # (sc, filename) -> bytes (compressed png in RAM)
}
_IS_PRELOADED = False

def preload_all_to_ram(data_root: str = "data/cached_1024") -> Dict:
    global _IS_PRELOADED, _RAM_CACHE
    if _IS_PRELOADED:
        return _RAM_CACHE

    t0 = time.time()
    clean_dir = osp.join(data_root, "clean")
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    # 1. Clean images
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".png")])
    for f in clean_files:
        sc = osp.splitext(f)[0]
        bgr = cv2.imread(osp.join(clean_dir, f), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        _RAM_CACHE["clean"][sc] = rgb

    # 2. Masks
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    for f in mask_files:
        sc = osp.splitext(f)[0]
        m = cv2.imread(osp.join(masks_dir, f), cv2.IMREAD_GRAYSCALE)
        _RAM_CACHE["masks"][sc] = m

    # 3. Degraded images (stored as raw compressed bytes in RAM for < 4 GB footprint)
    scenes = sorted(os.listdir(deg_dir))
    count_deg = 0
    for sc in scenes:
        sc_p = osp.join(deg_dir, sc)
        if osp.isdir(sc_p):
            for f in sorted(os.listdir(sc_p)):
                if f.endswith(".png"):
                    with open(osp.join(sc_p, f), "rb") as fp:
                        _RAM_CACHE["degraded"][(sc, f)] = fp.read()
                    count_deg += 1

    total_bytes = (
        sum(len(v) for v in _RAM_CACHE["degraded"].values()) +
        sum(v.nbytes for v in _RAM_CACHE["clean"].values()) +
        sum(v.nbytes for v in _RAM_CACHE["masks"].values())
    )
    _IS_PRELOADED = True
    print(f"[RAM_PRELOADER] Preloaded {len(_RAM_CACHE['clean'])} clean, {len(_RAM_CACHE['masks'])} masks, {count_deg} degraded into RAM in {time.time()-t0:.2f}s ({total_bytes / (1024**3):.2f} GB).", flush=True)
    return _RAM_CACHE

def get_clean_image(sc: str) -> np.ndarray:
    if not _IS_PRELOADED:
        preload_all_to_ram()
    return _RAM_CACHE["clean"][sc]

def get_mask(sc: str) -> np.ndarray:
    if not _IS_PRELOADED:
        preload_all_to_ram()
    return _RAM_CACHE["masks"][sc]

def get_degraded_rgb(sc: str, filename: str) -> np.ndarray:
    if not _IS_PRELOADED:
        preload_all_to_ram()
    raw_bytes = _RAM_CACHE["degraded"][(sc, filename)]
    bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb
