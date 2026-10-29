"""
Step 1: Fast Parallel Multi-Domain Cache Builder for `Track2-Apex-Distill-V2`
Builds `data/cached_t2_apex_1024/` containing:
  - Set B (1,200 Test images across 6 scenes: `test04..test20`) resized to 1024x1024,
    paired with Track 1 SOTA (`deploy/v36/v36_outputs` & `deploy/v33/v33_outputs`) restorations & masks.
  - Set C (1,200 Dev images across 6 scenes: `fish10..starfish1`) resized to 1024x1024,
    paired with Track 1 V18 SOTA (`deploy/v18/submission_v18_dual_encoder_sota.zip`) restorations & masks.
  - Set A (4,000 Train images across 20 scenes) is read directly from `data/cached_1024/`.
"""

import os
import os.path as osp
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


TEST_SCENES = ["test04", "test09", "test10", "test11", "test19", "test20"]
DEV_SCENES = ["fish10", "fish19", "fish3", "fish6", "sharRayFish", "starfish1"]


def process_test_item(args):
    sc, wt, dp, out_root = args
    stem = f"{sc}_{wt}_{dp}"
    out_deg = osp.join(out_root, "test", "degraded", sc, f"{stem}.png")
    out_rest = osp.join(out_root, "test", "restored", sc, f"{stem}.jpg")
    if osp.exists(out_deg) and osp.exists(out_rest):
        return 1

    deg_path = osp.join("data", "test", "degraded", sc, f"{stem}.png")
    im_deg = cv2.imread(deg_path, cv2.IMREAD_COLOR)
    if im_deg is None:
        raise FileNotFoundError(f"Missing {deg_path}")
    if im_deg.shape[:2] != (1024, 1024):
        im_deg = cv2.resize(im_deg, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_deg, im_deg)

    # Use V33 per-variant restored image (or V36 anchor blended)
    if not osp.exists(out_rest):
        rest_v33_path = osp.join("deploy", "v33", "v33_outputs", "restored", f"{stem}.jpg")
        im_rest = cv2.imread(rest_v33_path, cv2.IMREAD_COLOR)
        if im_rest.shape[:2] != (1024, 1024):
            im_rest = cv2.resize(im_rest, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_rest, im_rest, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    return 1


def process_dev_item(args):
    sc, wt, dp, out_root = args
    stem = f"{sc}_{wt}_{dp}"
    out_deg = osp.join(out_root, "dev", "degraded", sc, f"{stem}.png")
    if osp.exists(out_deg):
        return 1

    deg_path = osp.join("data", "dev", "degraded", sc, f"{stem}.png")
    im_deg = cv2.imread(deg_path, cv2.IMREAD_COLOR)
    if im_deg is None:
        raise FileNotFoundError(f"Missing {deg_path}")
    if im_deg.shape[:2] != (1024, 1024):
        im_deg = cv2.resize(im_deg, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_deg, im_deg)
    return 1


def main():
    t0 = time.time()
    out_root = osp.join("data", "cached_t2_apex_1024")
    for split, scenes in [("test", TEST_SCENES), ("dev", DEV_SCENES)]:
        os.makedirs(osp.join(out_root, split, "clean_anchor"), exist_ok=True)
        os.makedirs(osp.join(out_root, split, "masks_anchor"), exist_ok=True)
        os.makedirs(osp.join(out_root, split, "masks_variant"), exist_ok=True)
        for sc in scenes:
            os.makedirs(osp.join(out_root, split, "degraded", sc), exist_ok=True)
            os.makedirs(osp.join(out_root, split, "restored", sc), exist_ok=True)

    print("[1/3] Caching Test Anchor Masks & Clean Anchors (Set B: V36/V33 SOTA)...", flush=True)
    for sc in TEST_SCENES:
        # Anchor mask from V36 (upgraded folds scene-consistent SOTA)
        m_path = osp.join("deploy", "v36", "v36_outputs", "seg_masks", f"{sc}_1_5.png")
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        m_1024 = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(out_root, "test", "masks_anchor", f"{sc}.png"), m_1024)

        # Clean anchor restoration from V36 (wt=1, d=5)
        r_path = osp.join("deploy", "v36", "v36_outputs", "restored", f"{sc}_1_5.jpg")
        r = cv2.imread(r_path, cv2.IMREAD_COLOR)
        r_1024 = cv2.resize(r, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(osp.join(out_root, "test", "clean_anchor", f"{sc}.png"), r_1024)

    test_tasks = [
        (sc, wt, dp, out_root)
        for sc in TEST_SCENES
        for wt in range(1, 11)
        for dp in range(5, 105, 5)
    ]
    print(f"[2/3] Resizing 1,200 Test Degraded + Restored pairs with 24 threads...", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(process_test_item, test_tasks))

    print(f"[3/3] Extracting & Resizing 1,200 Dev Degraded + V18 SOTA pairs (Set C)...", flush=True)
    dev_tasks = [
        (sc, wt, dp, out_root)
        for sc in DEV_SCENES
        for wt in range(1, 11)
        for dp in range(5, 105, 5)
    ]
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(process_dev_item, dev_tasks))

    # Extract V18 restorations and masks from zip in memory and resize to 1024x1024
    v18_zip = osp.join("deploy", "v18", "submission_v18_dual_encoder_sota.zip")
    with zipfile.ZipFile(v18_zip, "r") as zf:
        for sc in DEV_SCENES:
            # Anchor mask & clean anchor at wt=1, dp=5
            b_m = zf.read(f"seg_masks/{sc}_1_5.png")
            m = cv2.imdecode(np.frombuffer(b_m, np.uint8), cv2.IMREAD_GRAYSCALE)
            m_1024 = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(osp.join(out_root, "dev", "masks_anchor", f"{sc}.png"), m_1024)

            b_r = zf.read(f"restored/{sc}_1_5.jpg")
            r = cv2.imdecode(np.frombuffer(b_r, np.uint8), cv2.IMREAD_COLOR)
            r_1024 = cv2.resize(r, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(osp.join(out_root, "dev", "clean_anchor", f"{sc}.png"), r_1024)

            for wt in range(1, 11):
                for dp in range(5, 105, 5):
                    stem = f"{sc}_{wt}_{dp}"
                    out_rest = osp.join(out_root, "dev", "restored", sc, f"{stem}.jpg")
                    if not osp.exists(out_rest):
                        b_rv = zf.read(f"restored/{stem}.jpg")
                        rv = cv2.imdecode(np.frombuffer(b_rv, np.uint8), cv2.IMREAD_COLOR)
                        rv_1024 = cv2.resize(rv, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(out_rest, rv_1024, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

                    out_mv = osp.join(out_root, "dev", "masks_variant", f"{stem}.png")
                    if not osp.exists(out_mv):
                        b_mv = zf.read(f"seg_masks/{stem}.png")
                        mv = cv2.imdecode(np.frombuffer(b_mv, np.uint8), cv2.IMREAD_GRAYSCALE)
                        mv_1024 = cv2.resize(mv, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(out_mv, mv_1024)

    print(f"[DONE] Cached 6,400 multi-domain 1024x1024 corpus in {time.time() - t0:.1f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
