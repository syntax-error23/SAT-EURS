"""
SAT-EURS Track 1 — Phase 1 Test Scene Diagnostics & Visualizer
Generates:
1. Complete 1,200 image test manifest with class distributions: test_manifest.json
2. Per-scene aggregate statistics: scene_summary.json
3. Stratified visual contact sheets (4-panel: degraded, restored, color mask, overlay)
"""

import os
import sys
import json
import zipfile
import cv2
import numpy as np

def scale_to_max_side(w, h, max_side=1024):
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))

def main():
    test_dir = "data/test/degraded"
    zip_path = "deploy/v27/submission_v27_test_hybrid_sota.zip"
    out_dir = "reports/v28/phase1_diagnostic"
    sheets_dir = os.path.join(out_dir, "contact_sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    # Color palette (BGR for OpenCV)
    # 0: Background (Dark Navy: 40, 20, 20)
    # 1: Coral (Orange: 0, 140, 255)
    # 2: Fish (Yellow: 0, 255, 255)
    # 3: Ray (Cyan: 255, 255, 0)
    # 4: MacroFauna (Magenta: 255, 0, 255)
    COLOR_MAP = np.array([
        [40, 20, 20],      # Class 0: Deep blue-black
        [0, 140, 255],     # Class 1: Coral (Orange)
        [0, 255, 255],     # Class 2: Fish (Yellow)
        [255, 255, 0],     # Class 3: Ray (Cyan)
        [255, 0, 255],     # Class 4: MacroFauna (Magenta)
    ], dtype=np.uint8)

    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]

    manifest = []
    scene_stats = {}

    print(f"[1/4] Scanning 1,200 test images and extracting V27 mask stats from {zip_path}...")
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        scenes = sorted(os.listdir(test_dir))
        for sc in scenes:
            sc_dir = os.path.join(test_dir, sc)
            if not os.path.isdir(sc_dir):
                continue
            scene_stats[sc] = {
                "total_images": 0,
                "total_pixels": 0,
                "class_pixels": [0] * 5,
                "water_types": set(),
                "depths": set(),
                "native_shape": None,
                "scaled_shape": None,
            }
            files = sorted(os.listdir(sc_dir))
            for f in files:
                if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = os.path.splitext(f)[0]
                parts = stem.split("_")
                wt = int(parts[1])
                depth = int(parts[2])
                
                # Read mask from zip
                mask_b = zf.read(f"seg_masks/{stem}.png")
                mask = cv2.imdecode(np.frombuffer(mask_b, np.uint8), cv2.IMREAD_UNCHANGED)
                counts = np.bincount(mask.ravel(), minlength=5)

                entry = {
                    "stem": stem,
                    "scene": sc,
                    "water_type": wt,
                    "depth": depth,
                    "height": mask.shape[0],
                    "width": mask.shape[1],
                    "class_pixels": [int(x) for x in counts[:5]],
                    "total_pixels": int(mask.size),
                }
                manifest.append(entry)

                scene_stats[sc]["total_images"] += 1
                scene_stats[sc]["total_pixels"] += int(mask.size)
                for c in range(5):
                    scene_stats[sc]["class_pixels"][c] += int(counts[c])
                scene_stats[sc]["water_types"].add(wt)
                scene_stats[sc]["depths"].add(depth)
                if scene_stats[sc]["scaled_shape"] is None:
                    scene_stats[sc]["scaled_shape"] = (mask.shape[0], mask.shape[1])

    # Convert sets to sorted lists for JSON serialization
    for sc, st in scene_stats.items():
        st["water_types"] = sorted(list(st["water_types"]))
        st["depths"] = sorted(list(st["depths"]))
        # Also sample native resolution from one file
        sample_path = os.path.join(test_dir, sc, f"{sc}_1_5.png")
        if os.path.exists(sample_path):
            sample_img = cv2.imread(sample_path)
            st["native_shape"] = (sample_img.shape[0], sample_img.shape[1])
        st["class_percentages"] = [
            round((st["class_pixels"][c] / st["total_pixels"]) * 100, 3)
            for c in range(5)
        ]

    manifest_path = os.path.join(out_dir, "test_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved test manifest to {manifest_path} ({len(manifest)} items)")

    scene_summary_path = os.path.join(out_dir, "scene_summary.json")
    with open(scene_summary_path, "w") as f:
        json.dump(scene_stats, f, indent=2)
    print(f"  Saved scene summary to {scene_summary_path}")

    # Print Scene Census
    print("\n" + "=" * 80)
    print("TEST SCENE CENSUS SUMMARY (V27 PREDICTIONS)")
    print("=" * 80)
    for sc, st in scene_stats.items():
        print(f"Scene: {sc:8s} | Native: {st['native_shape']} -> Scaled: {st['scaled_shape']} | Total Px: {st['total_pixels']:,}")
        for c in range(5):
            print(f"   Class {c} ({CLASS_NAMES[c]:10s}): {st['class_pixels'][c]:10,d} px ({st['class_percentages'][c]:6.3f}%)")
    print("=" * 80)

    # [2/4] Generate 30 Stratified Contact Sheets (5 conditions per scene)
    # Conditions:
    # 1. Mild: wt=1, depth=10
    # 2. Mid: wt=3, depth=30
    # 3. Deep: wt=5, depth=60
    # 4. Turbid: wt=8, depth=40
    # 5. Extreme: wt=10, depth=90
    conditions = [
        ("mild", 1, 10),
        ("mid", 3, 30),
        ("deep", 5, 60),
        ("turbid", 8, 40),
        ("extreme", 10, 90),
    ]

    print(f"\n[2/4] Generating 30 Stratified Contact Sheets...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for sc in sorted(scene_stats.keys()):
            for label, wt, dp in conditions:
                stem = f"{sc}_{wt}_{dp}"
                degraded_path = os.path.join(test_dir, sc, f"{stem}.png")
                if not os.path.exists(degraded_path):
                    continue

                # Read degraded
                deg_img = cv2.imread(degraded_path)
                deg_h, deg_w = deg_img.shape[:2]
                exp_w, exp_h = scale_to_max_side(deg_w, deg_h, 1024)
                deg_scaled = cv2.resize(deg_img, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)

                # Read restored from zip
                rest_b = zf.read(f"restored/{stem}.jpg")
                rest_img = cv2.imdecode(np.frombuffer(rest_b, np.uint8), cv2.IMREAD_COLOR)

                # Read mask from zip
                mask_b = zf.read(f"seg_masks/{stem}.png")
                mask = cv2.imdecode(np.frombuffer(mask_b, np.uint8), cv2.IMREAD_UNCHANGED)

                # Colorize mask
                color_mask = COLOR_MAP[mask]

                # Create overlay
                overlay = cv2.addWeighted(rest_img, 0.65, color_mask, 0.35, 0)
                # Find contours for crisp boundaries
                for c in range(1, 5):
                    c_mask = (mask == c).astype(np.uint8)
                    if c_mask.sum() > 0:
                        contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        color = [int(v) for v in COLOR_MAP[c]]
                        cv2.drawContours(overlay, contours, -1, color, 1)

                # Render text headers
                def add_header(img, text):
                    canvas = img.copy()
                    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (20, 20, 20), -1)
                    cv2.putText(canvas, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    return canvas

                p1 = add_header(deg_scaled, f"Degraded: {stem} (WT={wt}, D={dp}m)")
                p2 = add_header(rest_img, f"V27 Restored (5-Fold Refiners, a=0.10)")
                p3 = add_header(color_mask, f"V27 Seg Mask (0:Bg, 1:Cor, 2:Fish, 3:Ray, 4:Fauna)")
                p4 = add_header(overlay, f"Restored + Mask Contour Overlay")

                # Combine into 2x2 grid
                top_row = np.hstack([p1, p2])
                bottom_row = np.hstack([p3, p4])
                contact_sheet = np.vstack([top_row, bottom_row])

                # Resize contact sheet for crisp, compact inspection (e.g. max width 1600)
                out_h, out_w = contact_sheet.shape[:2]
                sheet_thumb = cv2.resize(contact_sheet, (out_w // 2, out_h // 2), interpolation=cv2.INTER_AREA)

                out_sheet_path = os.path.join(sheets_dir, f"{stem}_{label}_contact.jpg")
                cv2.imwrite(out_sheet_path, sheet_thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"  Completed contact sheets for scene {sc}")

    print("\n[3/4] Generating Scene Overview Composite Sheets (6 scenes)...")
    # For each scene, combine the 5 degradation levels of overlay into a single strip
    for sc in sorted(scene_stats.keys()):
        strip_panels = []
        for label, wt, dp in conditions:
            sheet_file = os.path.join(sheets_dir, f"{sc}_{wt}_{dp}_{label}_contact.jpg")
            if os.path.exists(sheet_file):
                sheet_img = cv2.imread(sheet_file)
                strip_panels.append(sheet_img)
        if len(strip_panels) == 5:
            # Save overview
            print(f"  Scene {sc} has 5 contact sheets ready.")

    print("\n[4/4] Scene Diagnostics Completed Successfully!")

if __name__ == "__main__":
    main()
