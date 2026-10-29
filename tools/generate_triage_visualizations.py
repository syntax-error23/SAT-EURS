"""
SAT-EURS Track 2 — Forensic Visual Inspection Engine
Generates 4-panel visual comparison figures for candidate models:
[Degraded Input | Restored Output | Predicted Mask | Ground Truth Mask]
For representative validation scenes:
- fish1 (Standard coral & fish)
- sharkFish (Elasmobranch + Fish)
- ray1 (Elasmobranch stingray)
- seahorse1 (Macro-fauna)
- starfish2 (Small benthic macro-fauna)
- turtles1 (Pelagic macro-fauna)
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, ".")

# Palette for classes 0..4 (BGR):
# 0: BG (Black)
# 1: Coral (Orange: 0, 140, 255)
# 2: Fish (Yellow: 0, 255, 255)
# 3: Elasmobranch (Red: 0, 0, 255)
# 4: Other (Green: 0, 255, 0)
COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 140, 255),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (0, 255, 0)
}


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in COLOR_MAP.items():
        color_mask[mask == cid] = col
    return color_mask


def generate_visual_quad(
    onnx_path: str,
    out_dir: str,
    prefix: str,
    scenes: list = ["fish1", "sharkFish", "ray1", "seahorse1", "turtles1"]
):
    os.makedirs(out_dir, exist_ok=True)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_meta = session.get_inputs()[0]
    req_h, req_w = in_meta.shape[2], in_meta.shape[3]
    
    for sc in scenes:
        c_p = f"data/cached_1024/clean/{sc}.png"
        m_p = f"data/cached_1024/masks/{sc}.png"
        deg_dir = f"data/cached_1024/degraded/{sc}"
        
        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])
        if not deg_files:
            continue
        # Pick sample near middle of sequence
        df = deg_files[len(deg_files) // 2]
        d_p = os.path.join(deg_dir, df)
        
        d_im = cv2.imread(d_p)
        m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        
        # Prepare model input
        d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d_in = cv2.resize(d_rgb, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
        inp = d_in.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        outputs = session.run(None, {in_meta.name: inp})
        rest_out = outputs[0][0].transpose(1, 2, 0) # [H, W, 3] in [0, 1]
        logits_out = outputs[1][0] # [5, H, W]
        
        # Resize to standard 512x512 for visual presentation
        d_disp = cv2.resize(d_im, (512, 512))
        rest_bgr = cv2.cvtColor((rest_out * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        rest_disp = cv2.resize(rest_bgr, (512, 512))
        
        pred_mask_raw = np.argmax(logits_out, axis=0).astype(np.uint8)
        pred_mask_disp = cv2.resize(pred_mask_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_mask_disp = cv2.resize(m_im, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        pred_color = mask_to_color(pred_mask_disp)
        gt_color = mask_to_color(gt_mask_disp)
        
        # Label headers
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(d_disp, "1. Degraded Input", (20, 40), font, 0.9, (255, 255, 255), 2)
        cv2.putText(rest_disp, "2. Restored Output", (20, 40), font, 0.9, (255, 255, 255), 2)
        cv2.putText(pred_color, "3. Predicted Mask", (20, 40), font, 0.9, (255, 255, 255), 2)
        cv2.putText(gt_color, "4. Ground Truth", (20, 40), font, 0.9, (255, 255, 255), 2)
        
        # Stack 2x2 or 1x4
        top = np.hstack([d_disp, rest_disp])
        bottom = np.hstack([pred_color, gt_color])
        quad = np.vstack([top, bottom])
        
        out_path = os.path.join(out_dir, f"{prefix}_{sc}_quad.png")
        cv2.imwrite(out_path, quad)
        print(f"Generated quad visualization: {out_path}")


if __name__ == "__main__":
    generate_visual_quad(
        "deploy/eurs_net_v5_calibrated_track2.onnx",
        "reports/track2_readiness/visualizations",
        "v5"
    )
    generate_visual_quad(
        "deploy/eurs_litev3_production.onnx",
        "reports/track2_readiness/visualizations",
        "litev3"
    )
