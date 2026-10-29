"""
Test Single-Image Independence for Phase 2 Submission
Verifies that evaluating an isolated Dev image produces identical byte-level outputs
to the packaged submission (zero batch communication, zero cross-image leakage).
"""

import os
import sys
import io
import zipfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")

from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from deploy.generate_phase2_verified_submission import scale_to_max_side

def test_independence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zip_path = "deploy/eurs_track1_phase2_verified.zip"
    assert os.path.exists(zip_path), f"Zip missing: {zip_path}"

    test_stem = "fish10_1_5"
    test_img_path = "data/dev/degraded/fish10/fish10_1_5.png"
    assert os.path.exists(test_img_path), f"Test image missing: {test_img_path}"

    # 1. Read packaged outputs
    with zipfile.ZipFile(zip_path, "r") as zf:
        pack_rest_bytes = zf.read(f"restored/{test_stem}.jpg")
        pack_mask_bytes = zf.read(f"seg_masks/{test_stem}.png")

    pack_rest = cv2.imdecode(np.frombuffer(pack_rest_bytes, np.uint8), cv2.IMREAD_COLOR)
    pack_mask = cv2.imdecode(np.frombuffer(pack_mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    # 2. Run isolated single-image pipeline
    table = torch.load("checkpoints/track1_screening/tbl19_20sc_clean.pt", map_location=device, weights_only=False)
    models = []
    for f in range(5):
        m = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(f"checkpoints/track1_screening/DINOV2_CP_F{f}/best_model.pt", map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)

    orig_bgr = cv2.imread(test_img_path)
    orig_h, orig_w = orig_bgr.shape[:2]
    target_w, target_h = scale_to_max_side(orig_w, orig_h, 1024)

    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    x_raw_1024 = torch.from_numpy(rgb_1024.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    wt_t = torch.tensor([1], device=device)
    dp_t = torch.tensor([5], device=device)

    with torch.no_grad():
        res_1024 = apply_rsuigm_batch_gpu(x_raw_1024, wt_t, dp_t, table)

    res_np = (res_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
    sharp_np = np.clip(cv2.addWeighted(res_np, 1.34, blur, -0.34, 0), 0, 255).astype(np.uint8)
    sharp_resized = cv2.resize(sharp_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Compare restored images
    enc_bytes = cv2.imencode('.jpg', cv2.cvtColor(sharp_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])[1].tobytes()
    assert enc_bytes == pack_rest_bytes, "Restored JPEG bytes do not match packaged image byte-for-byte!"
    dec_single = cv2.imdecode(np.frombuffer(enc_bytes, np.uint8), cv2.IMREAD_COLOR)
    rest_diff = np.abs(pack_rest.astype(np.int32) - dec_single.astype(np.int32)).max()
    print(f"Single-image restored byte match: PASS (Decoded max pixel diff: {rest_diff})")
    assert rest_diff == 0, f"Restored image difference should be 0, got {rest_diff}"

    # Segmentation
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_norm_1024 = (res_1024 - mean_im) / std_im

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]
    ensemble_probs = torch.zeros((1, 5, 518, 518), device=device)

    with torch.no_grad():
        for m in models:
            model_scale_probs = torch.zeros((1, 5, 518, 518), device=device)
            for sc_sz, sc_w in zip(scales, scale_weights):
                x_sc = F.interpolate(x_norm_1024, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                x_sc_flip = torch.flip(x_sc, dims=[3])

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_orig = m(x_sc)
                    l_flip = m(x_sc_flip)

                p_orig = F.softmax(l_orig.float(), dim=1)
                p_flip = torch.flip(F.softmax(l_flip.float(), dim=1), dims=[3])
                p_fused = 0.5 * p_orig + 0.5 * p_flip

                p_518 = F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)
                model_scale_probs += sc_w * p_518

            ensemble_probs += model_scale_probs

    ensemble_probs /= len(models)
    pred_mask_518 = ensemble_probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    mask_final = cv2.resize(pred_mask_518, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    mask_diff = np.abs(pack_mask.astype(np.int32) - mask_final.astype(np.int32)).max()
    print(f"Single-image segmentation mask max diff vs packaged: {mask_diff}")
    assert mask_diff == 0, f"Segmentation mask differs between single-image and batch! Diff: {mask_diff}"
    print("Single-Image Independence: PASS (Exact 100% bitwise mask match!)")

if __name__ == '__main__':
    test_independence()
