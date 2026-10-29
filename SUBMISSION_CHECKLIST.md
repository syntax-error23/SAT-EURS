# SAT-EURS Official Submission Checklist & Verification Protocol
## MaCVi @ WACV 2027 EURS Challenge (Tracks 1 & 2)
**Date:** October 16, 2026 | Local Time: 13:49 IST  
**Version:** 1.0.0 (Gate 1 Baseline Locked)  

---

## 1. Track 1 Submission Verification Protocol (Server Leaderboard)

A Track 1 submission is a single `.zip` archive containing two top-level directories:
- `restored/`: 1,200 restored color images matching Dev set file stems.
- `seg_masks/`: 1,200 single-channel PNG segmentation masks matching Dev set file stems.

### Mandatory Pre-Upload Verification Checklist:
- [ ] **1. Exact File Count:** Exactly 1,200 files in `restored/` and exactly 1,200 files in `seg_masks/`.
- [ ] **2. Stem Matching:** Every file stem in `seg_masks/` matches its counterpart in `restored/` and the official Dev set manifest (`data/dev/degraded`). Zero missing, zero extra.
- [ ] **3. Direct Root Structure:** No parent folder wrapping `restored/` and `seg_masks/` inside the ZIP archive. Both directories must sit directly at the ZIP root (`/restored/...`, `/seg_masks/...`).
- [ ] **4. Discrete Mask Validation:** Every mask in `seg_masks/` must be a single-channel 8-bit unsigned integer PNG (`uint8`, mode `L`). All pixel values must strictly belong to the discrete set:
  $$\mathcal{C} \in \{0, 1, 2, 3, 4\}$$
  $$\text{0: Background, 1: Coral, 2: Fish, 3: Elasmobranch, 4: Other}$$
  Assert: `np.all(np.isin(mask_np, [0, 1, 2, 3, 4])) == True`.
- [ ] **5. Mandatory Resize Wrapper:** The submission outputs MUST be post-processed using the official `tools/resize_for_submission.py` script:
  - Masks resized strictly with `cv2.INTER_NEAREST` to prevent floating-point interpolation artifacts or spurious class IDs.
  - Restored images resized with `cv2.INTER_LINEAR`.
  - Target dimensions must match the original aspect-preserved dimensions capped at 1024 pixels on the longest side.
- [ ] **6. Color & Range Validation:** Restored images must be valid 3-channel RGB/BGR images with pixel values in range $[0, 255]$ with zero NaN, Inf, or corrupted headers.
- [ ] **7. Cryptographic Hashing:** The final ZIP archive must be hashed via SHA-256 and recorded in `EXPERIMENT_LEDGER.csv` before upload.
- [ ] **8. Fallback Preservation:** Never overwrite a previously verified submission archive. Always create versioned files (e.g. `deploy/eurs_track1_v17_candidate.zip`). Verified fallback is permanently locked at `deploy/eurs_track1_dryrun_compact_candidate.zip`.
- [ ] **9. Daily Submission Cap:** Confirm remaining submissions for the 24-hour cycle on the live challenge dashboard (up to 10 submissions per day).

---

## 2. Track 2 Submission Verification Protocol (Jetson AGX Orin Edge Deployment)

A Track 2 submission is a single standalone `.onnx` model file evaluated on the competition's NVIDIA Jetson AGX Orin hardware (JetPack 7.2.1, 30W stock power mode, 16 GB unified memory cap).

### Mandatory Pre-Upload Verification Checklist:
- [ ] **1. Single Standalone File:** The model must be entirely self-contained in one `.onnx` file with embedded weights (`save_as_external_data=False`). No `.onnx.data` companion files permitted.
- [ ] **2. Static Input Tensor:** Exactly one input named `input` with static shape:
  $$\text{Shape: } [1, 3, 1024, 1024], \quad \text{Dtype: } \text{float32}, \quad \text{Range: } [0.0, 1.0]$$
  No dynamic spatial dimensions allowed.
- [ ] **3. Output Tensor 1 (`restored_image`):** Exactly named `restored_image` with shape:
  $$\text{Shape: } [1, 3, 1024, 1024], \quad \text{Dtype: } \text{float32}, \quad \text{Range: } [0.0, 1.0]$$
- [ ] **4. Output Tensor 2 (`seg_logits`):** Exactly named `seg_logits` with shape:
  $$\text{Shape: } [1, 5, 1024, 1024], \quad \text{Dtype: } \text{float32}$$
  Must contain raw un-thresholded floating-point logits in channel order:
  $$\text{Ch 0: Background, Ch 1: Coral, Ch 2: Fish, Ch 3: Elasmobranch, Ch 4: Other}$$
  Must NOT be an argmax discrete mask.
- [ ] **5. Official Design-Space Cap Compliance:** Run the official static checker `deploy/check_submission_flops.py`:
  - Total Compute: $\le \mathbf{12.0\text{ GFLOPs}}$ (Conv, Gemm, MatMul)
  - Total Parameters: $\le \mathbf{1.0\text{M}}$ ($1,000,000$ parameters)
  - Assert status output: `PASSED: within the configured compute/parameter caps.`
- [ ] **6. Structural ONNX Validation:** Pass `onnx.checker.check_model()` and `onnx.shape_inference.infer_shapes()`.
- [ ] **7. ONNX Runtime Numerical Equivalence:** Run `tools/evaluate_track2_onnx.py` on held-out validation scenes. Verify that ONNX Runtime CPU/CUDA predictions match PyTorch eager mode within maximum absolute tolerance $\Delta < 10^{-4}$.
- [ ] **8. Zero Metadata Dependency:** Verify that the model performs inference purely from the input image tensor without requiring image filenames, scene identifiers, water type, or depth metadata.
- [ ] **9. Cryptographic Hashing:** The final ONNX model must be hashed via SHA-256 and logged into `EXPERIMENT_LEDGER.csv`. Fallback reference is locked at `deploy/eurs_net_v5_calibrated_track2.onnx`.

---

## 3. Automated Post-Generation Validation Script

Execute this verification script on any new submission package before uploading:

```python
import os, zipfile, onnx, numpy as np
from PIL import Image

def validate_track1_zip(zip_path, expected_count=1200):
    print(f"[CHECK] Validating Track 1 Archive: {zip_path}")
    assert os.path.exists(zip_path), f"File not found: {zip_path}"
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        restored = [n for n in names if n.startswith("restored/") and not n.endswith("/")]
        masks = [n for n in names if n.startswith("seg_masks/") and not n.endswith("/")]
        assert len(restored) == expected_count, f"Restored count mismatch: {len(restored)} != {expected_count}"
        assert len(masks) == expected_count, f"Masks count mismatch: {len(masks)} != {expected_count}"

        # Test sample mask discrete classes
        sample_mask_name = masks[0]
        with z.open(sample_mask_name) as mf:
            im = Image.open(mf)
            arr = np.array(im)
            assert set(np.unique(arr)).issubset({0, 1, 2, 3, 4}), f"Invalid mask classes in {sample_mask_name}: {np.unique(arr)}"
    print("[CHECK] Track 1 validation: PASSED PERFECTLY.")

def validate_track2_onnx(onnx_path):
    print(f"[CHECK] Validating Track 2 ONNX Model: {onnx_path}")
    assert os.path.exists(onnx_path), f"File not found: {onnx_path}"
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    # Check I/O names
    input_names = [i.name for i in m.graph.input]
    output_names = [o.name for o in m.graph.output]
    assert "input" in input_names, f"Missing input: {input_names}"
    assert "restored_image" in output_names, f"Missing restored_image: {output_names}"
    assert "seg_logits" in output_names, f"Missing seg_logits: {output_names}"
    print("[CHECK] Track 2 validation: PASSED PERFECTLY.")
```
