# EURS-Net V2 Dataset & Validation Integrity Audit

**Evaluation:** Scene-Disjoint Triplet Dataset & 800-Image Validation Contract  
**Location:** [`splits/manifest_full.json`](file:///C:/Projects/SAT-EURS/splits/manifest_full.json), [`data/dataset.py`](file:///C:/Projects/SAT-EURS/data/dataset.py)  
**Status:** **100% VERIFIED & COMPLIANT**  

---

## 1. Scene-Disjoint Partition Integrity

The dataset partitions are governed strictly by the persistent frozen manifest: [`splits/manifest_full.json`](file:///C:/Projects/SAT-EURS/splits/manifest_full.json).

* **Total Scenes:** 20 scenes (4,000 total paired images)
* **Training Set:** 16 scenes (3,200 paired triplets)
  ```
  ['crab1', 'fish2', 'fish3', 'fish4', 'fish5', 'fish6', 'fish7', 'fish8', 
   'fish9', 'fish10', 'fish11', 'fish13', 'fish14', 'ray1', 'seahorse2', 'seahorse3']
  ```
* **Validation Set:** 4 scenes (800 paired triplets)
  ```
  ['fish1', 'fish12', 'seahorse1', 'sharkFish']
  ```
* **Disjointness Assertion:**
  $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset \quad (\text{Strictly Disjoint})$$
  Verified programmatically: zero scene overlap, zero data leakage between training and validation.

---

## 2. Triplet Pairing & Contract Invariants

In [`data/dataset.py`](file:///C:/Projects/SAT-EURS/data/dataset.py), every sample is indexed and asserted at runtime:

```python
deg_scene_check == clean_scene_check == mask_scene_check == scene_id
```

| Tensor Component | Type | Shape | Value Range | Interpolation Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Degraded Input ($x$)** | `torch.float32` | `[3, 1024, 1024]` | $[0.0, 1.0]$ (RGB) | `cv2.INTER_LINEAR` |
| **Clean Reference ($y$)** | `torch.float32` | `[3, 1024, 1024]` | $[0.0, 1.0]$ (RGB) | `cv2.INTER_LINEAR` |
| **Semantic Mask ($M$)** | `torch.int64` | `[1024, 1024]` | $\{0, 1, 2, 3, 4\}$ | `cv2.INTER_NEAREST` |

* **Color Format:** OpenCV reads BGR; `data/dataset.py` explicitly converts all images to RGB via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
* **Discrete Semantic Classes:**
  * `0`: Background
  * `1`: Coral
  * `2`: Fish
  * `3`: Elasmobranch (Shark & Ray)
  * `4`: Other Marine Life

---

## 3. Synchronized Spatial Augmentations

In [`data/transforms.py`](file:///C:/Projects/SAT-EURS/data/transforms.py), `SynchronizedTransform` guarantees exact geometric alignment:
1. Spatial parameters (`do_flip`, `crop_box`, `target_size`) are generated once per sample index.
2. Identical transformations are applied to the degraded image, clean target, and ground-truth mask.
3. Masks strictly use `cv2.INTER_NEAREST`, ensuring zero interpolation artifacts or spurious fractional class IDs.
4. Input and clean images use `cv2.INTER_LINEAR`.

---

## 4. Validation Contract Accounting

* **Validation Population:** Exactly 800 paired images.
  * `fish1`: 200 variations
  * `fish12`: 200 variations
  * `seahorse1`: 200 variations
  * `sharkFish`: 200 variations
* **Determinism:** Validation loader operates with `shuffle=False`, zero data augmentation, and batch size 4 (or 1), guaranteeing 100% deterministic reproducibility across all audit and verification runs.
