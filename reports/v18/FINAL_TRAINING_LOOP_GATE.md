# SAT-EURS Track 1 — V18 Final Training-Loop Verification Report

**Date:** 2026-10-17  
**Gate Verdict:** **`PASS`**  
**Audited Target:** Training-Loop Gradient Accumulation, Scheduling, Clipping, and Regression Tests  
**Execution Guardrails:** Zero GPU training, zero external downloads, zero submission generation.

---

## Executive Summary

Following the discovery during the pre-training audit that gradient accumulation was not active during the Fold 0 pilot benchmark, [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py) was updated to honor `accum_steps = 4` across both training stages.

This audit inspects the final implementation, mathematically verifies gradient equivalence on a toy model, validates partial accumulation window handling, reviews all Git changes since the completed GPU pilot, and confirms baseline preservation.

**Gate Verdict:** **`PASS`** — Both training stages correctly implement the intended gradient accumulation behavior. The V18 pipeline is formally validated and ready for the planned five-fold, 15-epoch training run upon authorization.

---

## 1. Verification of Gradient Accumulation in Both Training Stages

### 1.1 Stage 1: Restoration Refiner (`train_refiner_fold`)
Inspection of [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py#L215-L270):
```python
t_cfg = config.get("training", {})
epochs = max_epochs if max_epochs is not None else t_cfg.get("epochs", 15)
accum_steps = int(t_cfg.get("accum_steps", 1)) # Reads accum_steps=4 from restoration_config.json

train_ds = V18EURSInvarianceDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=seed)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

for epoch in range(epochs):
    ...
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        deg = batch["deg_rgb"].to(device)
        clean = batch["clean_rgb"].to(device)
        wt = batch["wt"].to(device)
        depth = batch["depth"].to(device)

        out = refiner(deg, wt, depth)
        l_dict = loss_fn(out, clean)
        loss = l_dict["loss"] / accum_steps # Loss divided by 4 before backward
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_updates += 1

        running_loss += l_dict["loss"].item() # Tracks unscaled per-minibatch loss
        step_count += 1

    scheduler.step() # Advances once per epoch (CosineAnnealingLR, T_max=epochs)
```

### 1.2 Stage 2: DINOv3 Segmenter (`train_segmenter_fold`)
Inspection of [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py#L380-L465):
```python
opt_cfg = config.get("optimization", {})
epochs = max_epochs if max_epochs is not None else opt_cfg.get("epochs", 15)
accum_steps = int(opt_cfg.get("accum_steps", 1)) # Reads accum_steps=4 from segmentation_config.json

train_ds = V18EURSInvarianceDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=seed)
sampler = V18SceneBalancedSampler(train_ds, batch_size=4, seed=seed)
train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=0)

for epoch in range(epochs):
    ...
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        ...
        seg_out = segmenter(restored, deg, wt, depth, unpadded_size=(512, 512))
        l_dict = loss_fn(seg_out, mask)
        loss = l_dict["loss"] / accum_steps # Loss divided by 4 before backward
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(segmenter.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_updates += 1

        running_loss += l_dict["loss"].item() # Tracks unscaled per-minibatch loss
        step_count += 1

    scheduler.step() # Advances once per epoch (CosineAnnealingLR, T_max=epochs)
```

### 1.3 Control Flow Analysis for an 800-Minibatch Epoch
| Property | Contract Specification | Actual Implementation Finding | Status |
| :--- | :---: | :---: | :---: |
| **Physical Minibatches** | 800 per epoch | 3,200 training samples $\div 4 = \mathbf{800}$ minibatches | **VERIFIED** |
| **Forward/Backward Iterations** | 800 per epoch | Loop runs 800 times, each calling `loss.backward()` | **VERIFIED** |
| **Loss Scaling** | Divided by 4 before backward | `loss = l_dict["loss"] / accum_steps` | **VERIFIED** |
| **Accumulated Gradient Scale** | Equivalent to batch size 16 | $\sum_{i=1}^4 \frac{1}{4} \nabla L_i = \nabla \left( \frac{1}{4} \sum_{i=1}^4 L_i \right)$ | **VERIFIED** |
| **`optimizer.step()` Boundaries** | Every 4 minibatches | Triggered when `(batch_idx + 1) % 4 == 0` | **VERIFIED** |
| **`optimizer.zero_grad()` Placement** | Before loop & after step | Initialized at $t=0$, reset immediately after `step()` | **VERIFIED** |
| **Optimizer Updates per Epoch** | Exactly 200 | $\frac{800}{4} = \mathbf{200}$ updates per epoch | **VERIFIED** |
| **Gradient Clipping Placement** | On accumulated gradient | `clip_grad_norm_` called after 4th batch, before `step()` | **VERIFIED** |
| **Learning-Rate Scheduler** | Advances per epoch | `scheduler.step()` called outside batch loop (`T_max=epochs`) | **VERIFIED** |
| **Logging Accuracy** | Unscaled per-minibatch loss | `running_loss += l_dict["loss"].item()` tracks true loss | **VERIFIED** |

### 1.4 Divisibility and Partial Window Handling
- **Production EURS Divisibility:** In EURS Track 1, 16 training scenes with 200 images each yield exactly **3,200 samples**. With `batch_size = 4`, the loader produces exactly **800 minibatches**. Because:
  $$800 \pmod 4 = 0$$
  every accumulation window has **exactly 4 minibatches**. There are zero partial windows in production training.
- **Partial Window Guard:** The guard `or (batch_idx + 1) == len(train_loader)` guarantees that on arbitrary splits or unit test fixtures where `len(loader)` is not divisible by 4, trailing minibatches trigger an update and are not silently discarded.

---

## 2. Deterministic CPU Regression Tests

A dedicated deterministic unit test, `test_gradient_accumulation_and_stage_contracts`, was added to [`tests/v18/test_v18_synthetic.py`](file:///c:/Projects/SAT-EURS/tests/v18/test_v18_synthetic.py#L560-L650) and verified on CPU:

### 2.1 Test Assertions & Results
1. **Mathematical Gradient Equivalence:**
   - Evaluated a linear model on 8 samples ($4 \times 2$ minibatches) against a cloned model on the combined $8$-sample batch.
   - Verified that accumulated gradients match combined batch gradients within floating-point tolerance:
     $$\| G_{\text{accum}} - G_{\text{comb}} \|_{\infty} < 10^{-6} \quad (\text{PASS})$$
   - Verified that exactly **1 optimizer update** was performed across the 4 minibatches.
2. **Partial Accumulation Window Handling:**
   - Evaluated a 7-minibatch sequence with `accum_steps = 4`.
   - Verified updates occurred at batch index 3 (`(3 + 1) % 4 == 0`) and batch index 6 (`(6 + 1) == 7`).
   - Verified exactly **2 optimizer updates** occurred, confirming trailing minibatches are preserved.
3. **Stage Contract AST / Code Verification:**
   - Tested that both `train_refiner_fold` and `train_segmenter_fold` query `accum_steps` from their respective config blocks, divide loss by `accum_steps`, and enforce `(batch_idx + 1) % accum_steps == 0`.
   - Verified that simulating 800 minibatches with `accum_steps = 4` yields **200 updates**, whereas with `accum_steps = 1` it yields **800 updates**. The test fails if either stage accidentally performs 800 updates when `accum_steps = 4`.

**Suite Result:** All **14/14** synthetic unit tests passed on CPU.

---

## 3. Review of Changes Since Completed GPU Pilot

A full audit of changes made since the Fold 0 GPU benchmark completed shows zero unintended regressions:

| File | Change Category | Description | Impact Assessment |
| :--- | :---: | :--- | :--- |
| [`tools/v18/evaluator.py`](file:///c:/Projects/SAT-EURS/tools/v18/evaluator.py) | Metrics | Added per-class pixel counts ($N_{\text{gt}}, N_{\text{pred}}, \text{TP}, \text{FP}, \text{FN}$) and dual macro averages (present-class mean vs conservative 5-class zero mean). Eliminates artificial $1.0$ on absent classes. | **Metrics only**. Zero impact on model architecture, weights, optimization, fold isolation, or packaging. |
| [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py) | Optimization & Reporting | 1. Formatted markdown OOF report to print per-class pixel stats table and dual means.<br>2. Wired `accum_steps` (4) in `train_refiner_fold` and `train_segmenter_fold`. Scales loss by $1/4$ and updates optimizer every 4 minibatches. | **Optimization control flow**. Implements blueprint-specified effective batch size 16 ($4 \times 4$). Total minibatches (800) and forward/backward passes are preserved. |
| [`tests/v18/test_v18_synthetic.py`](file:///c:/Projects/SAT-EURS/tests/v18/test_v18_synthetic.py) | Unit Tests | Added `test_absent_class_metric_handling` and `test_gradient_accumulation_and_stage_contracts`. | **Test harness only**. Expands regression coverage from 12 to 14 CPU tests. |
| Baseline files (V16/V17 ZIPs, manifests, tables) | Unchanged | Verified read-only byte-for-byte reference SHA256 hashes. | **Zero mutation**. |

### Baseline Cryptographic Hash Verification
| Baseline File | Verification Status | Reference SHA256 Hash |
| :--- | :---: | :--- |
| `deploy/submission_v16_apex_sota.zip` | MATCH | `943369FCDFF398C1C18C9B43D4AA792BAA1E343914A3DC6F186CA52BB8CED6DF` |
| `deploy/submission_v17_dual_backbone_sota.zip` | MATCH | `50F85EC2111AD6D62A03D718FB8FE09FE4E75201D7CDF359C9A81BAAC138CC4E` |
| `splits/manifest_5fold.json` | MATCH | `CFBA12CAAABC79BB092F7DFA88405482E2C62C0AF2F4ED296B33F766EAEBD179` |
| `checkpoints/track1_screening/tbl19_20sc_clean.pt` | MATCH | `2C35DF3B0F5D3317C3672B2867E0059B893ACECD8699C5A2271383A1179D9562` |
| `checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt` | MATCH | `097CC9380FE11CA672F712481EC8187BBCE2E3EE84F1B17D2DC2288793B0E9A4` |

---

## 4. Final Training-Loop Gate Conclusion

| Verification Item | Requirement | Evidence | Status |
| :---: | :--- | :--- | :---: |
| **Refiner Accumulation** | `batch_size=4`, `accum_steps=4`, loss scaling | Verified in `train_refiner_fold`, loss scaled by $1/4$ | **PASS** |
| **Segmenter Accumulation** | `batch_size=4`, `accum_steps=4`, loss scaling | Verified in `train_segmenter_fold`, loss scaled by $1/4$ | **PASS** |
| **Optimizer & Scheduler** | 200 updates per epoch; scheduler per epoch | Updates trigger every 4 minibatches; `scheduler.step()` once/epoch | **PASS** |
| **Mathematical Equivalence** | Gradients equal combined-batch calculation | Unit test verified: `allclose(atol=1e-6)` on CPU | **PASS** |
| **Partial Window Handling** | Trailing minibatches preserved | Guard verified: updates trigger at end of loader | **PASS** |
| **CPU Test Suite** | All synthetic tests pass | 14/14 tests pass; dry run passes in 0.20s | **PASS** |
| **Baseline Hashes** | Zero baseline mutation | All 5 reference SHA256 hashes verified byte-for-byte | **PASS** |

### Final Verdict: **`PASS`**

The V18 codebase is verified and ready for the planned **five-fold, 15-epoch training run** across both stages on the RTX 5090.

> [!IMPORTANT]
> **Scoring Parity Caveat:** While offline components (NAFNet refiner, DINOv3 dual-encoder segmenter, and $90\%$ score $S_{\text{exact\_90pct}} = 0.60 Q_{\text{seg}} + 0.30 Q_{\text{FR}}$) are mathematically verified, final performance on unseen competition test data is unknown until training completes, and official server evaluation behavior regarding the coverage penalty IoU-floor and $Q_{\text{NR}}$ underwater metrics remains closed-source and unverified locally.

**No GPU training has been launched.** The pipeline awaits your explicit authorization.
