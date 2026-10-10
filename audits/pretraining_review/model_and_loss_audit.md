# EURS-Net V2 Model Architecture & Loss Formulation Audit

**Network Architecture:** [`models/eurs_net_v2.py`](file:///C:/Projects/SAT-EURS/models/eurs_net_v2.py)  
**Decoders:** [`models/decoders.py`](file:///C:/Projects/SAT-EURS/models/decoders.py)  
**Loss Functions:** [`losses/restoration_loss.py`](file:///C:/Projects/SAT-EURS/losses/restoration_loss.py), [`losses/multi_task_loss.py`](file:///C:/Projects/SAT-EURS/losses/multi_task_loss.py)  

---

## 1. Network Sub-Modules & Parameter Distribution

| Component | Class | Parameters | Function & Design Rationale |
| :--- | :--- | :--- | :--- |
| **0. Pre-Net** | `FastWhiteBalance` | **6** | Learnable 3-channel affine correction ($x \cdot \alpha + \beta$). Initialized as identity ($\alpha=1, \beta=0$). Rapidly absorbs global color cast. |
| **1. Encoder** | `SharedBackbone` | **268,208** | MobileNetV2-style inverted bottleneck (MBConv) stages with Coordinate Attention at stages 2 & 3. Fuses multi-scale spatial context. |
| **2. Restoration Branch** | `ProgressiveRestorationDecoder` | **51,779** | Multi-scale guided restoration decoder fusing F128, F256, and Stem skips. Predicts bounded residual $\Delta I = 0.20 \cdot \tanh(\text{Head})$ at 512x512, upsamples to 1024x1024, adds to $wb\_x$, clamps to $[0, 1]$. Zero-initialized head. |
| **3. Segmentation Branch** | `ProgressiveSegmentationDecoder` | **91,717** | Progressive top-down FPN fusing F32 $\to$ F64 $\to$ F128 $\to$ F256. Includes Small-Object Neck (256x256) and Boundary Refinement Neck (512x512) with Coordinate Attention. Predicts 5-class logits. |
| **Total Model** | `EURSNetV2` | **411,710** | **Fully compliant:** $\le 1,000,000$ parameter ceiling ($41.17\%$ budget). |

---

## 2. Restoration Residual Head Parameterization

In [`models/decoders.py`](file:///C:/Projects/SAT-EURS/models/decoders.py):

```python
if self.bounded_residual:
    delta_512 = self.bound_scale * torch.tanh(self.head(x))
else:
    delta_512 = self.head(x)
delta_1024 = F.interpolate(delta_512, size=raw_input.shape[2:], mode='bilinear', align_corners=False)

base = wb_input if wb_input is not None else raw_input
restored = torch.clamp(base + delta_1024, 0.0, 1.0)
```

* **Bound Scale:** $\text{bound\_scale} = 0.20$.
* **Head Zero-Initialization:**
  ```python
  nn.init.zeros_(self.head.weight)
  nn.init.zeros_(self.head.bias)
  ```
  At step 0, $\tanh(0) = 0$, guaranteeing that $I_{\text{restored}} = \text{clamp}(FWB(x), 0, 1)$. This provides a mathematically safe baseline of $15.06\text{ dB}$ (FWB quality) and prevents initial negative residual collapse.

---

## 3. Loss Formulations & Multi-Task Objective

### Multi-Task Objective
$$L_{\text{total}} = w_{\text{seg}} L_{\text{seg}} + w_{\text{rest}} L_{\text{rest}}$$
* Default in [`losses/loss_config.py`](file:///C:/Projects/SAT-EURS/losses/loss_config.py): $w_{\text{seg}} = 1.0$, $w_{\text{rest}} = 0.20$.
* Balancing rationale: Keeps restoration gradients from dominating shared encoder representations, protecting the rare-class segmentation accuracy (especially Shark IoU $\ge 0.1472$).

### Segmentation Loss ($L_{\text{seg}}$)
Combines Focal Loss ($\gamma = 2.0$) with Soft Dice Loss, weighted by class-frequency weights:
$$w_c = \frac{1}{\sqrt{\max(f_c, \epsilon)}} \times \text{rare\_boost}_c, \quad \text{rare\_boost} = [1.0, 1.0, 1.5, 2.0, 2.0]$$
Normalised to $\text{mean}(\alpha) = 1.0$, giving elevated gradient emphasis to Fish (1.5x), Shark (2.0x), and Other Marine (2.0x).

### Restoration Loss ($L_{\text{rest}}$)
$$L_{\text{rest}} = w_{\text{charb}} L_{\text{charb}} + w_{\text{ssim}} L_{\text{msssim}} + w_{\text{color}} L_{\text{color}} + w_{\text{lap}} L_{\text{lap}}$$
1. **Charbonnier Loss ($w=1.0$):** Element-wise smooth L1 metric ($\epsilon=10^{-3}$).
2. **Multi-Scale SSIM ($w=0.50$ first 15 epochs $\to 1.00$):** Scale-dependent structure metric with $0.5$ early damping to prevent optimizer entrapment on severe color casts.
3. **Color Angle / Cosine Loss ($w=0.50$):** $1.0 - \text{cosine\_similarity}(p, t)$, penalizes chromatic divergence directly without being distorted by luminance scaling.
4. **Laplacian Edge Loss ($w=0.10$):** Depthwise $3 \times 3$ Laplacian filter with reflection padding, directly driving high-frequency sharpness preservation.
