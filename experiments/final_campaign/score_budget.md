# SAT-EURS — Mathematical Score Budget to Target 0.700+

**Official Formula:**
$$\text{Composite} = 0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} + 0.10 \times Q_{\text{NR}}$$
$$\text{where } Q_{\text{FR}} = 0.45 \times \text{PSNR}_{\text{norm}} + 0.45 \times \text{SSIM} + 0.10 \times \text{Sharpness}$$
$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0}, 0.0, 1.0\right)$$

---

## 1. Verified Official Baseline (V28-A) Breakdown

- **Segmentation:**
  - $Q_{\text{seg}} = 0.570$ ($\text{mIoU} = 0.704$, Macro $F_1 = 0.810$)
  - $0.60 \times Q_{\text{seg}} = 0.60 \times 0.570 = \mathbf{0.3420}$
- **Full-Reference Restoration ($Q_{\text{FR}}$):**
  - PSNR = $25.48\text{ dB} \implies \text{PSNR}_{\text{norm}} = \frac{25.48 - 15.0}{20.0} = 0.5240$
  - SSIM = $0.8290$
  - Sharpness Ratio = $0.9730$
  - $Q_{\text{FR}} = 0.45 \times 0.5240 + 0.45 \times 0.8290 + 0.10 \times 0.9730 = 0.2358 + 0.37305 + 0.0973 = \mathbf{0.70615}$
  - $0.30 \times Q_{\text{FR}} = 0.30 \times 0.70615 = \mathbf{0.2118}$
- **No-Reference Restoration ($Q_{\text{NR}}$):**
  - $\text{Composite} = 0.634 \implies 0.10 \times Q_{\text{NR}} = 0.634 - (0.3420 + 0.2118) = \mathbf{0.0802} \implies Q_{\text{NR}} \approx 0.8020$
- **Total Composite:** $0.3420 + 0.2118 + 0.0802 = \mathbf{0.6340}$

---

## 2. Competitive Benchmarks from Official Leaderboard

| Team / Submission | Composite | $Q_{\text{seg}}$ | mIoU | $F_1$ | PSNR (dB) | SSIM | Sharpness | Implied $Q_{\text{FR}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UW-IPL (#19050)** | **0.753** | 0.688 | 0.786 | 0.876 | 31.15 | 0.911 | 0.906 | **0.8639** (+0.158 vs V28-A) |
| **PicusLab/ENEA (#19012)** | **0.728** | 0.671 | 0.775 | 0.866 | 29.35 | 0.880 | 0.912 | **0.8105** (+0.104 vs V28-A) |
| **aimslabs (#19122)** | **0.706** | 0.649 | 0.760 | 0.854 | 28.14 | 0.893 | 0.908 | **0.7883** (+0.082 vs V28-A) |
| **APC_TDC (#19187)** | **0.686** | 0.610 | 0.732 | 0.833 | 29.03 | 0.891 | 0.874 | **0.8040** (+0.098 vs V28-A) |
| **MIT-BLR V28-A (Our Best)** | **0.634** | 0.570 | 0.704 | 0.810 | 25.48 | 0.829 | 0.973 | **0.7062** (Baseline) |

---

## 3. Scenarios to Reach the 0.700 Target

Assuming $Q_{\text{NR}}$ remains constant around $0.802$ ($0.10 \times Q_{\text{NR}} = 0.0802$), we must generate:
$$0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} \ge 0.7000 - 0.0802 = \mathbf{0.6198}$$

### Scenario A: Zero Restoration Improvement ($Q_{\text{FR}} = 0.7062$)
- $0.30 \times Q_{\text{FR}} = 0.2118$
- Required $0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2118 = 0.4080 \implies \mathbf{Q_{\text{seg}} \ge 0.6800}$ (+0.110 gain)
- *Assessment:* Very high risk. Segmentation alone would need to match the #1 team in the world.

### Scenario B: Moderate Restoration Gain (aimslabs level: PSNR 28.1 dB, SSIM 0.893, $Q_{\text{FR}} \approx 0.788$)
- $0.30 \times Q_{\text{FR}} = 0.2364$ (+0.0246 to Composite)
- Required $0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2364 = 0.3834 \implies \mathbf{Q_{\text{seg}} \ge 0.6390}$ (+0.069 gain)
- *Assessment:* Highly feasible balanced target.

### Scenario C: Strong Restoration Gain (PicusLab/ENEA level: PSNR 29.35 dB, SSIM 0.880, $Q_{\text{FR}} \approx 0.811$)
- $0.30 \times Q_{\text{FR}} = 0.2433$ (+0.0315 to Composite)
- Required $0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2433 = 0.3765 \implies \mathbf{Q_{\text{seg}} \ge 0.6275}$ (+0.0575 gain)
- *Assessment:* Very attractive path.

### Scenario D: High Restoration Gain (UW-IPL level: PSNR 31.15 dB, SSIM 0.911, $Q_{\text{FR}} \approx 0.864$)
- $0.30 \times Q_{\text{FR}} = 0.2592$ (+0.0474 to Composite)
- Required $0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2592 = 0.3606 \implies \mathbf{Q_{\text{seg}} \ge 0.6010}$ (+0.031 gain)
- *Assessment:* The most robust path. A modest segmentation improvement (+0.031) combined with strong restoration achieves the 0.700 target comfortably.

---

## 4. Workstream Priority Matrix

1. **Restoration (Highest ROI):** We lag top teams by **+5.67 dB PSNR** and **+0.082 SSIM**. Closing half this gap directly adds $+0.025$ to $+0.035$ to the official composite.
2. **Segmentation (Coupled Multiplier):** Moving $Q_{\text{seg}}$ from $0.570 \to 0.620 - 0.640$ adds $+0.030$ to $+0.042$ to the composite.
3. **Joint Composite:** $0.634 + 0.030 (\text{Rest}) + 0.040 (\text{Seg}) = \mathbf{0.704}$.
