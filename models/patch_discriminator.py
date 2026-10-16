"""
PatchGAN Binary Classifier Discriminator & Statistical Distribution Alignment Losses
for EURS-Net V5 (MaCVi @ WACV 2027 Track 1 & Track 2).

Implements:
1. `PatchGANDiscriminator`:
   - Conditional Markovian PatchGAN binary classifier (`Real Clean = 1`, `Fake Restored = 0` per patch).
   - Takes `concat([degraded_rgb, target_or_restored_rgb], dim=1)` ([B, 6, H, W]) and classifies
     overlapping 70x70 local patches as real clean vs. generator-restored.
   - Equipped with Spectral Normalization (`spectral_norm`) on all convolutional layers to
     guarantee Lipschitz stability during joint restoration + segmentation training.
   - Outputs both patch binary classification logits (`[B, 1, H_p, W_p]`) and multi-scale intermediate
     feature maps for Discriminator Feature-Matching Loss.
2. `PatchGANLoss`:
   - Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`, label-smoothed `0.95` real / `0.0` fake)
     + Least-Squares GAN (`LSGAN`) stabilization + Multi-Layer Feature-Matching L1 Loss.
3. `MeanVarianceDistributionLoss`:
   - Computes overall image RGB mean & variance AND individual class RGB mean & variance (`c in 0..4`),
     penalizing the Euclidean + Wasserstein-2 (`W2`) distance between `restored_rgb` and `clean_target`
     both globally and per semantic class, as well as anchoring class-to-overall variance ratios.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    70x70 Receptive-Field Conditional PatchGAN Binary Classifier:
    - Input: `degraded_rgb` [B, 3, H, W] and `candidate_rgb` [B, 3, H, W] (concatenated to 6 channels).
    - Downsamples to 512x512 before 4 strided convolutional blocks -> [B, 1, 30, 30] patch grid.
    - Each output unit is a binary classifier logit (`> 0` => Real Clean Patch, `< 0` => Fake Restored Patch).
    - Used strictly during training (zero inference FLOPs/parameter overhead for Track 2 ONNX).
    """
    def __init__(self, in_channels: int = 6, base_channels: int = 48):
        super().__init__()
        c1 = base_channels          # 48
        c2 = base_channels * 2      # 96
        c3 = base_channels * 4      # 192
        c4 = base_channels * 4      # 192

        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(c2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(c3, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(c4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Binary classification head per patch (logit for BCEWithLogitsLoss)
        self.patch_classifier = spectral_norm(
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )

    def forward(
        self,
        degraded_rgb: torch.Tensor,
        candidate_rgb: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.cat([degraded_rgb, candidate_rgb], dim=1)
        # Operate at 512x512 native half-scale for fast, crisp 70x70 patch classification
        if x.shape[2] > 512 or x.shape[3] > 512:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        patch_logits = self.patch_classifier(f4)

        if return_features:
            return patch_logits, [f1, f2, f3, f4]
        return patch_logits, []


class PatchGANLoss(nn.Module):
    """
    Binary Patch Classifier GAN Loss + Multi-Scale Feature Matching Loss (strict FP32):
      - `compute_d_loss(d_model, degraded, clean, restored_detached)`:
          Trains PatchGAN binary classifier to predict 1 (0.95 smoothed) on real `(degraded, clean)`
          patches and 0 on fake `(degraded, restored.detach())` patches.
      - `compute_g_loss(d_model, degraded, clean, restored)`:
          Trains Generator `EURSNetV5` to fool the PatchGAN binary classifier (`target = 1.0`)
          AND match multi-layer intermediate PatchGAN feature maps of the real clean image.
    """
    def __init__(self, adv_weight: float = 0.05, fm_weight: float = 0.25, real_smooth: float = 0.95):
        super().__init__()
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.real_smooth = real_smooth
        self.bce = nn.BCEWithLogitsLoss()

    def compute_d_loss(
        self,
        discriminator: PatchGANDiscriminator,
        degraded_rgb: torch.Tensor,
        clean_rgb: torch.Tensor,
        restored_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = degraded_rgb.device.type if degraded_rgb.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            deg_f = degraded_rgb.detach().float()
            clean_f = clean_rgb.detach().float()
            fake_f = restored_rgb.detach().float()

            real_logits, _ = discriminator(deg_f, clean_f, return_features=False)
            fake_logits, _ = discriminator(deg_f, fake_f, return_features=False)

            real_targets = torch.full_like(real_logits, self.real_smooth)
            fake_targets = torch.zeros_like(fake_logits)

            # Binary classification loss (BCEWithLogits) + LSGAN hinge stabilization
            loss_real_bce = self.bce(real_logits, real_targets)
            loss_fake_bce = self.bce(fake_logits, fake_targets)
            loss_real_ls = F.mse_loss(torch.sigmoid(real_logits), real_targets)
            loss_fake_ls = F.mse_loss(torch.sigmoid(fake_logits), fake_targets)

            d_loss = 0.5 * (loss_real_bce + loss_fake_bce) + 0.5 * (loss_real_ls + loss_fake_ls)

            with torch.no_grad():
                acc_real = float((real_logits > 0.0).float().mean().item())
                acc_fake = float((fake_logits < 0.0).float().mean().item())

            return d_loss, {
                'd_loss': float(d_loss.detach().item()),
                'd_acc_real': acc_real,
                'd_acc_fake': acc_fake,
                'd_acc_total': 0.5 * (acc_real + acc_fake),
            }

    def compute_g_loss(
        self,
        discriminator: PatchGANDiscriminator,
        degraded_rgb: torch.Tensor,
        clean_rgb: torch.Tensor,
        restored_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = degraded_rgb.device.type if degraded_rgb.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            deg_f = degraded_rgb.detach().float()
            clean_f = clean_rgb.detach().float()
            rest_f = restored_rgb.float()

            with torch.no_grad():
                _, real_feats = discriminator(deg_f, clean_f, return_features=True)

            fake_logits, fake_feats = discriminator(deg_f, rest_f, return_features=True)
            real_targets = torch.ones_like(fake_logits)

            g_adv_bce = self.bce(fake_logits, real_targets)
            g_adv_ls = F.mse_loss(torch.sigmoid(fake_logits), real_targets)
            g_adv = 0.5 * g_adv_bce + 0.5 * g_adv_ls

            fm_loss = (rest_f * 0.0).sum()
            for rf, ff in zip(real_feats, fake_feats):
                fm_loss = fm_loss + F.l1_loss(ff, rf.detach())
            fm_loss = fm_loss / max(1, len(real_feats))

            total_g_gan = self.adv_weight * g_adv + self.fm_weight * fm_loss
            return total_g_gan, {
                'g_adv_loss': float(g_adv.detach().item()),
                'g_fm_loss': float(fm_loss.detach().item()),
                'g_gan_total': float(total_g_gan.detach().item()),
            }


class MeanVarianceDistributionLoss(nn.Module):
    """
    Explicit Overall & Per-Class Mean/Variance Statistical Distance Loss (strict FP32):
      1. Computes overall image RGB mean `mu_overall` and variance `var_overall` (and std `sigma_overall`)
         for `restored_rgb` vs `clean_target`, minimizing their Wasserstein-2 Gaussian distance:
             ||mu_rest - mu_clean||_1 + ||sigma_rest - sigma_clean||_1 + ||var_rest - var_clean||_1
      2. Computes per-class RGB mean `mu_c` and variance `var_c` for every class `c in {0, 1, 2, 3, 4}`
         present in `mask_target` (weighting rare classes 2, 3, 4 by 2.0x), minimizing per-class
         mean & variance distance AND matching the class-to-overall relative distribution offset
         (`(mu_c - mu_overall)`, `sigma_c / sigma_overall`).
    """
    def __init__(self, num_classes: int = 5, min_pixels: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.min_pixels = min_pixels
        self.register_buffer(
            "class_weights",
            torch.tensor([0.6, 0.8, 1.5, 2.2, 2.2], dtype=torch.float32),
        )

    def forward(
        self,
        restored_rgb: torch.Tensor,
        clean_target: torch.Tensor,
        mask_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = restored_rgb.device.type if restored_rgb.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            r = restored_rgb.float()
            c = clean_target.float()

            # 1. Overall per-image RGB mean, variance, and standard deviation [B, 3]
            mu_r = r.mean(dim=(2, 3))
            mu_c = c.mean(dim=(2, 3))
            var_r = r.var(dim=(2, 3), unbiased=False)
            var_c = c.var(dim=(2, 3), unbiased=False)
            std_r = torch.sqrt(var_r + 1e-6)
            std_c = torch.sqrt(var_c + 1e-6)

            loss_overall_mean = F.l1_loss(mu_r, mu_c)
            loss_overall_var = F.l1_loss(var_r, var_c) + F.l1_loss(std_r, std_c)
            loss_overall = loss_overall_mean + loss_overall_var

            # 2. Individual class RGB mean & variance + class-vs-overall distance consistency
            loss_class = (r * 0.0).sum()
            valid_classes = 0

            if mask_target is not None:
                # Subsample to 256x256 for fast per-class statistics
                if r.shape[2] > 256:
                    r_sub = F.avg_pool2d(r, kernel_size=4, stride=4)
                    c_sub = F.avg_pool2d(c, kernel_size=4, stride=4)
                    m_sub = F.interpolate(
                        mask_target.unsqueeze(1).float(), size=r_sub.shape[2:], mode='nearest'
                    ).squeeze(1).long()
                else:
                    r_sub, c_sub, m_sub = r, c, mask_target.long()

                r_flat = r_sub.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
                c_flat = c_sub.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
                m_flat = m_sub.reshape(-1)                         # [N]

                batch_mu_r = r_flat.mean(dim=0)
                batch_mu_c = c_flat.mean(dim=0)
                batch_std_r = torch.sqrt(r_flat.var(dim=0, unbiased=False) + 1e-6)
                batch_std_c = torch.sqrt(c_flat.var(dim=0, unbiased=False) + 1e-6)

                for cls_id in range(self.num_classes):
                    sel = (m_flat == cls_id)
                    if int(sel.sum().item()) < self.min_pixels:
                        continue
                    r_cls = r_flat[sel]
                    c_cls = c_flat[sel]

                    mu_r_cls = r_cls.mean(dim=0)
                    mu_c_cls = c_cls.mean(dim=0)
                    var_r_cls = r_cls.var(dim=0, unbiased=False)
                    var_c_cls = c_cls.var(dim=0, unbiased=False)
                    std_r_cls = torch.sqrt(var_r_cls + 1e-6)
                    std_c_cls = torch.sqrt(var_c_cls + 1e-6)

                    # Direct class mean & variance distance
                    d_cls = (
                        F.l1_loss(mu_r_cls, mu_c_cls)
                        + F.l1_loss(var_r_cls, var_c_cls)
                        + F.l1_loss(std_r_cls, std_c_cls)
                    )
                    # Class-vs-overall relative mean shift & variance ratio distance
                    rel_shift_r = mu_r_cls - batch_mu_r
                    rel_shift_c = mu_c_cls - batch_mu_c
                    rel_std_r = std_r_cls / (batch_std_r + 1e-4)
                    rel_std_c = std_c_cls / (batch_std_c + 1e-4)
                    d_rel = F.l1_loss(rel_shift_r, rel_shift_c) + 0.25 * F.l1_loss(rel_std_r, rel_std_c)

                    w = float(self.class_weights[cls_id].item())
                    loss_class = loss_class + w * (d_cls + 0.5 * d_rel)
                    valid_classes += 1

                if valid_classes > 0:
                    loss_class = loss_class / float(valid_classes)

            total_dist_loss = loss_overall + 0.75 * loss_class
            return total_dist_loss, {
                'loss_dist_overall': float(loss_overall.detach().item()),
                'loss_dist_class': float(loss_class.detach().item()),
                'loss_dist_total': float(total_dist_loss.detach().item()),
            }
