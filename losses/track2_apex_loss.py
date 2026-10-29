"""
Official-Metric-Aligned Joint Loss for `Track2-Apex-Distill-V2` (`Track2ApexOrin`):
  - Fix 2: Direct Auxiliary Cross-Entropy Supervision (`0.20 * L_cond_ce`) on the 200 `(wt, depth)` bins.
  - Fix 4: Separate segmentation supervision for Set A (True GT: class weights `[1.0, 1.2, 2.0, 3.0, 3.0]`)
    vs Set B/C (Teacher Pseudo-Labels: Unweighted `[1.0, 1.0, 1.0, 1.0, 1.0]` Label-Smoothed CE `eps=0.05`
    to prevent compound rare-class bias explosion), combined with Differentiable Soft-mIoU (Jaccard),
    Soft Macro-F1 (Dice), and Absent-Class False-Positive Suppression Loss.
  - Restoration Loss: Charbonnier L1 + Multi-Scale SSIM + Sobel Edge-Gradient Matching Loss.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Track2ApexLoss(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        # Set A (True GT) class weights
        self.register_buffer(
            "gt_class_weights",
            torch.tensor([1.0, 1.2, 2.0, 3.0, 3.0], dtype=torch.float32)
        )
        # Sobel kernels for edge/sharpness alignment
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", ky.repeat(3, 1, 1, 1))

    def _soft_iou_and_f1_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Differentiable Soft-mIoU, Soft-Macro-F1, and Absent-Class False-Positive Penalty.
        """
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        valid = (targets >= 0) & (targets < c)
        safe_t = torch.where(valid, targets, torch.zeros_like(targets))
        one_hot = F.one_hot(safe_t, num_classes=c).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = probs * valid.unsqueeze(1).float()

        # Pool across batch or per-image
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dim=dims)
        pred_sum = torch.sum(probs, dim=dims)
        gt_sum = torch.sum(one_hot, dim=dims)
        union = pred_sum + gt_sum - inter

        # Present classes mask in batch
        present = (gt_sum > 10.0).float()
        iou = (inter + 1.0) / (union + 1.0)
        f1 = (2.0 * inter + 1.0) / (pred_sum + gt_sum + 1.0)

        # Weight rare classes slightly higher in batch IoU/F1
        cls_w = torch.tensor([1.0, 1.0, 1.3, 1.6, 1.6], device=logits.device)
        active_w = cls_w * torch.clamp(present, min=0.2)
        loss_iou = 1.0 - torch.sum(iou * active_w) / torch.sum(active_w)
        loss_f1 = 1.0 - torch.sum(f1 * active_w) / torch.sum(active_w)

        # Per-image Absent-Class False-Positive Suppression for classes 1..4
        # If a class has 0 GT pixels in image b, penalize mean + top-k probability of that class in image b
        gt_per_img = torch.sum(one_hot, dim=(2, 3))        # [B, C]
        absent_mask = (gt_per_img == 0).float()            # [B, C]
        absent_mask[:, 0] = 0.0                            # Ignore background
        probs_flat = probs.view(b, c, -1)
        mean_fp = probs_flat.mean(dim=-1)                  # [B, C]
        # Top-64 pixel mean probability (approximates soft max without noisy single-pixel gradients)
        top_fp, _ = torch.topk(probs_flat, k=min(64, probs_flat.shape[-1]), dim=-1)
        max_fp = top_fp.mean(dim=-1)                       # [B, C]
        loss_absent = ((2.0 * mean_fp + 0.25 * max_fp) * absent_mask).sum() / max(1.0, absent_mask.sum().item())

        return loss_iou, loss_f1, loss_absent

    def _restoration_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Charbonnier L1 Loss
        diff = pred - target
        l_charb = torch.mean(torch.sqrt(diff * diff + 1e-6))

        # 2. Local Structural SSIM Loss (at 512x512 for speed & multi-scale structure)
        p_sub = F.avg_pool2d(pred, kernel_size=2, stride=2)
        t_sub = F.avg_pool2d(target, kernel_size=2, stride=2)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_p = F.avg_pool2d(p_sub, kernel_size=7, stride=1, padding=3)
        mu_t = F.avg_pool2d(t_sub, kernel_size=7, stride=1, padding=3)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p2 = F.avg_pool2d(p_sub * p_sub, 7, 1, 3) - mu_p2
        sig_t2 = F.avg_pool2d(t_sub * t_sub, 7, 1, 3) - mu_t2
        sig_pt = F.avg_pool2d(p_sub * t_sub, 7, 1, 3) - mu_pt
        ssim_map = ((2.0 * mu_pt + c1) * (2.0 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2))
        l_ssim = 1.0 - ssim_map.mean()

        # 3. Sobel Edge-Gradient Loss (drives Sharpness Ratio toward 1.0)
        gx_p = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        gx_t = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        gy_t = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        l_grad = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

        return 1.2 * l_charb + 0.40 * l_ssim + 0.20 * l_grad

    def forward(
        self,
        restored_pred: torch.Tensor,
        seg_logits: torch.Tensor,
        cond_logits: torch.Tensor,
        clean_target: torch.Tensor,
        seg_target: torch.Tensor,
        cond_target: torch.Tensor,
        is_gt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
          restored_pred: [B, 3, H, W]
          seg_logits:    [B, 5, H, W]
          cond_logits:   [B, 200]
          clean_target:  [B, 3, H, W]
          seg_target:    [B, H, W] long in 0..4
          cond_target:   [B] long in 0..199
          is_gt_mask:    [B] bool/float (1.0 for Set A True GT, 0.0 for Set B/C Teacher Pseudo-Labels)
        """
        b = seg_logits.shape[0]
        gt_flag = is_gt_mask.bool()
        pseudo_flag = ~gt_flag

        # Fix 4: Separate CE for True GT (Set A) vs Pseudo-Labels (Set B/C)
        loss_ce = torch.tensor(0.0, device=seg_logits.device)
        if gt_flag.any():
            ce_gt = F.cross_entropy(
                seg_logits[gt_flag],
                seg_target[gt_flag],
                weight=self.gt_class_weights,
                ignore_index=255
            )
            loss_ce = loss_ce + ce_gt * (gt_flag.float().sum() / b)

        if pseudo_flag.any():
            ce_pseudo = F.cross_entropy(
                seg_logits[pseudo_flag],
                seg_target[pseudo_flag],
                label_smoothing=0.05,
                ignore_index=255
            )
            loss_ce = loss_ce + ce_pseudo * (pseudo_flag.float().sum() / b)

        loss_iou, loss_f1, loss_absent = self._soft_iou_and_f1_loss(seg_logits, seg_target)
        loss_seg = 0.80 * loss_ce + 1.00 * loss_iou + 1.00 * loss_f1 + 0.45 * loss_absent

        # Fix 2: Direct (wt, depth) Condition Router Supervision
        loss_cond = F.cross_entropy(cond_logits, cond_target, label_smoothing=0.05)

        # Restoration Loss
        loss_rest = self._restoration_loss(restored_pred, clean_target)

        total_loss = loss_seg + 0.85 * loss_rest + 0.20 * loss_cond

        metrics = {
            "loss": float(total_loss.item()),
            "seg": float(loss_seg.item()),
            "iou_l": float(loss_iou.item()),
            "f1_l": float(loss_f1.item()),
            "abs_l": float(loss_absent.item()),
            "rest": float(loss_rest.item()),
            "cond": float(loss_cond.item()),
        }
        return total_loss, metrics
