"""
Compound Loss for Track 2 SOTA Architecture
Combines:
  1. Present-Class Lovasz-Softmax Loss (Direct continuous surrogate for mIoU)
  2. Soft-Dice Loss (Maximizes Macro-F1 across marine fauna)
  3. Class-Weighted Cross-Entropy (Balanced gradient)
  4. Teacher Knowledge Distillation (KL Divergence from Track 1 SOTA Teacher)
  5. Restoration Loss (L1 + SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from losses.segmentation_loss import SoftDiceLoss


class Track2SOTALoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        w_lovasz: float = 0.40,
        w_dice: float = 0.30,
        w_ce: float = 0.30,
        w_kd: float = 0.35,
        w_rest: float = 0.25,
        temperature: float = 3.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_lovasz = w_lovasz
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.w_kd = w_kd
        self.w_rest = w_rest
        self.temperature = temperature

        # Class weights: [Background, Coral, Fish, Shark/Ray, Other/Macro-fauna]
        # Balanced weights that give proper emphasis to rare fauna
        weights = torch.tensor([0.70, 0.90, 1.50, 3.00, 3.50], dtype=torch.float32)
        self.register_buffer("class_weights", weights)

        self.lovasz_loss = PresentClassLovaszSoftmaxLoss()
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        student_restored: torch.Tensor,
        student_logits: torch.Tensor,
        clean_target: torch.Tensor,
        mask_target: torch.Tensor,
        teacher_logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Restoration Loss (L1)
        loss_rest = F.l1_loss(student_restored, clean_target)

        # 2. Segmentation Losses
        loss_lovasz = self.lovasz_loss(student_logits, mask_target)
        loss_dice = self.dice_loss(student_logits, mask_target)
        loss_ce = self.ce_loss(student_logits, mask_target)

        loss_seg = (
            self.w_lovasz * loss_lovasz
            + self.w_dice * loss_dice
            + self.w_ce * loss_ce
        )

        # 3. Knowledge Distillation Loss (Properly Per-Pixel Normalized)
        if teacher_logits is not None:
            if teacher_logits.shape[2:] != student_logits.shape[2:]:
                t_logits = F.interpolate(
                    teacher_logits,
                    size=student_logits.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            else:
                t_logits = teacher_logits

            t = self.temperature
            p_s = F.log_softmax(student_logits / t, dim=1)
            p_t = F.softmax(t_logits / t, dim=1)
            b, c, h, w = student_logits.shape
            loss_kd = (F.kl_div(p_s, p_t, reduction="sum") / (b * h * w)) * (t * t)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device)

        # Total Loss
        total_loss = loss_seg + self.w_kd * loss_kd + self.w_rest * loss_rest

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_seg": float(loss_seg.item()),
            "loss_lovasz": float(loss_lovasz.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_kd": float(loss_kd.item()) if teacher_logits is not None else 0.0,
            "loss_rest": float(loss_rest.item())
        }
        return total_loss, metrics
