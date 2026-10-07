"""
Non-Reference Quality Metric (Q_NR) Placeholder Stub
Status: UNVERIFIED EXTERNAL DEPENDENCY
Because official evaluation scripts (e.g., UIQM, UCIQE, NIQE, or BRISQUE specific to MaCVi)
have not yet been provided in the repository, this module serves as an explicit isolated stub.
"""

from typing import Dict, Any
import numpy as np
import torch


class NonReferenceQualityEvaluator:
    """
    Explicit stub for Non-Reference Image Quality (Q_NR).
    Official competition parity status: UNVERIFIED / PENDING OFFICIAL CODE.
    """
    def __init__(self):
        self.scores = []

    def reset(self):
        self.scores = []

    def update(self, pred_rgb: torch.Tensor):
        """
        Takes restored RGB tensor [B, 3, H, W] in [0, 1].
        Computes a provisional sharpness/contrast surrogate or assigns neutral baseline.
        """
        pred_np = pred_rgb.detach().cpu().numpy()
        b = pred_np.shape[0]
        for i in range(b):
            # Provisional heuristic: normalized standard deviation of image as contrast surrogate
            contrast = float(np.std(pred_np[i]))
            # Bound surrogate in [0.0, 1.0]
            surrogate_score = min(1.0, max(0.0, contrast * 2.5))
            self.scores.append(surrogate_score)

    def compute(self) -> Dict[str, Any]:
        mean_score = float(np.mean(self.scores)) if self.scores else 0.50
        return {
            'Q_NR': round(mean_score, 4),
            'status': 'PROVISIONAL_STUB',
            'official_parity': False,
            'note': 'Official MaCVi Q_NR script pending import into repository.'
        }
