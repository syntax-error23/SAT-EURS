"""
SAT-EURS Track 1 — V18 Nested Scene-Level Class-Logit Calibration
Optimizes class logit offsets (b0..b4) to maximize Q_seg = Macro F1 * mIoU * coverage_penalty.
Evaluates offsets strictly on held-out subscenes within the training split.
Never accesses validation scenes or Dev test ground truth.
"""

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


class V18LogitCalibrator:
    """
    Nested cross-scene logit offset optimizer.
    Finds additive offsets b in R^5 such that argmax(logits + b) maximizes Q_seg
    while preserving 100% coverage (penalty = 1.0).
    """
    def __init__(
        self,
        num_classes: int = 5,
        search_range_rare: Tuple[float, float] = (0.0, 0.30),
        num_steps: int = 7
    ):
        self.num_classes = num_classes
        self.search_range_rare = search_range_rare
        self.num_steps = num_steps
        self.best_offsets = np.zeros(num_classes, dtype=np.float32)

    def calibrate(
        self,
        scene_logits: Dict[str, np.ndarray],
        scene_targets: Dict[str, np.ndarray],
        nested_val_scenes: List[str]
    ) -> Dict[str, Any]:
        """
        Calibrates offsets on nested_val_scenes.
        Args:
            scene_logits: map from scene_name -> array of logits [N, 5, H, W] (downsampled or sampled)
            scene_targets: map from scene_name -> array of masks [N, H, W]
            nested_val_scenes: subset of training scenes reserved for calibration selection
        """
        assert len(nested_val_scenes) > 0, "nested_val_scenes must not be empty"

        # Grid search over rare class offsets (b3 for Shark/Ray, b4 for Macro-Fauna)
        b3_vals = np.linspace(self.search_range_rare[0], self.search_range_rare[1], self.num_steps)
        b4_vals = np.linspace(self.search_range_rare[0], self.search_range_rare[1], self.num_steps)

        best_score = -1.0
        best_b = (0.0, 0.0)
        best_metrics = {}

        for b3 in b3_vals:
            for b4 in b4_vals:
                offsets = np.array([0.0, 0.0, 0.0, b3, b4], dtype=np.float32)
                m = self.evaluate_offsets(scene_logits, scene_targets, nested_val_scenes, offsets)
                score = m["Q_seg"]

                if score > best_score:
                    best_score = score
                    best_b = (float(b3), float(b4))
                    best_metrics = m

        self.best_offsets = np.array([0.0, 0.0, 0.0, best_b[0], best_b[1]], dtype=np.float32)

        return {
            "best_offsets": self.best_offsets.tolist(),
            "best_b3": best_b[0],
            "best_b4": best_b[1],
            "best_q_seg": best_score,
            "metrics": best_metrics
        }

    def evaluate_offsets(
        self,
        scene_logits: Dict[str, np.ndarray],
        scene_targets: Dict[str, np.ndarray],
        scenes_to_eval: List[str],
        offsets: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates a specific offset candidate on the target scene subset."""
        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        b_vec = offsets.reshape(1, self.num_classes, 1, 1)

        for sc in scenes_to_eval:
            if sc not in scene_logits or sc not in scene_targets:
                continue
            l = scene_logits[sc] + b_vec
            t = scene_targets[sc]
            preds = np.argmax(l, axis=1)

            valid = (t >= 0) & (t < self.num_classes)
            y_t = t[valid].reshape(-1)
            y_p = preds[valid].reshape(-1)

            indices = self.num_classes * y_t + y_p
            counts = np.bincount(indices, minlength=self.num_classes ** 2)
            conf_mat += counts.reshape(self.num_classes, self.num_classes)

        # Compute metrics from confusion matrix
        tp = np.diag(conf_mat).astype(np.float64)
        fp = conf_mat.sum(axis=0) - tp
        fn = conf_mat.sum(axis=1) - tp

        ious = []
        f1s = []
        coverage = 1.0

        for c in range(self.num_classes):
            support = conf_mat[c, :].sum()
            pred_support = conf_mat[:, c].sum()

            if support == 0 and pred_support == 0:
                ious.append(1.0)
                f1s.append(1.0)
            elif support > 0 and tp[c] == 0:
                ious.append(0.0)
                f1s.append(0.0)
                coverage = 0.0  # Zero coverage collapse
            else:
                denom_iou = tp[c] + fp[c] + fn[c] + 1e-10
                denom_f1 = 2 * tp[c] + fp[c] + fn[c] + 1e-10
                ious.append(float(tp[c] / denom_iou))
                f1s.append(float((2 * tp[c]) / denom_f1))

        miou = float(np.mean(ious))
        macro_f1 = float(np.mean(f1s))
        q_seg = float(macro_f1 * miou * coverage)

        return {
            "mIoU": miou,
            "Macro_F1": macro_f1,
            "coverage_penalty": coverage,
            "Q_seg": q_seg,
            "class_ious": ious,
            "class_f1s": f1s
        }

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies calibrated offsets to logits tensor [B, 5, H, W]."""
        b_tensor = torch.from_numpy(self.best_offsets).to(device=logits.device, dtype=logits.dtype).view(1, 5, 1, 1)
        return logits + b_tensor
