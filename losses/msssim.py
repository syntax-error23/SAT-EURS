"""
Multi-Scale Structural Similarity (MS-SSIM) Loss
Pinned, self-contained PyTorch implementation for normalized [0, 1] RGB tensors.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_filter(channel: int, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    g1d = _fspecial_gauss_1d(kernel_size, sigma)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    kernel = g2d.expand(channel, 1, kernel_size, kernel_size).contiguous()
    return kernel


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes SSIM luminance and contrast-structure maps."""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    channels = x.shape[1]
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq).clamp(min=0.0)
    sigma_y_sq = (F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq).clamp(min=0.0)
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    # Contrast-structure map
    cs_map = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    # Luminance map
    l_map = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)

    return l_map, cs_map


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    weights: Optional[List[float]] = None,
    kernel_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    Computes Multi-Scale Structural Similarity (MS-SSIM) between x and y in strict FP32.
    Args:
        x, y: [B, C, H, W] in [0, 1]
    Returns:
        Scalar MS-SSIM score in [0, 1].
    """
    device_type = x.device.type if x.device.type in ('cuda', 'cpu') else 'cpu'
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x = x.float()
        y = y.float()

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=x.device)

        b, c, h, w = x.shape
        window = _gaussian_filter(c, kernel_size, sigma).to(device=x.device, dtype=torch.float32)

        levels = len(weights)
        mcs_list = []

        for i in range(levels):
            l_map, cs_map = _ssim_per_channel(x, y, window, data_range=data_range)

            if i < levels - 1:
                mcs_list.append(torch.relu(cs_map.mean(dim=(1, 2, 3))))
                # Downsample by 2x for next scale
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
            else:
                # Final scale: multiply luminance with contrast-structure
                l_final = torch.relu(l_map.mean(dim=(1, 2, 3)))
                cs_final = torch.relu(cs_map.mean(dim=(1, 2, 3)))
                final_ssim = l_final * cs_final

        # Stack and weight: prod(cs_j ^ w_j) * (l_M ^ w_M)
        mcs_stack = torch.stack(mcs_list, dim=0)  # [levels - 1, B]
        weights_cs = weights_tensor[:-1].unsqueeze(1)  # [levels - 1, 1]

        # Add small epsilon to prevent NaN in power/grad
        eps = 1e-6
        powered = (mcs_stack + eps).clamp(max=1.0) ** weights_cs
        ms_ssim_val = torch.prod(powered, dim=0) * ((final_ssim + eps).clamp(max=1.0) ** weights_tensor[-1])
        return ms_ssim_val.mean()


class MSSSIMLoss(nn.Module):
    """
    MS-SSIM Loss = 1.0 - MS-SSIM(x, y) (strictly computed in FP32)
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = ms_ssim(pred, target, data_range=self.data_range)
        return 1.0 - score

