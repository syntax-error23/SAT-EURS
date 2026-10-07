"""
Synchronized Spatial Transformation Module for SAT-EURS.

Ensures that spatial transformations (resizing, horizontal flips, vertical flips)
are applied identically and simultaneously across all three paired elements:
  - degraded image:   [3, H, W] (RGB)
  - clean_gt image:   [3, H, W] (RGB)
  - segmentation mask:[1, H, W] or [H, W] (Discrete integer labels 0..4)

Interpolation Contract:
  - Images (degraded, clean_gt): Bilinear interpolation with align_corners=False.
  - Masks: Nearest-neighbor interpolation, strictly preserving discrete class IDs.
"""

from typing import Tuple, Union, Sequence, Optional
import random
import torch
import torch.nn.functional as F


Tuple3Tensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Size2D = Tuple[int, int]


def _check_and_prep_image(img: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    """
    Validates image shape [C, H, W] or [B, C, H, W] and returns 4D tensor for interpolation.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(img).__name__}")
    if img.dim() == 3:
        # [C, H, W] -> [1, C, H, W]
        return img.unsqueeze(0), False
    elif img.dim() == 4:
        # Batched [B, C, H, W]
        return img, True
    else:
        raise ValueError(f"{name} must be 3D [C, H, W] or 4D [B, C, H, W], got shape {list(img.shape)}")


def _check_and_prep_mask(mask: torch.Tensor, name: str) -> Tuple[torch.Tensor, str, torch.dtype]:
    """
    Validates mask shape [H, W], [1, H, W], or [B, 1, H, W].
    Returns (mask_4d_float, original_format, original_dtype).
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    
    orig_dtype = mask.dtype
    
    if mask.dim() == 2:
        # [H, W] -> [1, 1, H, W]
        return mask.unsqueeze(0).unsqueeze(0).float(), "2D", orig_dtype
    elif mask.dim() == 3:
        # [1, H, W] -> [1, 1, H, W]
        return mask.unsqueeze(0).float(), "3D", orig_dtype
    elif mask.dim() == 4:
        # [B, 1, H, W]
        return mask.float(), "4D", orig_dtype
    else:
        raise ValueError(f"{name} must be 2D [H, W], 3D [1, H, W], or 4D [B, 1, H, W], got shape {list(mask.shape)}")


def _restore_mask_shape(mask_4d: torch.Tensor, orig_format: str, orig_dtype: torch.dtype) -> torch.Tensor:
    """
    Restores interpolated 4D mask back to its original dimensionality and integer dtype.
    """
    mask_converted = mask_4d.to(dtype=orig_dtype)
    if orig_format == "2D":
        return mask_converted.squeeze(0).squeeze(0)
    elif orig_format == "3D":
        return mask_converted.squeeze(0)
    else:
        return mask_converted


class SynchronizedTransform:
    """Abstract base class for all synchronized spatial transforms."""
    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        raise NotImplementedError


class SynchronizedResize(SynchronizedTransform):
    """
    Synchronously resizes degraded, clean_gt, and mask to a parameterized target size (H, W).

    Image interpolation: Bilinear with align_corners=False.
    Mask interpolation:  Nearest-neighbor, preserving discrete class IDs.

    Args:
        size (int or Tuple[int, int]): Desired output size.
            If int (e.g., 768 or 1024), output will be (size, size).
            If tuple (e.g., (384, 768) or (768, 1024)), output will be (height, width).
    """
    def __init__(self, size: Union[int, Sequence[int]]):
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"Size must be positive, got {size}")
            self.target_size: Size2D = (size, size)
        elif isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise ValueError(f"Size tuple/list must have length 2 (H, W), got {size}")
            h, w = int(size[0]), int(size[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Height and width must be positive, got {(h, w)}")
            self.target_size = (h, w)
        else:
            raise TypeError(f"Size must be int or Tuple[int, int], got {type(size).__name__}")

    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        # Validate spatial consistency prior to resize
        deg_h, deg_w = degraded.shape[-2], degraded.shape[-1]
        gt_h, gt_w = clean_gt.shape[-2], clean_gt.shape[-1]
        msk_h, msk_w = mask.shape[-2], mask.shape[-1]

        if not (deg_h == gt_h == msk_h and deg_w == gt_w == msk_w):
            raise ValueError(
                f"Spatial dimensions must match before resizing: "
                f"degraded=({deg_h}, {deg_w}), clean_gt=({gt_h}, {gt_w}), mask=({msk_h}, {msk_w})"
            )

        deg_4d, deg_batched = _check_and_prep_image(degraded, "degraded")
        gt_4d, gt_batched = _check_and_prep_image(clean_gt, "clean_gt")
        msk_4d, msk_format, msk_dtype = _check_and_prep_mask(mask, "mask")

        # Bilinear interpolation on images
        # Ensure float tensor for bilinear interpolation
        deg_float = deg_4d.float() if not deg_4d.is_floating_point() else deg_4d
        gt_float = gt_4d.float() if not gt_4d.is_floating_point() else gt_4d

        res_deg = F.interpolate(deg_float, size=self.target_size, mode='bilinear', align_corners=False)
        res_gt = F.interpolate(gt_float, size=self.target_size, mode='bilinear', align_corners=False)

        # Restore original image dtype if it was integer (e.g. uint8)
        if not deg_4d.is_floating_point():
            res_deg = res_deg.round().to(dtype=deg_4d.dtype)
        if not gt_4d.is_floating_point():
            res_gt = res_gt.round().to(dtype=gt_4d.dtype)

        # Nearest-neighbor interpolation on mask
        res_msk_4d = F.interpolate(msk_4d, size=self.target_size, mode='nearest')
        res_msk = _restore_mask_shape(res_msk_4d, msk_format, msk_dtype)

        out_deg = res_deg if deg_batched else res_deg.squeeze(0)
        out_gt = res_gt if gt_batched else res_gt.squeeze(0)

        return out_deg, out_gt, res_msk

    def __repr__(self) -> str:
        return f"SynchronizedResize(size={self.target_size})"


class SynchronizedHorizontalFlip(SynchronizedTransform):
    """
    Deterministically flips degraded, clean_gt, and mask horizontally (left-right).
    Applies identically to all three inputs along the width dimension (dim=-1).
    """
    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        out_deg = torch.flip(degraded, dims=[-1])
        out_gt = torch.flip(clean_gt, dims=[-1])
        out_msk = torch.flip(mask, dims=[-1])
        return out_deg, out_gt, out_msk

    def __repr__(self) -> str:
        return "SynchronizedHorizontalFlip()"


class SynchronizedVerticalFlip(SynchronizedTransform):
    """
    Deterministically flips degraded, clean_gt, and mask vertically (top-bottom).
    Applies identically to all three inputs along the height dimension (dim=-2).
    """
    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        out_deg = torch.flip(degraded, dims=[-2])
        out_gt = torch.flip(clean_gt, dims=[-2])
        out_msk = torch.flip(mask, dims=[-2])
        return out_deg, out_gt, out_msk

    def __repr__(self) -> str:
        return "SynchronizedVerticalFlip()"


class SynchronizedRandomHorizontalFlip(SynchronizedTransform):
    """
    Randomly flips degraded, clean_gt, and mask horizontally with probability p.
    The random decision is evaluated ONCE per call and applied identically to all 3 tensors.

    Args:
        p (float): Probability of applying flip. Default: 0.5.
    """
    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0.0, 1.0], got {p}")
        self.p = p

    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        if random.random() < self.p:
            return (
                torch.flip(degraded, dims=[-1]),
                torch.flip(clean_gt, dims=[-1]),
                torch.flip(mask, dims=[-1])
            )
        return degraded, clean_gt, mask

    def __repr__(self) -> str:
        return f"SynchronizedRandomHorizontalFlip(p={self.p})"


class SynchronizedRandomVerticalFlip(SynchronizedTransform):
    """
    Randomly flips degraded, clean_gt, and mask vertically with probability p.
    The random decision is evaluated ONCE per call and applied identically to all 3 tensors.

    Args:
        p (float): Probability of applying flip. Default: 0.5.
    """
    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0.0, 1.0], got {p}")
        self.p = p

    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        if random.random() < self.p:
            return (
                torch.flip(degraded, dims=[-2]),
                torch.flip(clean_gt, dims=[-2]),
                torch.flip(mask, dims=[-2])
            )
        return degraded, clean_gt, mask

    def __repr__(self) -> str:
        return f"SynchronizedRandomVerticalFlip(p={self.p})"


class SynchronizedCompose(SynchronizedTransform):
    """
    Sequentially applies a series of synchronized spatial transforms.

    Args:
        transforms (Sequence[SynchronizedTransform]): List of transform instances.
    """
    def __init__(self, transforms: Sequence[SynchronizedTransform]):
        for t in transforms:
            if not callable(t):
                raise TypeError(f"All items in transforms must be callable, got {type(t).__name__}")
        self.transforms = list(transforms)

    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        for t in self.transforms:
            degraded, clean_gt, mask = t(degraded, clean_gt, mask)
        return degraded, clean_gt, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class LetterboxMetadata:
    """
    Stores spatial dimensions and padding offsets for unpadding letterboxed tensors.
    """
    def __init__(
        self,
        target_size: Tuple[int, int],
        orig_size: Tuple[int, int],
        content_size: Tuple[int, int],
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        valid_mask: torch.Tensor
    ):
        self.target_size = target_size
        self.orig_size = orig_size
        self.content_size = content_size
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.valid_mask = valid_mask

    def to_dict(self) -> dict:
        return {
            "target_size": self.target_size,
            "orig_size": self.orig_size,
            "content_size": self.content_size,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "valid_mask": self.valid_mask
        }


class SynchronizedLetterbox(SynchronizedTransform):
    """
    Aspect-ratio-preserving resize with symmetric center padding to a fixed target canvas.

    Guarantees:
      - Zero geometric distortion on marine creatures and corals.
      - Degraded and clean images are padded with pad_val_image (default: 0.0).
      - Masks are padded with pad_val_mask (default: 0).
      - Tracks exact padding offsets and provides a boolean valid_mask (True inside content, False on padding).

    Padding Convention:
      - Symmetrically centered:
          pad_top = (target_h - content_h) // 2
          pad_bottom = target_h - content_h - pad_top
          pad_left = (target_w - content_w) // 2
          pad_right = target_w - content_w - pad_left
    """
    def __init__(
        self,
        target_size: Union[int, Sequence[int]],
        pad_val_image: float = 0.0,
        pad_val_mask: int = 0
    ):
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError(f"Target size must be positive, got {target_size}")
            self.target_size: Size2D = (target_size, target_size)
        elif isinstance(target_size, (tuple, list)):
            if len(target_size) != 2:
                raise ValueError(f"Target size tuple/list must have length 2 (H, W), got {target_size}")
            h, w = int(target_size[0]), int(target_size[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Target dimensions must be positive, got {(h, w)}")
            self.target_size = (h, w)
        else:
            raise TypeError(f"Target size must be int or Tuple[int, int], got {type(target_size).__name__}")

        self.pad_val_image = float(pad_val_image)
        self.pad_val_mask = int(pad_val_mask)
        self.last_metadata: Optional[LetterboxMetadata] = None

    def __call__(
        self,
        degraded: torch.Tensor,
        clean_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple3Tensors:
        deg_h, deg_w = degraded.shape[-2], degraded.shape[-1]
        gt_h, gt_w = clean_gt.shape[-2], clean_gt.shape[-1]
        msk_h, msk_w = mask.shape[-2], mask.shape[-1]

        if not (deg_h == gt_h == msk_h and deg_w == gt_w == msk_w):
            raise ValueError(
                f"Spatial dimensions must match before letterboxing: "
                f"degraded=({deg_h}, {deg_w}), clean_gt=({gt_h}, {gt_w}), mask=({msk_h}, {msk_w})"
            )

        orig_h, orig_w = deg_h, deg_w
        target_h, target_w = self.target_size

        # Compute aspect-preserving scale factor
        scale = min(target_h / orig_h, target_w / orig_w)
        content_h = int(round(orig_h * scale))
        content_w = int(round(orig_w * scale))

        # First, resize to content dimensions
        resizer = SynchronizedResize(size=(content_h, content_w))
        deg_res, gt_res, msk_res = resizer(degraded, clean_gt, mask)

        # Compute symmetric center padding offsets
        pad_top = (target_h - content_h) // 2
        pad_bottom = target_h - content_h - pad_top
        pad_left = (target_w - content_w) // 2
        pad_right = target_w - content_w - pad_left

        # Pad images [C, H, W] or [B, C, H, W]
        # F.pad expects (left, right, top, bottom)
        deg_padded = F.pad(deg_res, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=self.pad_val_image)
        gt_padded = F.pad(gt_res, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=self.pad_val_image)
        msk_padded = F.pad(msk_res, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=self.pad_val_mask)

        # Construct boolean valid mask: True inside real content, False in padding
        valid_mask = torch.zeros((target_h, target_w), dtype=torch.bool, device=degraded.device)
        valid_mask[pad_top:pad_top + content_h, pad_left:pad_left + content_w] = True

        self.last_metadata = LetterboxMetadata(
            target_size=(target_h, target_w),
            orig_size=(orig_h, orig_w),
            content_size=(content_h, content_w),
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            valid_mask=valid_mask
        )

        return deg_padded, gt_padded, msk_padded

    def __repr__(self) -> str:
        return f"SynchronizedLetterbox(target_size={self.target_size})"


def unletterbox(
    tensor: torch.Tensor,
    metadata: Union[dict, LetterboxMetadata],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Inverts letterboxing: strips symmetric padding and resizes content back to original dimensions.

    Args:
        tensor: [C, H, W] or [B, C, H, W] or [H, W] letterboxed tensor.
        metadata: LetterboxMetadata or dictionary containing padding offsets and orig_size.
        mode: Interpolation mode ('bilinear' for images/logits, 'nearest' for discrete masks).

    Returns:
        torch.Tensor unpadded and resized back to original resolution.
    """
    if isinstance(metadata, dict):
        pad_top = metadata["pad_top"]
        pad_left = metadata["pad_left"]
        content_h, content_w = metadata["content_size"]
        orig_h, orig_w = metadata["orig_size"]
    else:
        pad_top = metadata.pad_top
        pad_left = metadata.pad_left
        content_h, content_w = metadata.content_size
        orig_h, orig_w = metadata.orig_size

    # Slice out content
    cropped = tensor[..., pad_top:pad_top + content_h, pad_left:pad_left + content_w]

    orig_dtype = tensor.dtype
    is_float = cropped.is_floating_point()
    interp_mode = 'nearest' if not is_float or mode == 'nearest' else 'bilinear'

    # Reshape for F.interpolate
    is_2d = (cropped.dim() == 2)
    is_3d = (cropped.dim() == 3)

    if is_2d:
        t_4d = cropped.unsqueeze(0).unsqueeze(0).float()
    elif is_3d:
        t_4d = cropped.unsqueeze(0).float() if not is_float else cropped.unsqueeze(0)
    else:
        t_4d = cropped.float() if not is_float else cropped

    if interp_mode == 'bilinear':
        restored_4d = F.interpolate(t_4d, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    else:
        restored_4d = F.interpolate(t_4d, size=(orig_h, orig_w), mode='nearest')

    if not is_float:
        restored_4d = restored_4d.round().to(dtype=orig_dtype)

    if is_2d:
        return restored_4d.squeeze(0).squeeze(0)
    elif is_3d:
        return restored_4d.squeeze(0)
    else:
        return restored_4d


def build_preprocessing_transform(
    resolution: Union[int, Sequence[int]],
    mode: str = "direct"
) -> SynchronizedTransform:
    """
    Factory function for configuration-driven preprocessing.

    Supported configurations:
      - mode='direct': SynchronizedResize(size=resolution)
      - mode='letterbox': SynchronizedLetterbox(target_size=resolution)
    """
    m = mode.lower().strip()
    if m == "direct":
        return SynchronizedResize(size=resolution)
    elif m == "letterbox":
        return SynchronizedLetterbox(target_size=resolution)
    else:
        raise ValueError(f"Unsupported preprocessing mode '{mode}'. Choose 'direct' or 'letterbox'.")

