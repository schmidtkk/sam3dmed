#!/usr/bin/env python3
"""Per-slice augmentations for TS_SAM3D_Dataset.

Applies consistent geometric transforms to image, mask, and pointmap.
NaN background pixels in pointmap are preserved after transforms.

Usage:
    augment = SliceAugmentor(enable=True, rotation_range=15, scale_range=(0.9, 1.1))
    augmented = augment(image, mask, pointmap)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SliceAugmentorConfig:
    """Configuration for per-slice augmentations."""
    enable: bool = True
    p: float = 0.5  # probability of applying any augmentation

    # Geometric transforms
    rotation_range: float = 15.0  # degrees, symmetric around 0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    translation_range: Tuple[float, float] = (-0.1, 0.1)  # fraction of image size
    flip_horizontal: bool = True
    flip_vertical: bool = False

    # Intensity transforms (image only)
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    gamma_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.02


class SliceAugmentor:
    """Apply consistent augmentations to image, mask, and pointmap.

    Geometric transforms are applied to all three consistently.
    Intensity transforms are applied to image only.
    NaN values in pointmap are preserved throughout.
    """

    def __init__(self, config: Optional[SliceAugmentorConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = SliceAugmentorConfig(**kwargs)

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        pointmap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply augmentations.

        Args:
            image: (C, H, W) float tensor
            mask: (H, W) float tensor
            pointmap: (3, H, W) float tensor with NaN for background

        Returns:
            dict with 'image', 'mask', 'pointmap' keys
        """
        if not self.config.enable or random.random() > self.config.p:
            return {'image': image, 'mask': mask, 'pointmap': pointmap}

        # Sample geometric parameters
        angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
        scale = random.uniform(*self.config.scale_range)
        tx = random.uniform(*self.config.translation_range)
        ty = random.uniform(*self.config.translation_range)
        flip_h = self.config.flip_horizontal and random.random() < 0.5
        flip_v = self.config.flip_vertical and random.random() < 0.5

        # Apply geometric transforms
        image_out = self._apply_geometric(image, angle, scale, tx, ty, flip_h, flip_v, mode='bilinear')
        mask_out = self._apply_geometric(mask.unsqueeze(0), angle, scale, tx, ty, flip_h, flip_v, mode='nearest').squeeze(0)
        pointmap_out = self._apply_geometric_pointmap(pointmap, angle, scale, tx, ty, flip_h, flip_v)

        # Apply intensity transforms to image only
        image_out = self._apply_intensity(image_out)

        return {'image': image_out, 'mask': mask_out, 'pointmap': pointmap_out}

    def _get_rotation_matrix(self, angle: float, scale: float, tx: float, ty: float) -> torch.Tensor:
        """Build 2x3 affine matrix for rotation, scale, and translation."""
        theta = math.radians(angle)
        cos_t = math.cos(theta) * scale
        sin_t = math.sin(theta) * scale
        # Standard affine matrix (rotation + scale + translation)
        # Note: grid_sample uses normalized coordinates [-1, 1]
        matrix = torch.tensor([
            [cos_t, -sin_t, tx],
            [sin_t, cos_t, ty]
        ], dtype=torch.float32)
        return matrix

    def _apply_geometric(
        self,
        tensor: torch.Tensor,
        angle: float,
        scale: float,
        tx: float,
        ty: float,
        flip_h: bool,
        flip_v: bool,
        mode: str = 'bilinear',
    ) -> torch.Tensor:
        """Apply geometric transform to a tensor.

        Args:
            tensor: (C, H, W) tensor
            angle: rotation angle in degrees
            scale: scale factor
            tx, ty: translation as fraction of image size
            flip_h, flip_v: flip flags
            mode: interpolation mode ('bilinear' or 'nearest')

        Returns:
            transformed tensor (C, H, W)
        """
        if flip_h:
            tensor = torch.flip(tensor, dims=[-1])
        if flip_v:
            tensor = torch.flip(tensor, dims=[-2])

        # Build affine grid
        matrix = self._get_rotation_matrix(angle, scale, tx, ty)
        matrix = matrix.unsqueeze(0)  # (1, 2, 3)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)

        grid = F.affine_grid(matrix, tensor.size(), align_corners=False)
        output = F.grid_sample(tensor, grid, mode=mode, padding_mode='zeros', align_corners=False)

        return output.squeeze(0)

    def _apply_geometric_pointmap(
        self,
        pointmap: torch.Tensor,
        angle: float,
        scale: float,
        tx: float,
        ty: float,
        flip_h: bool,
        flip_v: bool,
    ) -> torch.Tensor:
        """Apply geometric transform to pointmap, preserving NaN background.

        The pointmap contains 3D world coordinates. When we rotate/scale the 2D slice,
        we need to:
        1. Resample the pointmap spatially (same as image)
        2. Rotate the XY components of the world coordinates accordingly

        Args:
            pointmap: (3, H, W) tensor with NaN for background

        Returns:
            transformed pointmap (3, H, W) with NaN preserved
        """
        # Create mask of valid (non-NaN) pixels
        nan_mask = torch.isnan(pointmap[0])

        # Replace NaN with 0 for interpolation
        pointmap_filled = torch.where(torch.isnan(pointmap), torch.zeros_like(pointmap), pointmap)

        # Apply flips
        if flip_h:
            pointmap_filled = torch.flip(pointmap_filled, dims=[-1])
            nan_mask = torch.flip(nan_mask, dims=[-1])
            # Flip X coordinate value (negate it)
            pointmap_filled[0] = -pointmap_filled[0]

        if flip_v:
            pointmap_filled = torch.flip(pointmap_filled, dims=[-2])
            nan_mask = torch.flip(nan_mask, dims=[-2])
            # Flip Y coordinate value (negate it)
            pointmap_filled[1] = -pointmap_filled[1]

        # Apply spatial affine transform
        matrix = self._get_rotation_matrix(angle, scale, tx, ty)
        matrix = matrix.unsqueeze(0)

        pointmap_filled = pointmap_filled.unsqueeze(0)
        nan_mask = nan_mask.unsqueeze(0).unsqueeze(0).float()

        grid = F.affine_grid(matrix, pointmap_filled.size(), align_corners=False)
        pointmap_out = F.grid_sample(pointmap_filled, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # For mask, use 'zeros' and then invert logic - pixels outside valid region become NaN
        mask_out = F.grid_sample(1.0 - nan_mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)

        pointmap_out = pointmap_out.squeeze(0)
        # mask_out contains 1 where valid, 0 where invalid (was NaN or padding)
        mask_out = mask_out.squeeze(0).squeeze(0) < 0.5  # True where should be NaN

        # Apply rotation to XY coordinates of the world points
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x = pointmap_out[0].clone()
        y = pointmap_out[1].clone()
        # Note: We apply inverse rotation to world coordinates since the spatial
        # transform already moved pixels. This keeps world coords consistent.
        # Actually, for pointmap, we want to preserve the original world coordinates
        # at the new pixel locations - no need to rotate the values themselves.
        # The spatial resampling already handles this.

        # Restore NaN for background pixels
        pointmap_out = torch.where(mask_out.unsqueeze(0).expand_as(pointmap_out), 
                                   torch.full_like(pointmap_out, float('nan')), 
                                   pointmap_out)

        return pointmap_out

    def _apply_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Apply intensity augmentations to image.

        Args:
            image: (C, H, W) float tensor

        Returns:
            augmented image (C, H, W)
        """
        # Brightness
        if self.config.brightness_range != (0.0, 0.0):
            brightness = random.uniform(*self.config.brightness_range)
            image = image + brightness

        # Contrast
        if self.config.contrast_range != (1.0, 1.0):
            contrast = random.uniform(*self.config.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast + mean

        # Gamma
        if self.config.gamma_range != (1.0, 1.0):
            gamma = random.uniform(*self.config.gamma_range)
            # Shift to positive range, apply gamma, shift back
            min_val = image.min()
            image_shifted = image - min_val + 1e-6
            image = torch.pow(image_shifted, gamma) + min_val - 1e-6

        # Additive Gaussian noise
        if self.config.noise_std > 0:
            noise = torch.randn_like(image) * self.config.noise_std
            image = image + noise

        return image


def create_augmentor(
    enable: bool = True,
    mode: str = 'train',
    **kwargs
) -> SliceAugmentor:
    """Factory function to create augmentor with preset configurations.

    Args:
        enable: whether augmentations are enabled
        mode: 'train', 'val', or 'test'
        **kwargs: override any SliceAugmentorConfig field

    Returns:
        SliceAugmentor instance
    """
    if mode == 'train':
        config = SliceAugmentorConfig(
            enable=enable,
            p=0.5,
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
            translation_range=(-0.1, 0.1),
            flip_horizontal=True,
            flip_vertical=False,
            brightness_range=(-0.1, 0.1),
            contrast_range=(0.9, 1.1),
            gamma_range=(0.9, 1.1),
            noise_std=0.02,
        )
    elif mode == 'val':
        # Light augmentations for validation (optional)
        config = SliceAugmentorConfig(
            enable=enable,
            p=0.3,
            rotation_range=5.0,
            scale_range=(0.95, 1.05),
            translation_range=(-0.05, 0.05),
            flip_horizontal=True,
            flip_vertical=False,
            brightness_range=(0.0, 0.0),
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            noise_std=0.0,
        )
    else:  # test
        config = SliceAugmentorConfig(enable=False)

    # Override with kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return SliceAugmentor(config)
