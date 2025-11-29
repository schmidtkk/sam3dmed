#!/usr/bin/env python3
"""Signed Distance Field (SDF) computation utilities.

Provides functions to compute signed distance fields from binary masks.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field from a binary mask.
    
    The SDF is negative inside the object and positive outside.
    
    Args:
        mask: Binary mask array (uint8 or bool), any dimensionality
              Non-zero values are considered foreground.
    
    Returns:
        SDF array (float32) with same shape as input.
        Values are negative inside the object, positive outside,
        and zero at the boundary.
    """
    mask = mask.astype(bool)
    
    if mask.sum() == 0:
        # All background - return large positive distance
        return np.ones(mask.shape, dtype=np.float32) * 1e6
    
    if (~mask).sum() == 0:
        # All foreground - return negative distance from edge
        return -ndimage.distance_transform_edt(mask).astype(np.float32)
    
    # Distance transform for inside (foreground)
    dist_inside = ndimage.distance_transform_edt(mask)
    
    # Distance transform for outside (background) 
    dist_outside = ndimage.distance_transform_edt(~mask)
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    return sdf.astype(np.float32)


def compute_sdf_normalized(mask: np.ndarray, truncate: float = 5.0) -> np.ndarray:
    """Compute normalized and truncated SDF.
    
    Args:
        mask: Binary mask array
        truncate: Truncation distance (values beyond this are clipped)
    
    Returns:
        Truncated SDF normalized to [-1, 1] range.
    """
    sdf = compute_sdf(mask)
    sdf = np.clip(sdf, -truncate, truncate) / truncate
    return sdf.astype(np.float32)
