"""
Metrics utilities used for medical evaluation: Dice, Chamfer, HD95.
This file uses optional external libs when available and provides NumPy/Scipy fallbacks.
"""
from __future__ import annotations
import warnings
import numpy as np
from typing import Tuple

try:
    # optional: PyTorch3D chamfer distance
    from pytorch3d.loss import chamfer_distance as _chamfer_distance_p3d
    _HAS_PYTORCH3D = True
except Exception:
    _HAS_PYTORCH3D = False

try:
    from surface_distance import compute_surface_distances, compute_robust_hausdorff
    _HAS_SURFACE_DISTANCE = True
except Exception:
    _HAS_SURFACE_DISTANCE = False

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Compute voxel-level Dice score for binary volumes.

    Args:
        pred: binary ndarray (Bools or ints 0/1) or float mask
        gt: binary ndarray
    Returns:
        Dice score float in [0,1]
    """
    pred_b = (pred > 0).astype(np.float32)
    gt_b = (gt > 0).astype(np.float32)
    intersection = np.sum(pred_b * gt_b)
    denom = np.sum(pred_b) + np.sum(gt_b)
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / (denom + eps))


def _compute_chamfer_numpy(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute Chamfer distance between two point sets using SciPy's cKDTree fallback.
    Returns per-point mean squared distances sum.
    """
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float('inf')
    if not _HAS_SCIPY:
        raise RuntimeError("scipy must be installed to compute chamfer fallback")
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_ab, _ = tree_b.query(points_a, k=1)
    d_ba, _ = tree_a.query(points_b, k=1)
    # Mean squared distances
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def compute_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute Chamfer distance between two point sets. Use PyTorch3D when available.

    Args:
        points_a: (N, 3) float
        points_b: (M, 3) float
    Returns:
        Scalar float Chamfer distance (sum of mean squared dists in each direction)
    """
    points_a = np.asarray(points_a, dtype=np.float32)
    points_b = np.asarray(points_b, dtype=np.float32)
    if _HAS_PYTORCH3D:
        import torch
        pta = torch.from_numpy(points_a[None]).to(torch.float32)
        ptb = torch.from_numpy(points_b[None]).to(torch.float32)
        with torch.no_grad():
            loss, _ = _chamfer_distance_p3d(pta, ptb)
        return float(loss.item())
    else:
        return _compute_chamfer_numpy(points_a, points_b)


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Compute robust Hausdorff distance HD95 between binary masks.

    Args:
        pred_mask: binary ndarray
        gt_mask: binary ndarray
        spacing: voxel spacing (mm) along (z,y,x)
    Returns:
        HD95 metric float (95th percentile)
    """
    pred_mask = np.asarray(pred_mask).astype(np.bool_)
    gt_mask = np.asarray(gt_mask).astype(np.bool_)
    if _HAS_SURFACE_DISTANCE:
        distances = compute_surface_distances(gt_mask, pred_mask, spacing)
        return float(compute_robust_hausdorff(distances, 95))
    # Fallback: compute boundary points and compute 95th percentile directed Hausdorff approx via KD-tree
    if not _HAS_SCIPY:
        raise RuntimeError("scipy (scipy.spatial) is required for fallback HD95 implementation")
    from scipy.ndimage import binary_erosion
    # compute surfaces using binary erosion
    gt_surface = gt_mask & (~binary_erosion(gt_mask))
    pred_surface = pred_mask & (~binary_erosion(pred_mask))
    gt_pts = np.argwhere(gt_surface) * np.array(spacing)
    pred_pts = np.argwhere(pred_surface) * np.array(spacing)
    if gt_pts.size == 0 or pred_pts.size == 0:
        return float('inf')
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    # 95th percentile
    hd95 = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
    return float(hd95)
