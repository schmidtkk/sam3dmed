import numpy as np
from sam3d_objects.utils.metrics import compute_dice, compute_chamfer, compute_hd95


def test_dice_identical():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    a[1:3, 1:3, 1:3] = 1
    assert compute_dice(a, a) == 1.0


def test_dice_disjoint():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    b = np.zeros((4, 4, 4), dtype=np.uint8)
    a[0:1, 0:1, 0:1] = 1
    b[3:4, 3:4, 3:4] = 1
    assert compute_dice(a, b) == 0.0


def test_chamfer_identical_pointclouds():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    c = compute_chamfer(pts, pts)
    assert abs(c) < 1e-6


def test_hd95_identical():
    a = np.zeros((8, 8, 8), dtype=np.uint8)
    a[2:6, 2:6, 2:6] = 1
    assert compute_hd95(a, a) == 0.0
