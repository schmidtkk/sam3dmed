"""Shared pytest fixtures for SAM3D-Objects tests."""

import os

import numpy as np
import pytest

# Skip LIDRA initialization during tests
os.environ["LIDRA_SKIP_INIT"] = "1"


@pytest.fixture
def sample_volume():
    """Create a sample 3D volume for testing."""
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    # Create a sphere in the center
    center = np.array([32, 32, 32])
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if np.linalg.norm(np.array([x, y, z]) - center) < 15:
                    volume[x, y, z] = 1.0
    return volume


@pytest.fixture
def sample_mask():
    """Create a sample binary mask for testing."""
    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    mask[20:45, 20:45, 20:45] = 1
    return mask


@pytest.fixture
def sample_points():
    """Create sample 3D point cloud for testing."""
    rng = np.random.default_rng(42)
    points = rng.random((1000, 3)).astype(np.float32)
    return points


@pytest.fixture
def sample_image():
    """Create a sample 2D image for testing."""
    return np.random.rand(256, 256).astype(np.float32)


@pytest.fixture
def sample_spacing():
    """Standard isotropic spacing for medical images."""
    return (1.0, 1.0, 1.0)


@pytest.fixture
def tmp_nifti(tmp_path):
    """Create a temporary NIfTI file for testing."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    volume = np.random.rand(64, 64, 64).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(volume, affine)
    filepath = tmp_path / "test_volume.nii.gz"
    nib.save(img, filepath)
    return filepath


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
    except ImportError:
        pass
    return "cpu"
