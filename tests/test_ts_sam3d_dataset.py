#!/usr/bin/env python3
"""Tests for TS_SAM3D_Dataset loader.

Validates:
1. Dataset initialization with various configurations
2. Augmentation integration
3. Occupancy threshold filtering
4. Output dictionary structure
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

# Ensure the module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_mock_slice_cache(cache_dir: Path, num_slices: int = 5, add_empty: bool = True):
    """Create mock .npz files for testing."""
    for i in range(num_slices):
        H, W = 64, 64
        image = np.random.randn(3, H, W).astype(np.float32)

        # Create mask with varying occupancy
        mask = np.zeros((H, W), dtype=np.uint8)
        if i < num_slices - 1 or not add_empty:  # Last one is empty if add_empty
            size = min(10 + i * 5, 60)
            mask[10 : 10 + size, 10 : 10 + size] = 1

        pointmap = np.random.randn(3, H, W).astype(np.float32)
        pointmap[:, mask == 0] = np.nan

        affine = np.eye(4, dtype=np.float32)

        out_path = cache_dir / f"test_case_axis0_slice{i:04d}.npz"
        np.savez_compressed(
            out_path,
            image=image,
            mask=mask,
            pointmap=pointmap,
            affine=affine,
            slice_idx=i,
            axis=0,
        )


def test_dataset_output_structure():
    """Test that dataset returns correct dictionary structure."""
    # Import here to avoid early loading issues
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "slice_cache"
        cache_dir.mkdir()
        create_mock_slice_cache(cache_dir, num_slices=3, add_empty=False)

        # Mock the PreProcessor to avoid complex dependencies
        mock_preprocessor = MagicMock()
        mock_preprocessor._process_image_mask_pointmap_mess.return_value = {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "pointmap": torch.randn(3, 64, 64),
            "pointmap_scale": torch.tensor([1.0]),
            "pointmap_shift": torch.zeros(3),
        }

        # Create dataset - _gather_cases will return empty but we have cached files
        ds = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            augment=False,
            preprocessor=mock_preprocessor,
        )

        assert len(ds) == 3, f"Expected 3 slices, got {len(ds)}"

        item = ds[0]
        assert "image" in item
        assert "mask" in item
        assert "pointmap" in item
        assert "affine" in item
        assert "name" in item


def test_occupancy_threshold_filtering():
    """Test that slices below occupancy threshold are filtered out."""
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "slice_cache"
        cache_dir.mkdir()
        create_mock_slice_cache(cache_dir, num_slices=5, add_empty=True)

        mock_preprocessor = MagicMock()
        mock_preprocessor._process_image_mask_pointmap_mess.return_value = {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "pointmap": torch.randn(3, 64, 64),
        }

        # Low threshold - should keep all non-empty
        ds_low = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            occupancy_threshold=0.01,
            augment=False,
            preprocessor=mock_preprocessor,
        )

        # High threshold - should filter more
        ds_high = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            occupancy_threshold=0.3,
            augment=False,
            preprocessor=mock_preprocessor,
        )

        # Low threshold should have more slices than high threshold
        assert len(ds_low) >= len(ds_high), (
            f"Low threshold ({len(ds_low)}) should have >= slices than high ({len(ds_high)})"
        )


def test_augmentation_integration():
    """Test that augmentor is properly integrated."""
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "slice_cache"
        cache_dir.mkdir()
        create_mock_slice_cache(cache_dir, num_slices=2, add_empty=False)

        mock_preprocessor = MagicMock()
        mock_preprocessor._process_image_mask_pointmap_mess.return_value = {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "pointmap": torch.randn(3, 64, 64),
        }

        # Train mode - augmentations enabled
        ds_train = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            augment=True,
            augment_mode="train",
            preprocessor=mock_preprocessor,
        )
        assert ds_train.augmentor.config.enable is True
        assert ds_train.augmentor.config.p > 0

        # Test mode - augmentations disabled
        ds_test = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            augment=True,
            augment_mode="test",
            preprocessor=mock_preprocessor,
        )
        assert ds_test.augmentor.config.enable is False


def test_include_pose_flag():
    """Test that include_pose adds pose target fields."""
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import TS_SAM3D_Dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "slice_cache"
        cache_dir.mkdir()
        create_mock_slice_cache(cache_dir, num_slices=1, add_empty=False)

        mock_preprocessor = MagicMock()
        mock_preprocessor._process_image_mask_pointmap_mess.return_value = {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "pointmap": torch.randn(3, 64, 64),
        }

        # Without pose
        ds_no_pose = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            augment=False,
            include_pose=False,
            preprocessor=mock_preprocessor,
        )
        item_no_pose = ds_no_pose[0]
        assert "pose_target" not in item_no_pose

        # With pose
        ds_with_pose = TS_SAM3D_Dataset(
            original_nifti_dir=tmpdir,
            cache_slices=True,
            slice_cache_dir=str(cache_dir),
            augment=False,
            include_pose=True,
            preprocessor=mock_preprocessor,
        )
        item_with_pose = ds_with_pose[0]
        assert "pose_target" in item_with_pose
        assert "x_instance_scale" in item_with_pose["pose_target"]
        assert "x_instance_rotation" in item_with_pose["pose_target"]


def test_collate_function():
    """Test that collate function properly batches samples."""
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import data_collate

    batch = [
        {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "mask_sdf": torch.randn(1, 32, 32, 32),
            "affine": torch.eye(4),
            "name": "case1",
        },
        {
            "image": torch.randn(3, 64, 64),
            "mask": torch.randint(0, 2, (64, 64)).float(),
            "mask_sdf": torch.randn(1, 32, 32, 32),
            "affine": torch.eye(4),
            "name": "case2",
        },
    ]

    collated = data_collate(batch)

    assert collated["image"].shape[0] == 2, "Batch size should be 2"
    assert collated["image"].shape[1] == 3, "Should have 3 channels"
    assert len(collated["name"]) == 2, "Should have 2 names"
    assert collated["affine"].shape == (2, 4, 4), "Affine should be batched"


def test_augmentor_create():
    """Test augmentor creation function."""
    from sam3d_objects.data.dataset.ts.slice_augmentations import create_augmentor

    train_aug = create_augmentor(enable=True, mode="train")
    assert train_aug.config.enable is True
    assert train_aug.config.p == 0.5

    test_aug = create_augmentor(enable=True, mode="test")
    assert test_aug.config.enable is False


def test_slice_filtering_logic():
    """Test the occupancy filtering logic in isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Create test npz files with different occupancy
        for i, occupancy in enumerate([0.0, 0.05, 0.2, 0.5]):
            H, W = 64, 64
            mask = np.zeros((H, W), dtype=np.uint8)
            if occupancy > 0:
                size = int(np.sqrt(occupancy * H * W))
                mask[:size, :size] = 1

            np.savez_compressed(
                cache_dir / f"test_slice_{i}.npz",
                image=np.zeros((3, H, W), dtype=np.float32),
                mask=mask,
                pointmap=np.zeros((3, H, W), dtype=np.float32),
                affine=np.eye(4, dtype=np.float32),
            )

        # Test filtering logic directly
        def filter_by_occupancy(file_paths, threshold):
            filtered = []
            for fp in file_paths:
                data = np.load(str(fp), allow_pickle=True)
                mask = data["mask"]
                fg_ratio = (mask > 0).sum() / mask.size
                if fg_ratio >= threshold:
                    filtered.append(fp)
            return filtered

        all_files = sorted(cache_dir.glob("*.npz"))

        # Low threshold
        filtered_low = filter_by_occupancy(all_files, 0.01)
        assert len(filtered_low) == 3, (
            f"Expected 3 slices with threshold 0.01, got {len(filtered_low)}"
        )

        # High threshold
        filtered_high = filter_by_occupancy(all_files, 0.3)
        assert len(filtered_high) == 1, (
            f"Expected 1 slice with threshold 0.3, got {len(filtered_high)}"
        )


if __name__ == "__main__":
    test_dataset_output_structure()
    print("✓ test_dataset_output_structure passed")

    test_occupancy_threshold_filtering()
    print("✓ test_occupancy_threshold_filtering passed")

    test_augmentation_integration()
    print("✓ test_augmentation_integration passed")

    test_include_pose_flag()
    print("✓ test_include_pose_flag passed")

    test_collate_function()
    print("✓ test_collate_function passed")

    test_augmentor_create()
    print("✓ test_augmentor_create passed")

    test_slice_filtering_logic()
    print("✓ test_slice_filtering_logic passed")

    print("\nAll dataset loader tests passed!")
