#!/usr/bin/env python3
"""TS_SAM3D_Dataset: per-slice dataset for SAM3D object medical fine-tuning.

Features implemented:
- One sample per 2D slice (random axis sampling by default)
- Option to cache per-slice `.npz` files for fast IO
- Compute pointmap from raw NIfTI `affine` and per-slice index
- Use PreProcessor to apply transforms and SSIPointmapNormalizer
- Return per-sample dict compatible with `InferencePipeline` and `PreProcessor`
"""

from __future__ import annotations

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sam3d_objects.data.dataset.tdfy.preprocessor import PreProcessor
from sam3d_objects.data.dataset.ts.slice_augmentations import create_augmentor
from utils.sdf_fn import compute_sdf


class TS_SAM3D_Dataset(Dataset):
    """Dataset yielding one sample per 2D slice.

    Config options (kwargs):
    - original_nifti_dir: str -> root dir of raw NIfTI
    - cache_slices: bool -> whether to use per-slice cached npz files
    - slice_cache_dir: str -> where to write/read cached .npz files
    - slice_sampling_method: str -> 'random_axis' | 'uniform' | 'central' | 'all'
    - num_slices_per_volume: int -> sample this many slices per volume for training

    Returned dictionary:
    - 'image' : torch.Tensor (C,H,W)
    - 'mask': torch.Tensor (H,W)
    - 'pointmap': torch.Tensor (3,H,W)
    - 'pointmap_scale', 'pointmap_shift' (if normalization done by PreProcessor)
    - 'affine': torch.Tensor (4,4)
    - 'gt_sdf': torch.Tensor (C_gt, D, H, W) or path pointer via 'gt_sdf_path' depending on config
    - 'name' : case id or sample id

    """

    def __init__(
        self,
        original_nifti_dir: str,
        cache_slices: bool = True,
        slice_cache_dir: str | None = None,
        slice_sampling_method: str = "random_axis",
        num_slices_per_volume: int = 3,
        preprocess_crop_size: tuple[int, int] = (256, 256),
        use_resampled_dir: str | None = None,
        classes: int = 1,
        augment: bool = True,
        augment_mode: str = "train",
        occupancy_threshold: float = 0.01,
        include_pose: bool = False,
        **kwargs,
    ):
        """Initialize TS_SAM3D_Dataset.

        Args:
            original_nifti_dir: Root directory containing raw NIfTI files
            cache_slices: Whether to cache per-slice .npz files for fast IO
            slice_cache_dir: Directory for cached slices (default: original_nifti_dir/slice_cache)
            slice_sampling_method: How to sample slices - 'random_axis', 'uniform', 'central', 'all'
            num_slices_per_volume: Number of slices to sample per volume for training
            preprocess_crop_size: Target crop size (H, W) after preprocessing
            use_resampled_dir: Optional path to pre-resampled data
            classes: Number of foreground classes
            augment: Whether to apply per-slice augmentations
            augment_mode: 'train', 'val', or 'test' - controls augmentation intensity
            occupancy_threshold: Minimum foreground ratio to include a slice
            include_pose: Whether to include pose target fields (for pose-aware models)
            **kwargs: Additional arguments (e.g., 'preprocessor' to inject custom PreProcessor)
        """
        super().__init__()
        self.original_nifti_dir = Path(original_nifti_dir)
        self.cache_slices = bool(cache_slices)
        self.slice_cache_dir = Path(slice_cache_dir or (self.original_nifti_dir / "slice_cache"))
        self.slice_cache_dir.mkdir(parents=True, exist_ok=True)
        self.slice_sampling_method = slice_sampling_method
        self.num_slices_per_volume = int(num_slices_per_volume)
        self.preprocess_crop_size = preprocess_crop_size
        self.use_resampled_dir = Path(use_resampled_dir) if use_resampled_dir is not None else None
        self.classes = int(classes)
        self.occupancy_threshold = float(occupancy_threshold)
        self.include_pose = bool(include_pose)

        # PreProcessor: default settings; the user can inject custom transforms via kwargs
        self.preprocessor = kwargs.get(
            "preprocessor",
            PreProcessor(preprocess_crop_size=self.preprocess_crop_size),
        )

        # Per-slice augmentor
        self.augmentor = create_augmentor(enable=augment, mode=augment_mode)

        # Build case list from original_nifti_dir (assume patterns *_img.nii.gz and *_mask.nii.gz)
        self.cases = self._gather_cases()

        # If caching, ensure the cache is built or build on-the-fly
        if self.cache_slices:
            self._build_or_find_slice_cache()
            all_files = sorted(list(self.slice_cache_dir.glob("*.npz")))
            # Filter by occupancy threshold if needed
            if self.occupancy_threshold > 0:
                self.file_names = self._filter_by_occupancy(all_files)
            else:
                self.file_names = all_files
        else:
            self.file_names = []  # no file names, we will sample on-the-fly

    def _filter_by_occupancy(self, file_paths: list[Path]) -> list[Path]:
        """Filter cached slice files by minimum foreground occupancy.
        
        Uses a cache file to avoid re-scanning all npz files on every run.
        """
        # Check for cached filter results
        cache_file = self.slice_cache_dir / f"_occupancy_filter_{self.occupancy_threshold:.4f}.txt"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_files = [self.slice_cache_dir / line.strip() for line in f if line.strip()]
                # Verify files still exist
                existing = [fp for fp in cached_files if fp.exists()]
                if len(existing) == len(cached_files):
                    print(f"[Dataset] Loaded {len(existing)} filtered slices from cache")
                    return existing
            except Exception:
                pass
        
        # Filter and cache results
        print(f"[Dataset] Filtering {len(file_paths)} slices by occupancy >= {self.occupancy_threshold}...")
        filtered = []
        for i, fp in enumerate(file_paths):
            if (i + 1) % 500 == 0:
                print(f"[Dataset] Filtering progress: {i+1}/{len(file_paths)}")
            try:
                data = np.load(str(fp), allow_pickle=True)
                mask = data["mask"]
                fg_ratio = (mask > 0).sum() / mask.size
                if fg_ratio >= self.occupancy_threshold:
                    filtered.append(fp)
            except Exception:
                continue
        
        # Save cache
        try:
            with open(cache_file, "w") as f:
                for fp in filtered:
                    f.write(fp.name + "\n")
            print(f"[Dataset] Cached {len(filtered)} filtered slices")
        except Exception as e:
            print(f"[Dataset] Warning: could not cache filter results: {e}")
        
        return filtered

    def _gather_cases(self):
        # Search for pairs of NIfTI files; accept many naming conventions like *img.nii.gz and *mask.nii.gz
        img_candidates = list(self.original_nifti_dir.glob("**/*img*.nii*"))
        if len(img_candidates) == 0:
            # fallback to all .nii* files paired with *_seg or *_mask
            img_candidates = [
                p
                for p in self.original_nifti_dir.glob("**/*.nii*")
                if not p.name.lower().endswith("_seg.nii.gz")
            ]
        cases = []
        for img_path in img_candidates:
            mask_path = None
            # heuristic for mask
            for candidate in (
                img_path.parent.glob(img_path.stem + "*seg*.nii*"),
                img_path.parent.glob(img_path.stem + "*mask*.nii*"),
            ):
                for cm in candidate:
                    mask_path = cm
                    break
                if mask_path is not None:
                    break
            if mask_path is None:
                # try sibling mask in same directory
                for p in img_path.parent.glob("*seg*.nii*"):
                    mask_path = p
                    break
            if mask_path is None:
                continue
            cases.append((img_path, mask_path))
        return cases

    def _build_or_find_slice_cache(self):
        # If slice_cache_dir contains files already, assume cache built.
        existing = list(self.slice_cache_dir.glob("*.npz"))
        if len(existing) > 0:
            return
        # Otherwise, build on the fly: extract per-case per-slice caches
        for img_path, mask_path in self.cases:
            case_id = img_path.stem
            try:
                nii_img = nib.load(str(img_path))
                nii_mask = nib.load(str(mask_path))
            except Exception as e:
                print(f"Error loading case {img_path}, skipping: {e}")
                continue
            img = nii_img.get_fdata()
            mask = nii_mask.get_fdata().astype(np.uint8)
            affine = nii_img.affine

            # Optionally resample to isotropic spacing - if necessary, use torchio
            # We leave resampling steps to the reprocessing script in scripts/reprocess_ts_nifti.py

            # Precompute per-class sdf for the volume
            sdf_channels = []
            for c in range(1, self.classes + 1):
                sdf_c = compute_sdf((mask == c).astype(np.uint8))
                sdf_channels.append(sdf_c.astype(np.float32))
            mask_sdf = np.stack(sdf_channels, axis=0) if len(sdf_channels) > 0 else None
            if mask_sdf is not None:
                # Save SDF per case
                np.save(self.slice_cache_dir / f"{case_id}_sdf.npy", mask_sdf)

            # Extract all slices and save as cache; store metadata about slice sampling
            # We cache all slices by default here; training sampling will pick random ones.
            H, W = img.shape[1], img.shape[2]
            for z in range(img.shape[0]):
                # For simplicity store axial slices in cache; random axis sampling still possible in on-the-fly
                image_2d = img[z, :, :]
                mask_2d = mask[z, :, :]

                # Build pointmap for axial
                xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
                k = np.full_like(xv, z)
                voxel_coords = np.stack([xv, yv, k, np.ones_like(xv)], axis=-1)
                world_coords = (affine @ voxel_coords.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]
                # Replace masked positions (background) with zeros rather than NaN so
                # that downstream embedding layers (e.g., Linear) don't produce NaN outputs.
                xyz_masked = (
                    np.where(mask_2d[..., None], world_coords, 0.0)
                    .astype(np.float32)
                    .transpose(2, 0, 1)
                )

                # convert image to 3 channels (replicate) and float32
                image_3ch = np.stack([image_2d, image_2d, image_2d], axis=0).astype(np.float32)

                fname = self.slice_cache_dir / f"{case_id}_slice_{z:03d}.npz"
                np.savez_compressed(
                    fname,
                    image=image_3ch,
                    mask=mask_2d.astype(np.uint8),
                    pointmap=xyz_masked,
                    affine=affine,
                    slice_idx=z,
                    gt_sdf_path=str(self.slice_cache_dir / f"{case_id}_sdf.npy"),
                )

    def __len__(self):
        if self.cache_slices:
            return len(self.file_names)
        # if no cache, estimate based on volumes and slices per volume
        total_slices = sum(
            [
                img.shape[0]
                for img, _ in [(nib.load(str(i[0])).get_fdata(), None) for i in self.cases]
            ]
        )
        return total_slices

    def _sample_slice_on_the_fly(self, case_idx: int):
        img_path, mask_path = self.cases[case_idx]
        nii_img = nib.load(str(img_path))
        nii_mask = nib.load(str(mask_path))
        img = nii_img.get_fdata()
        mask = nii_mask.get_fdata().astype(np.uint8)
        affine = nii_img.affine

        axis = self.slice_sampling_method
        if axis == "random_axis":
            axis_choice = random.choice([0, 1, 2])
        elif axis == "central":
            axis_choice = 2  # default axial
        else:
            axis_choice = 2

        if axis_choice == 2:
            valid_idx = np.where(mask.sum(axis=(1, 2)) > 0)[0]
            if len(valid_idx) == 0:
                z = img.shape[0] // 2
            else:
                z = int(random.choice(list(valid_idx)))
            image_2d = img[z, :, :]
            mask_2d = mask[z, :, :]
        elif axis_choice == 1:
            valid_idx = np.where(mask.sum(axis=(0, 2)) > 0)[0]
            if len(valid_idx) == 0:
                z = img.shape[1] // 2
            else:
                z = int(random.choice(list(valid_idx)))
            image_2d = img[:, z, :]
            mask_2d = mask[:, z, :]
        else:
            valid_idx = np.where(mask.sum(axis=(0, 1)) > 0)[0]
            if len(valid_idx) == 0:
                z = img.shape[2] // 2
            else:
                z = int(random.choice(list(valid_idx)))
            image_2d = img[:, :, z]
            mask_2d = mask[:, :, z]

        # compute pointmap based on axis and affine
        H, W = image_2d.shape
        xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        if axis_choice == 2:
            k = np.full_like(xv, z)
            voxel_coords = np.stack([xv, yv, k, np.ones_like(xv)], axis=-1)
        elif axis_choice == 1:
            k = np.full_like(xv, z)
            voxel_coords = np.stack([k, xv, yv, np.ones_like(xv)], axis=-1)
        else:
            k = np.full_like(xv, z)
            voxel_coords = np.stack([yv, k, xv, np.ones_like(xv)], axis=-1)

        world_coords = (affine @ voxel_coords.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]
        # Replace masked positions with zeros to avoid NaNs in downstream computations.
        xyz_masked = (
            np.where(mask_2d[..., None], world_coords, 0.0).astype(np.float32).transpose(2, 0, 1)
        )
        image_3ch = np.stack([image_2d, image_2d, image_2d], axis=0).astype(np.float32)

        return image_3ch, mask_2d, xyz_masked, affine, z

    def __getitem__(self, index):
        if self.cache_slices:
            file_path = self.file_names[index]
            data = np.load(str(file_path), allow_pickle=True)
            image = torch.from_numpy(data["image"]).float()
            mask = torch.from_numpy(data["mask"]).float()
            pointmap = torch.from_numpy(data["pointmap"]).float()
            affine = torch.from_numpy(data["affine"]).float() if "affine" in data else torch.eye(4)
            gt_sdf_path = data["gt_sdf_path"].item() if "gt_sdf_path" in data else None
            # Load per-volume SDF but return the 2D per-slice SDF corresponding to this sample
            gt_sdf = None
            if gt_sdf_path is not None:
                try:
                    gt_sdf_np = np.load(gt_sdf_path)
                except Exception:
                    gt_sdf_np = None
                if gt_sdf_np is not None:
                    # Slice index (axial by default) saved in cache during preprocessing
                    slice_idx = int(data["slice_idx"]) if "slice_idx" in data else 0
                    # Guard against mismatched axis orders or off-by-one indices by clamping
                    # the slice index to the valid range for the volume's depth.
                    if gt_sdf_np.ndim == 4:
                        max_d = gt_sdf_np.shape[1]
                    elif gt_sdf_np.ndim == 3:
                        max_d = gt_sdf_np.shape[0]
                    else:
                        max_d = 0
                    if max_d > 0:
                        slice_idx = min(slice_idx, max_d - 1)
                    # If SDF has shape (C, D, H, W)
                    if gt_sdf_np.ndim == 4:
                        # select the slice and keep channel dimension: (C, H, W)
                        gt_sdf_slice = gt_sdf_np[:, slice_idx, :, :]
                    # If SDF has shape (D, H, W), convert to (1, H, W)
                    elif gt_sdf_np.ndim == 3:
                        gt_sdf_slice = gt_sdf_np[slice_idx, :, :][None, ...]
                    else:
                        # Unexpected shape - fallback to None
                        gt_sdf_slice = None
                    if gt_sdf_slice is not None:
                        gt_sdf = torch.from_numpy(gt_sdf_slice.astype(np.float32))
            name = file_path.stem

            # Apply per-slice augmentations (before PreProcessor)
            aug_result = self.augmentor(image, mask, pointmap)
            image = aug_result["image"]
            mask = aug_result["mask"]
            pointmap = aug_result["pointmap"]

            # Use PreProcessor for transforms
            pp_return = self.preprocessor._process_image_mask_pointmap_mess(image, mask, pointmap)
            item = {
                "image": pp_return["image"],
                "mask": pp_return["mask"],
                "pointmap": pp_return.get("pointmap", None),
                "pointmap_scale": pp_return.get("pointmap_scale", None),
                "pointmap_shift": pp_return.get("pointmap_shift", None),
                "affine": affine,
                # Return per-slice SDF (C, H, W) for compatibility with per-slice models
                "mask_sdf": gt_sdf,
                "name": name,
            }
            # Optionally include default identity pose (if requested)
            if self.include_pose:
                # PoseTarget fields following ScaleShiftInvariant convention
                item["pose_target"] = {
                    "x_instance_scale": torch.ones(1, 1, 3),
                    "x_instance_rotation": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "x_instance_translation": torch.zeros(1, 1, 3),
                    "x_scene_scale": torch.ones(1, 1, 3),
                    "x_scene_center": torch.zeros(1, 1, 3),
                    "x_translation_scale": torch.ones(1, 1, 1),
                    "pose_target_convention": "ScaleShiftInvariant",
                }
            return item
        else:
            # sample on-the-fly from volumes: index maps into which volume and which slice
            case_idx = index % len(self.cases)
            image_3ch, mask_2d, pointmap, affine, slice_idx = self._sample_slice_on_the_fly(
                case_idx
            )
            image = torch.from_numpy(image_3ch).float()
            mask = torch.from_numpy(mask_2d).float()
            pointmap = torch.from_numpy(pointmap).float()
            affine = torch.from_numpy(affine).float()

            # Apply per-slice augmentations (before PreProcessor)
            aug_result = self.augmentor(image, mask, pointmap)
            image = aug_result["image"]
            mask = aug_result["mask"]
            pointmap = aug_result["pointmap"]

            # mix transforms via PreProcessor
            pp_return = self.preprocessor._process_image_mask_pointmap_mess(image, mask, pointmap)
            item = {
                "image": pp_return["image"],
                "mask": pp_return["mask"],
                "pointmap": pp_return.get("pointmap", None),
                "pointmap_scale": pp_return.get("pointmap_scale", None),
                "pointmap_shift": pp_return.get("pointmap_shift", None),
                "affine": affine,
                "mask_sdf": None,
                "name": f"{self.cases[case_idx][0].stem}_slice_{slice_idx:03d}",
            }
            if self.include_pose:
                item["pose_target"] = {
                    "x_instance_scale": torch.ones(1, 1, 3),
                    "x_instance_rotation": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "x_instance_translation": torch.zeros(1, 1, 3),
                    "x_scene_scale": torch.ones(1, 1, 3),
                    "x_scene_center": torch.zeros(1, 1, 3),
                    "x_translation_scale": torch.ones(1, 1, 1),
                    "pose_target_convention": "ScaleShiftInvariant",
                }
            return item


# collate function similar to TS loader
def _pad_to_target(tensor: torch.Tensor, target_h: int, target_w: int, pad_value=0.0) -> torch.Tensor:
    """Pad a tensor of shape (C,H,W) or (H,W) to (C,target_h,target_w) or (target_h,target_w).
    pad_value will be used for padding constant values (e.g., 0 or NaN).
    """
    if tensor is None:
        return None
    if tensor.ndim == 3:
        c, h, w = tensor.shape
    elif tensor.ndim == 2:
        c = None
        h, w = tensor.shape
    else:
        # Unexpected dims - return unchanged
        return tensor
    if h == target_h and w == target_w:
        return tensor
    # compute padding: pad=(left, right, top, bottom)
    pad_w = target_w - w
    pad_h = target_h - h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    # F.pad expects pad=(left, right, top, bottom)
    if c is None:
        padded = F.pad(tensor.unsqueeze(0), (left, right, top, bottom), value=pad_value).squeeze(0)
    else:
        padded = F.pad(tensor, (left, right, top, bottom), value=pad_value)
    return padded


def data_collate(batch):
    # Determine the max height & width in the batch to pad to
    heights = [item["image"].shape[-2] for item in batch]
    widths = [item["image"].shape[-1] for item in batch]
    max_h = max(heights)
    max_w = max(widths)

    # Pad images (pad with zeros)
    img_list = [
        _pad_to_target(item["image"], max_h, max_w, pad_value=0.0) for item in batch
    ]
    img = torch.stack(img_list)
    # mask_sdf may be of shape (C, D, H, W) or None; pad H/W dims if needed
    mask_sdf_list = []
    for item in batch:
        ms = item["mask_sdf"] if item["mask_sdf"] is not None else None
        if ms is None:
            # create zeros of (1,1,max_h,max_w)
            mask_sdf_list.append(torch.zeros((1, 1, max_h, max_w)))
        else:
            # ms can be (C,D,H,W), (C,H,W) or (H,W) depending on precomputed cache
            if ms.ndim == 4:
                c, d, h, w = ms.shape
            elif ms.ndim == 3:
                # (C,H,W) -> (C,1,H,W)
                c, h, w = ms.shape
                d = 1
                ms = ms.unsqueeze(1)
            elif ms.ndim == 2:
                # (H,W) -> (1,1,H,W)
                h, w = ms.shape
                c, d = 1, 1
                ms = ms.unsqueeze(0).unsqueeze(0)
            else:
                # Unexpected dims - convert to zeros
                mask_sdf_list.append(torch.zeros((1, 1, max_h, max_w)))
                continue
            if h != max_h or w != max_w:
                # pad (left,right,top,bottom) for last two dims
                pad_w = max_w - w
                pad_h = max_h - h
                left = pad_w // 2
                right = pad_w - left
                top = pad_h // 2
                bottom = pad_h - top
                # pad expects (left,right,top,bottom) and operates on last dims
                ms = F.pad(ms, (left, right, top, bottom), value=0.0)
            mask_sdf_list.append(ms)
    mask_sdf = torch.stack(mask_sdf_list)
    # segmentation masks may vary in H/W - pad with zeros
    segmentation = torch.stack([
        _pad_to_target(item["mask"], max_h, max_w, pad_value=0.0) for item in batch
    ])
    name = [item["name"] for item in batch]
    
    # Stack pointmaps (required for training)
    # pointmap: pad with NaN for unknown areas
    pointmap_list = []
    for item in batch:
        pm = item.get("pointmap", None)
        if pm is None:
            # generate NaN-filled pointmap with same channels as image (assume 3)
            ch = item["image"].shape[0]
            pm = torch.full((ch, max_h, max_w), float("nan"))
        else:
            pm = _pad_to_target(pm, max_h, max_w, pad_value=float("nan"))
        pointmap_list.append(pm)
    pointmap = torch.stack(pointmap_list)

    affine_tensors = []
    for item in batch:
        affine_value = item["affine"]
        if not isinstance(affine_value, torch.Tensor):
            affine_value = torch.from_numpy(np.asarray(affine_value)).float()
        affine_tensors.append(affine_value)
    affine = torch.stack(affine_tensors)

    batch_dict = {
        "image": img.float(),
        "mask_sdf": mask_sdf.float(),
        "segmentation": segmentation,
        "pointmap": pointmap.float(),
        "name": name,
        "affine": affine,
    }
    # optional pose targets
    if "pose_target" in batch[0]:
        # create a dictionary of stacked fields
        pose_keys = list(batch[0]["pose_target"].keys())
        stacked_pose = {}
        for k in pose_keys:
            stacked_pose[k] = torch.stack([item["pose_target"][k] for item in batch], dim=0)
        batch_dict["pose_target"] = stacked_pose
    return batch_dict


if __name__ == "__main__":
    # Quick sanity test / small run
    ds = TS_SAM3D_Dataset(
        original_nifti_dir="/mnt/nas1/disk01/weidongguo/dataset/TS", cache_slices=True, classes=5
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=data_collate)
    batch = next(iter(loader))
    print({k: (v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()})
