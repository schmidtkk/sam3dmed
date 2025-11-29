#!/usr/bin/env python3
"""Adapted TS dataloader for nnUNet preprocessed data."""

import glob
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.sdf_fn import compute_sdf

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _ensure_range(value, default):
    """Normalize config values to a (min, max) tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric, numeric
    return default


def _clamp_tensor(tensor: torch.Tensor, clamp_range):
    if clamp_range is None:
        return tensor
    if isinstance(clamp_range, (list, tuple)) and len(clamp_range) == 2:
        return tensor.clamp(float(clamp_range[0]), float(clamp_range[1]))
    return tensor


def _make_domain_shift_lambda(cfg):
    contrast_range = _ensure_range(cfg.get("contrast_range"), (0.3, 1.7))
    brightness_range = _ensure_range(cfg.get("brightness_range"), (-0.6, 0.6))
    blur_kernel = int(cfg.get("blur_kernel_size", 3) or 0)
    blur_iterations = int(cfg.get("blur_iterations", 1) or 0)
    voxel_dropout_prob = float(cfg.get("voxel_dropout_prob", 0.35))
    clamp_range = cfg.get("clamp_range", [-4.5, 4.5])

    if blur_kernel > 0 and blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel < 3:
        blur_iterations = 0

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor

        if blur_iterations > 0:
            padding = blur_kernel // 2
            kernel = blur_kernel
            needs_batch = out.dim() == 4
            if needs_batch:
                out = out.unsqueeze(0)
            for _ in range(blur_iterations):
                out = F.avg_pool3d(
                    out, kernel_size=kernel, stride=1, padding=padding, count_include_pad=False
                )
            if needs_batch:
                out = out.squeeze(0)

        if contrast_range is not None:
            min_contrast, max_contrast = contrast_range
            if max_contrast - min_contrast > 0:
                factor = torch.empty(1).uniform_(min_contrast, max_contrast).item()
                out = out * factor

        if brightness_range is not None:
            min_shift, max_shift = brightness_range
            if max_shift - min_shift > 0:
                shift = torch.empty(1).uniform_(min_shift, max_shift).item()
                out = out + shift

        if 0.0 < voxel_dropout_prob < 1.0:
            dropout_mask = torch.rand_like(out) > voxel_dropout_prob
            out = out * dropout_mask.float()

        out = _clamp_tensor(out, clamp_range)
        return out

    return transform


def build_domain_shift_transform(cfg: dict | None) -> tio.transforms.Transform | None:
    """Create TorchIO transform sequence for domain shift simulation."""
    if not cfg or not cfg.get("enabled", False):
        return None

    noise_range = _ensure_range(cfg.get("noise_std"), (0.25, 0.45))
    gamma_range = _ensure_range(cfg.get("gamma_log"), (-0.9, 0.9))

    transforms: list[tio.transforms.Transform] = []
    if noise_range is not None:
        transforms.append(tio.RandomNoise(std=noise_range, include=["image"], p=1.0))
    if gamma_range is not None:
        transforms.append(tio.RandomGamma(log_gamma=gamma_range, include=["image"], p=1.0))

    transforms.append(tio.Lambda(_make_domain_shift_lambda(cfg), include=["image"]))
    return tio.Compose(transforms)


def GetTrainTransforms(target_shape, stage_two=False):
    """Training augmentations for nnUNet preprocessed data.

    Based on analysis of 50 cases:
    - Range: [-3.434, 3.784]
    - Mean: ~ -1.547 (not standard z-score)
    - Std: ~1.562 (not standard z-score)
    - nnUNet uses some form of robust normalization, not standard z-score
    """
    if not stage_two:
        # Basic augmentations for stage 1 training
        train_transforms = tio.Compose(
            [
                tio.CropOrPad(target_shape=target_shape),  # Apply first to ensure consistent size
                tio.RandomNoise(std=(0, 0.15), p=0.5, include=["image"]),
                tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.5, include=["image"]),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=(5, 5, 5),
                    translation=(4, 4, 4),
                    image_interpolation="linear",
                    label_interpolation="nearest",
                    p=0.5,
                ),
                # tio.RandomFlip(axes=(0, 1, 2), p=0.5),  # Spatial augmentation: random flip
                tio.Lambda(lambda x: x.clamp(-3.5, 4.0), include=["image"]),
            ]
        )
    else:
        # Enhanced augmentations for stage 2 training
        p = 0.5
        train_transforms = tio.Compose(
            [
                tio.CropOrPad(target_shape=target_shape),  # Apply first to ensure consistent size
                tio.RandomNoise(std=(0, 0.2), p=p, include=["image"]),
                tio.RandomGamma(log_gamma=(-0.5, 0.5), p=p, include=["image"]),
                tio.RandomAffine(
                    scales=(0.85, 1.15),
                    degrees=(8, 8, 8),
                    translation=(6, 6, 6),
                    image_interpolation="linear",
                    label_interpolation="nearest",
                    p=p,
                ),
                # tio.RandomFlip(axes=(0, 1, 2), p=0.5),  # Spatial augmentation: random flip
                tio.RandomElasticDeformation(  # Spatial augmentation: elastic deformation
                    num_control_points=7, max_displacement=7.5, image_interpolation="linear", p=0.25
                ),
                tio.Lambda(lambda x: x.clamp(-3.5, 4.0), include=["image"]),
            ]
        )
    return train_transforms


def GetTestTransforms(target_shape, domain_shift_cfg: dict | None = None):
    """Test transforms with optional domain shift degradation."""
    transforms: list[tio.transforms.Transform] = [
        tio.CropOrPad(target_shape=target_shape),  # Ensure consistent size
        tio.Lambda(lambda x: x.clamp(-3.5, 4.0), include=["image"]),
    ]

    domain_shift_transform = build_domain_shift_transform(domain_shift_cfg)
    if domain_shift_transform is not None:
        transforms.append(domain_shift_transform)

    return tio.Compose(transforms)


class TS_nnUNet_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        stage_two=False,
        num_foreground_classes=0,
        predict_background_sdf=False,
        categorical_use_background=True,
        split_config: dict | None = None,
        domain_shift_config: dict | None = None,
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.stage_two = stage_two
        self.num_foreground_classes = int(num_foreground_classes)
        if self.num_foreground_classes <= 0:
            raise ValueError("num_foreground_classes must be positive")
        self.predict_background_sdf = bool(predict_background_sdf)
        self.categorical_use_background = bool(categorical_use_background)
        self.categorical_ignore_index = -1
        self.num_categorical_classes = self.num_foreground_classes + (
            1 if self.categorical_use_background else 0
        )
        self.split_config = split_config or {}
        self.domain_shift_config = domain_shift_config or {}
        self._apply_domain_shift = False

        if isinstance(self.domain_shift_config, dict):
            apply_modes = self.domain_shift_config.get("apply_to_modes", ["test"])
            if isinstance(apply_modes, str):
                apply_modes = [apply_modes]
            apply_modes = {str(item).lower() for item in (apply_modes or [])}
            self._apply_domain_shift = (
                bool(self.domain_shift_config.get("enabled", False))
                and self.mode.lower() in apply_modes
            )
        else:
            self.domain_shift_config = {}

        print(
            f"DEBUG: TS_nnUNet_Dataset mode={self.mode}, _apply_domain_shift={self._apply_domain_shift}, domain_shift_config={self.domain_shift_config}"
        )
        env_root = os.environ.get("nnUNet_preprocessed")
        default_root = "/mnt/nas1/disk01/weidongguo/workspace/diffseg-master/nnUNet_preprocessed"
        root = env_root or default_root

        # Set nnUNet preprocessed directory path (in workspace, not dataset dir)
        self.nnunet_preprocessed_dir = os.path.join(
            root,
            "Dataset001_TS_Heart",
            "nnUNetPlans_3d_fullres",
        )

        # Get target shape from config
        self.target_shape = (
            int(split_config.get("roi_z", 64)),
            int(split_config.get("roi_y", 64)),
            int(split_config.get("roi_x", 64)),
        )

        # Initialize transforms based on mode
        if mode == "train":
            self.transforms = GetTrainTransforms(self.target_shape, stage_two)
        else:
            domain_cfg = self.domain_shift_config if self._apply_domain_shift else None
            self.transforms = GetTestTransforms(self.target_shape, domain_cfg)

        self.file_names = self.get_file_names()

    def get_file_names(self):
        """Get file names from nnUNet preprocessed data respecting split configuration."""
        if not self.nnunet_preprocessed_dir:
            raise ValueError("nnUNet_preprocessed environment variable not set")

        # The nnunet_preprocessed_dir already points to the nnUNetPlans_3d_fullres directory
        data_dir = self.nnunet_preprocessed_dir

        if not os.path.exists(data_dir):
            raise ValueError(f"nnUNet preprocessed data not found at {data_dir}")

        # Get all .pkl files (metadata files)
        all_files = glob.glob(os.path.join(data_dir, "case_*.pkl"))
        if not all_files:
            raise RuntimeError(f"No preprocessed cases found in {data_dir}")

        case_map = {Path(path).stem: path for path in all_files}
        all_case_names = sorted(case_map.keys())
        selected_case_names = self._select_case_subset(all_case_names)

        missing_cases = [name for name in selected_case_names if name not in case_map]
        if missing_cases:
            raise RuntimeError(f"Split references unknown cases: {missing_cases[:5]}")

        return [case_map[name] for name in selected_case_names]

    def _select_case_subset(self, all_case_names: Sequence[str]) -> list[str]:
        """Select the subset of case names according to the deterministic split."""
        split_mapping = self._load_or_create_split(all_case_names)

        split_key = "train"
        if self.mode == "val":
            split_key = "val"
        elif self.mode == "test":
            split_key = "test"

        if split_key not in split_mapping:
            if split_key == "val" and "validation" in split_mapping:
                split_key = "validation"
            elif split_key == "val" and "test" in split_mapping:
                split_key = "test"
            else:
                raise KeyError(f"Split key '{split_key}' not found in split configuration")

        selected = split_mapping.get(split_key, [])
        if not selected:
            raise RuntimeError(f"Split '{split_key}' contains no cases")
        return selected

    def _load_or_create_split(self, all_case_names: Sequence[str]) -> dict[str, list[str]]:
        """Load an existing split definition or create a new one deterministically."""
        json_path = self.split_config.get("json_path")
        if json_path:
            split_path = Path(json_path)
            if not split_path.is_absolute():
                split_path = WORKSPACE_ROOT / split_path
        else:
            split_path = WORKSPACE_ROOT / "dataset" / "splits" / "ts_nnunet_split.json"

        split_path.parent.mkdir(parents=True, exist_ok=True)

        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return self._validate_split_content(data, all_case_names, split_path)

        split_data = self._generate_split(all_case_names)
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(split_data, handle, indent=2)
        print(f"[TS_nnUNet_Dataset] Created dataset split at {split_path}")
        return split_data

    def _validate_split_content(
        self,
        data: dict[str, list[str]],
        all_case_names: Sequence[str],
        split_path: Path,
    ) -> dict[str, list[str]]:
        """Validate split file content against available cases."""
        required_keys = {"train", "val", "test"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise KeyError(f"Split file {split_path} missing keys: {sorted(missing_keys)}")

        available_cases = set(all_case_names)
        validated: dict[str, list[str]] = {}
        for key in required_keys:
            entries = data.get(key, [])
            unknown = sorted(set(entries) - available_cases)
            if unknown:
                raise ValueError(
                    f"Split file {split_path} references unknown cases in '{key}': {unknown[:5]}"
                )
            validated[key] = sorted(set(entries))

        return validated

    def _generate_split(self, all_case_names: Sequence[str]) -> dict[str, list[str]]:
        """Generate a deterministic split using configured ratios and seed."""
        seed = int(self.split_config.get("seed", 42))
        train_ratio = float(self.split_config.get("train_ratio", 0.8))
        val_ratio = float(self.split_config.get("val_ratio", 0.1))
        test_ratio = float(self.split_config.get("test_ratio", 0.1))

        if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("Split ratios must sum to a positive value")

        num_cases = len(all_case_names)
        if num_cases == 0:
            raise RuntimeError("No cases available to generate splits")

        # Normalize ratios to sum to 1
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio = 1.0 - train_ratio - val_ratio

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(num_cases)

        train_count = math.floor(train_ratio * num_cases)
        val_count = math.floor(val_ratio * num_cases)
        test_count = num_cases - train_count - val_count
        if test_count <= 0:
            raise ValueError("Split configuration leaves no samples for the test split")

        train_indices = shuffled_indices[:train_count]
        val_indices = shuffled_indices[train_count : train_count + val_count]
        test_indices = shuffled_indices[train_count + val_count :]

        return {
            "train": sorted(all_case_names[idx] for idx in train_indices),
            "val": sorted(all_case_names[idx] for idx in val_indices),
            "test": sorted(all_case_names[idx] for idx in test_indices),
        }

    def load_nnUNet_data(self, pkl_file):
        """Load nnUNet preprocessed data from .pkl and .b2nd files"""
        # Load metadata
        properties = load_pickle(pkl_file)

        # Get corresponding .b2nd files
        data_file = pkl_file.replace(".pkl", ".b2nd")
        seg_file = pkl_file.replace(".pkl", "_seg.b2nd")

        # Load data and segmentation using blosc2 (nnUNet's compressed format)
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}

        data = blosc2.open(urlpath=data_file, mode="r", dparams=dparams, **mmap_kwargs)
        seg = blosc2.open(urlpath=seg_file, mode="r", dparams=dparams, **mmap_kwargs)

        # Convert to numpy arrays if needed
        if hasattr(data, "shape"):
            data = np.asarray(data)
        if hasattr(seg, "shape"):
            seg = np.asarray(seg)

        # nnUNet data already has channel dimension [C, D, H, W]
        # Don't add extra dimensions
        img = torch.from_numpy(data).float()
        mask = torch.from_numpy(seg).float()

        return img, mask, properties

    def apply_transform(self, image, label):
        """Apply TorchIO transforms to nnUNet data."""
        subject = tio.Subject(
            image=image,
            label=label,
        )
        transformed = self.transforms(subject)
        img_transformed = transformed.image
        mask_transformed = transformed.label
        return img_transformed, mask_transformed

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        pkl_file = self.file_names[index]

        # Load nnUNet preprocessed data
        img, mask, _ = self.load_nnUNet_data(pkl_file)

        # Extract case name
        name = os.path.basename(pkl_file).replace(".pkl", "")

        # Apply configured transforms (train augments or test domain shift)
        img_tio = tio.ScalarImage(tensor=img, affine=torch.eye(4))
        mask_tio = tio.LabelMap(tensor=mask, affine=torch.eye(4))

        if self.transforms is not None:
            img_tio, mask_tio = self.apply_transform(img_tio, mask_tio)

        img = img_tio.data
        mask = mask_tio.data

        # nnUNet already applied CTNormalization, so intensities are approximately z-scored.
        # Keep identity affine placeholder for downstream compatibility.
        affine = torch.eye(4)

        # Prepare categorical mask (with optional background removal)
        mask_categorical = mask.long()
        if not self.categorical_use_background:
            # Shift foreground labels down by one and mark background as ignore_index
            mask_categorical = mask_categorical - 1
            mask_categorical = torch.where(
                mask_categorical >= 0,
                mask_categorical,
                torch.full_like(mask_categorical, self.categorical_ignore_index),
            )

        # Compute SDF for each class on CPU
        mask_np = np.squeeze(mask.cpu().numpy())
        sdf_channels = []

        if self.predict_background_sdf:
            background_mask = (mask_np == 0).astype(np.float32)
            background_sdf = compute_sdf(background_mask)
            sdf_channels.append(torch.from_numpy(background_sdf).float())

        for class_idx in range(1, self.num_foreground_classes + 1):
            class_mask = (mask_np == class_idx).astype(np.float32)
            class_sdf = compute_sdf(class_mask)
            sdf_channels.append(torch.from_numpy(class_sdf).float())

        label_sdf = torch.stack(sdf_channels, dim=0)

        sample = {"name": name, "img": img, "mask_sdf": label_sdf, "mask": mask, "affine": affine}

        sample["categorical_mask"] = mask_categorical

        return sample


def data_collate(batch):
    img = torch.stack([item["img"] for item in batch])
    mask_sdf = torch.stack([item["mask_sdf"] for item in batch])
    segmentation = torch.stack([item["mask"] for item in batch])
    categorical_mask = torch.stack([item["categorical_mask"] for item in batch])
    name = [item["name"] for item in batch]
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
        "categorical_mask": categorical_mask.long(),
        "name": name,
        "affine": affine,
    }
    return batch_dict


def get_loader(cfg, mode="train", is_distributed=False, stage_two=False, root_override=None):
    assert mode in ["train", "val", "test"]
    root_dir = root_override or cfg["root_dir"]
    if mode == "val" and root_override is None:
        root_dir = cfg.get("val_root_dir", root_dir)

    if mode == "train":
        batch_size = cfg["batch_size"]
        num_workers = cfg.get("num_workers", 8)
    else:
        batch_size = cfg.get("test_batch_size", cfg["batch_size"])
        num_workers = cfg.get("val_num_workers", cfg.get("num_workers", 8))

    print("batch_size:", batch_size)

    split_config = cfg.get("split_config")
    if split_config is not None:
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(split_config):
                split_config = OmegaConf.to_container(split_config, resolve=True)
        except Exception:
            try:
                split_config = dict(split_config)
            except Exception:
                split_config = None

    # Ensure split_config has roi dimensions
    if split_config is None:
        split_config = {}
    split_config["roi_x"] = cfg.get("roi_x", 64)
    split_config["roi_y"] = cfg.get("roi_y", 64)
    split_config["roi_z"] = cfg.get("roi_z", 64)

    domain_shift_cfg = cfg.get("domain_shift")
    if domain_shift_cfg is not None:
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(domain_shift_cfg):
                domain_shift_cfg = OmegaConf.to_container(domain_shift_cfg, resolve=True)
        except Exception:
            try:
                domain_shift_cfg = dict(domain_shift_cfg)
            except Exception:
                domain_shift_cfg = None

    dataset = TS_nnUNet_Dataset(
        root_dir=root_dir,
        mode=mode,
        stage_two=stage_two,
        num_foreground_classes=int(cfg.get("mask_classes", 0)),
        predict_background_sdf=bool(cfg.get("predict_background_sdf", False)),
        categorical_use_background=bool(cfg.get("categorical_use_background", True)),
        split_config=split_config,
        domain_shift_config=domain_shift_cfg,
    )

    if is_distributed:
        # Use distributed sampler when using DDP
        sampler = DistributedSampler(dataset)
        shuffle = False  # Don't shuffle when using DistributedSampler
    else:
        sampler = None
        shuffle = True if mode == "train" else False

    drop_last = True if mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # Only shuffle if no sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=data_collate,
    )
    return dataloader
