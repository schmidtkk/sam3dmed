#!/usr/bin/env python3
"""reprocess_ts_nifti.py
Preprocess raw NIfTI files for SAM3D medical fine-tuning.
- Resample volumes to isotropic spacing (adaptive per modality or fixed)
- Compute per-class SDFs and save per-case
- Optionally extract meshes using marching cubes
- Extract per-slice caches across specified axes (0,1,2) and save `.npz` files with: image, mask, pointmap, affine, slice_idx, axis, gt_sdf_path
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio

from utils.sdf_fn import compute_sdf

# Default spacing per modality (mm)
MODALITY_SPACING = {
    "ct": (1.0, 1.0, 1.0),
    "mri": (1.0, 1.0, 1.0),  # Cardiac MRI often 1.25-1.5mm in-plane
    "ultrasound": (0.5, 0.5, 0.5),  # Higher resolution for US
    "default": (1.0, 1.0, 1.0),
}


def detect_modality(img_path: Path) -> str:
    """Heuristic modality detection from filename or metadata."""
    name = img_path.name.lower()
    if "ct" in name or "ctce" in name:
        return "ct"
    elif "mr" in name or "mri" in name or "t1" in name or "t2" in name:
        return "mri"
    elif "us" in name or "ultrasound" in name:
        return "ultrasound"
    return "default"


def get_spacing_for_modality(
    modality: str, override: float | None = None
) -> tuple[float, float, float]:
    """Get target spacing for modality, with optional override."""
    if override is not None:
        return (override, override, override)
    return MODALITY_SPACING.get(modality, MODALITY_SPACING["default"])


def resample_volume(img_nii, target_spacing=(1.0, 1.0, 1.0)):
    subject = tio.Subject(image=tio.ScalarImage(img_nii))
    resample = tio.Resample(target_spacing)
    resampled = resample(subject)
    return resampled.image.numpy(), resampled.image.affine


def extract_and_cache_slices(
    img: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    case_id: str,
    out_dir: Path,
    axes: tuple[int, ...] = (0, 1, 2),
    sdf_path: Path | None = None,
    min_foreground_ratio: float = 0.01,
) -> int:
    """Extract per-slice caches and save as .npz files.

    Args:
        img: 3D image array (D, H, W)
        mask: 3D mask array (D, H, W)
        affine: 4x4 affine matrix
        case_id: case identifier string
        out_dir: output directory
        axes: which axes to slice along
        sdf_path: path to pre-computed SDF file (optional, stored as reference)
        min_foreground_ratio: skip slices with less foreground than this ratio

    Returns:
        Number of slices saved
    """
    saved_count = 0
    for axis in axes:
        depth = img.shape[axis]
        for idx in range(depth):
            if axis == 0:
                image_2d = img[idx, :, :]
                mask_2d = mask[idx, :, :]
            elif axis == 1:
                image_2d = img[:, idx, :]
                mask_2d = mask[:, idx, :]
            else:
                image_2d = img[:, :, idx]
                mask_2d = mask[:, :, idx]

            # Skip slices with insufficient foreground
            fg_ratio = (mask_2d > 0).sum() / mask_2d.size
            if fg_ratio < min_foreground_ratio:
                continue

            H, W = image_2d.shape
            xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
            if axis == 2:
                k = np.full_like(xv, idx)
                voxel_coords = np.stack([xv, yv, k, np.ones_like(xv)], axis=-1)
            elif axis == 1:
                k = np.full_like(xv, idx)
                voxel_coords = np.stack([k, xv, yv, np.ones_like(xv)], axis=-1)
            else:
                k = np.full_like(xv, idx)
                voxel_coords = np.stack([yv, k, xv, np.ones_like(xv)], axis=-1)

            world_coords = (affine @ voxel_coords.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]
            xyz_masked = (
                np.where(mask_2d[..., None] > 0, world_coords, np.nan)
                .astype(np.float32)
                .transpose(2, 0, 1)
            )
            image_3ch = np.stack([image_2d, image_2d, image_2d], axis=0).astype(np.float32)

            out_path = out_dir / f"{case_id}_axis{axis}_slice{idx:04d}.npz"
            save_dict = {
                "image": image_3ch,
                "mask": mask_2d.astype(np.uint8),
                "pointmap": xyz_masked,
                "affine": affine,
                "slice_idx": int(idx),
                "axis": int(axis),
            }
            if sdf_path is not None:
                save_dict["gt_sdf_path"] = str(sdf_path)

            np.savez_compressed(out_path, **save_dict)
            saved_count += 1

    return saved_count


def compute_per_class_sdf(mask: np.ndarray, out_path: Path, classes: int) -> np.ndarray:
    """Compute per-class signed distance fields.

    Args:
        mask: 3D label mask (D, H, W) with values 0..classes
        out_path: where to save the SDF array
        classes: number of foreground classes (1..classes)

    Returns:
        SDF array of shape (classes, D, H, W)
    """
    sdf_channels = []
    for c in range(1, classes + 1):
        sdf_c = compute_sdf((mask == c).astype(np.uint8))
        sdf_channels.append(sdf_c.astype(np.float32))
    mask_sdf = np.stack(sdf_channels, axis=0)
    np.save(out_path, mask_sdf)
    return mask_sdf


def extract_mesh(
    mask: np.ndarray, affine: np.ndarray, out_path: Path, class_id: int = 1, level: float = 0.5
) -> bool:
    """Extract mesh from binary mask using marching cubes.

    Args:
        mask: 3D label mask
        affine: 4x4 affine matrix for world coordinates
        out_path: output .obj file path
        class_id: which class to extract (default 1)
        level: isosurface level for marching cubes

    Returns:
        True if mesh was successfully extracted, False otherwise
    """
    try:
        from skimage import measure
    except ImportError:
        print("Warning: skimage not available, skipping mesh extraction")
        return False

    binary_mask = (mask == class_id).astype(np.float32)
    if binary_mask.sum() < 100:  # too small to mesh
        return False

    try:
        verts, faces, normals, values = measure.marching_cubes(binary_mask, level=level)
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return False

    # Transform vertices to world coordinates
    ones = np.ones((verts.shape[0], 1))
    verts_homo = np.hstack([verts, ones])
    verts_world = (affine @ verts_homo.T).T[:, :3]

    # Write OBJ file
    with open(out_path, "w") as f:
        for v in verts_world:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fn in normals:
            f.write(f"vn {fn[0]:.6f} {fn[1]:.6f} {fn[2]:.6f}\n")
        for face in faces:
            f.write(
                f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n"
            )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw NIfTI files for SAM3D medical fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original_nifti_dir", required=True, help="Input directory with raw NIfTI files"
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--classes", type=int, default=5, help="Number of foreground classes")
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Target isotropic spacing (mm). If None, uses adaptive per-modality defaults",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["ct", "mri", "ultrasound"],
        help="Force modality (otherwise auto-detect)",
    )
    parser.add_argument(
        "--axes", default="0,1,2", help="Comma-separated list of axes to slice along"
    )
    parser.add_argument(
        "--extract_mesh", action="store_true", help="Extract meshes using marching cubes"
    )
    parser.add_argument(
        "--min_foreground_ratio",
        type=float,
        default=0.01,
        help="Skip slices with less foreground than this ratio",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--save_metadata", action="store_true", help="Save processing metadata as JSON"
    )
    args = parser.parse_args()

    src_dir = Path(args.original_nifti_dir)
    dst_dir = Path(args.out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    axes = tuple(int(x) for x in args.axes.split(","))

    # Find pairs of img/mask - handle multiple naming conventions
    # Pattern 1: *_img.nii.gz / *_seg.nii.gz
    # Pattern 2: *-image.nii.gz / *-label.nii.gz (TS dataset style)
    imgs = list(src_dir.glob("**/*img*.nii*"))
    if len(imgs) == 0:
        # Try TS dataset pattern
        imgs = list(src_dir.glob("**/*-image.nii*"))
    if len(imgs) == 0:
        imgs = [
            p
            for p in src_dir.glob("**/*.nii*")
            if not p.name.lower().endswith(("_seg.nii.gz", "_mask.nii.gz", "-label.nii.gz"))
        ]

    metadata = {
        "source_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "classes": args.classes,
        "axes": list(axes),
        "cases": [],
    }

    for img_path in imgs:
        # try to find mask - handle multiple naming conventions
        base_name = img_path.stem.replace("-image", "").replace("_img", "")
        mask_candidates = list(img_path.parent.glob(base_name + "*seg*.nii*")) + \
                          list(img_path.parent.glob(base_name + "*mask*.nii*")) + \
                          list(img_path.parent.glob(base_name + "*-label*.nii*")) + \
                          list(img_path.parent.glob(base_name + "*_label*.nii*"))
        if len(mask_candidates) == 0:
            mask_candidates = list(img_path.parent.glob("*seg*.nii*")) + \
                              list(img_path.parent.glob("*-label*.nii*"))
        if len(mask_candidates) == 0:
            print(f"No mask found for {img_path}, skipping")
            continue

        mask_path = mask_candidates[0]
        # Extract case_id from image path - handle both patterns
        case_id = img_path.stem.replace("-image", "").replace("_img", "")
        print(f"Processing case {case_id} ...")

        # Detect modality and get spacing
        modality = args.modality or detect_modality(img_path)
        target_spacing = get_spacing_for_modality(modality, args.spacing)
        print(f"  Modality: {modality}, target spacing: {target_spacing}")

        try:
            img_nii = nib.load(str(img_path))
            mask_nii = nib.load(str(mask_path))
        except Exception as e:
            print(f"  Error loading NIfTI: {e}, skipping")
            continue

        # Handle 4D images (take first volume)
        img_data = img_nii.get_fdata()
        if img_data.ndim == 4:
            print("  4D image detected, using first volume")
            img_data = img_data[..., 0]
            # Create new NIfTI with 3D data
            img_nii = nib.Nifti1Image(img_data, img_nii.affine, img_nii.header)

        mask_data = mask_nii.get_fdata()
        if mask_data.ndim == 4:
            mask_data = mask_data[..., 0]
            mask_nii = nib.Nifti1Image(mask_data, mask_nii.affine, mask_nii.header)

        # Resample to target spacing
        try:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img_data[np.newaxis], affine=img_nii.affine),
                mask=tio.LabelMap(
                    tensor=mask_data[np.newaxis].astype(np.int32), affine=mask_nii.affine
                ),
            )
            resample = tio.Resample(target_spacing)
            res = resample(subject)
            img_np = res.image.numpy().squeeze(0)  # Remove channel dim
            mask_np = res.mask.numpy().squeeze(0).astype(np.uint8)
            affine = res.image.affine
        except Exception as e:
            print(f"  Resampling failed: {e}, skipping")
            continue

        print(f"  Resampled shape: {img_np.shape}")

        # Compute per-class SDF
        sdf_out_path = dst_dir / f"{case_id}_sdf.npy"
        if not sdf_out_path.exists() or args.overwrite:
            print(f"  Computing SDFs for {args.classes} classes...")
            compute_per_class_sdf(mask_np, sdf_out_path, classes=args.classes)

        # Extract meshes if requested
        mesh_paths = []
        if args.extract_mesh:
            mesh_dir = dst_dir / "meshes"
            mesh_dir.mkdir(exist_ok=True)
            for c in range(1, args.classes + 1):
                mesh_path = mesh_dir / f"{case_id}_class{c}.obj"
                if not mesh_path.exists() or args.overwrite:
                    success = extract_mesh(mask_np, affine, mesh_path, class_id=c)
                    if success:
                        mesh_paths.append(str(mesh_path))
                        print(f"  Extracted mesh for class {c}")

        # Extract & cache slices across axes
        num_slices = extract_and_cache_slices(
            img_np,
            mask_np,
            affine,
            case_id,
            dst_dir,
            axes=axes,
            sdf_path=sdf_out_path,
            min_foreground_ratio=args.min_foreground_ratio,
        )
        print(f"  Saved {num_slices} slices")

        # Record metadata
        case_meta = {
            "case_id": case_id,
            "modality": modality,
            "spacing": target_spacing,
            "shape": list(img_np.shape),
            "num_slices": num_slices,
            "sdf_path": str(sdf_out_path),
            "mesh_paths": mesh_paths,
        }
        metadata["cases"].append(case_meta)

    # Save metadata
    if args.save_metadata:
        meta_path = dst_dir / "processing_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")

    print(f"\nDone. Processed {len(metadata['cases'])} cases.")


if __name__ == "__main__":
    main()
