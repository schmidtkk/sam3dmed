# Data Preprocessing Guide for Cardiac MRI (3D volumes) -> SAM3D-Object Fine-tuning

This document describes a reproducible and explicit workflow for converting your 3D cardiac MRI dataset (NIfTI `.nii.gz` images + label `.nii.gz` segmentation masks) into the format required by the SAM3D-Object training / inference pipelines in this repository.

It covers:
- How to extract 2D slices and create per-slice inputs: image, mask, and 3-channel `pointmap` (x,y,z). 
- How to prepare 3D ground truth shapes (volumetric occupancy, SDF, and mesh) for supervision of SAM3D decoders (mesh or gaussian spline decoders). 
- Best-practice normalization and transform settings consistent with repo utilities such as `PreProcessor`, `SSIPointmapNormalizer`, `PointPatchEmbed`, and the `TS_nnUNet_Dataset` loader.
- Implementation example snippets and tips to integrate outputs with the `InferencePipelinePointMap` and `Pipeline` utilities.

---

## Quick overview: Expected Items Per-Sample (what the pipeline expects)

Key fields used in the repo pipeline for a single training sample (single 2D slice + the 3D GT):
- `image` (torch Tensor, C x H x W): RGB (3 channels) or RGBA (4 channels with alpha for mask overlay) 2D image
- `mask` (H x W or 1 x H x W): binary/label segmentation for the slice
- `pointmap` (3 x H x W): (x, y, z) coordinates per pixel, mapping each foreground pixel to 3D coordinates; background pixels should be NaN
- `pointmap_scale`, `pointmap_shift`: normalization parameters created by `SSIPointmapNormalizer` or related normalizers
- `affine` (4 x 4): affine matrix that maps voxel indices to world coordinates (real mm coordinates) — important for building accurate point maps and shape metrics
- `gt_3d`: ground-truth 3D object representation (one of: TSDF voxel grid, binary occupancy grid, mesh (.ply/.obj), or precomputed gaussian splats)

Note: The repo uses `PreProcessor` (see `sam3d_objects/data/dataset/tdfy/preprocessor.py`) to apply joint transforms and to compute/normalize pointmaps. The `PointPatchEmbed` expects the `pointmap` to be of shape (B, 3, H, W) and to contain NaNs where points are invalid.

---

## 1) Verify your raw NIfTI data and preprocessed folder

1. Check the directory structure used by the repo (the dataset should be under `dataset/ts_processed` after preprocessing):
   - `*.npz` files hold preprocessed slices with image, mask, and pointmap data.
   - `meshes/*.obj` files contain ground-truth 3D meshes extracted from the volume segmentation masks.
   - For original NIfTI files (.nii / .nii.gz), the raw data is located at `/mnt/nas1/disk01/weidongguo/dataset/TS` and contains the true `affine` transforms, voxel spacing, and world coordinates.

2. If you have legacy nnUNet preprocessed data, note that it may have removed or resampled spatial metadata. For accurate SAM3D reconstruction, use the provided preprocessing scripts to process raw NIfTI files directly. This ensures correct `affine`, spacing and orientation for pointmap computation.

3. If `affine` is missing in preprocessed data, re-associate it with the original NIfTI for accurate world coordinates and shape extraction.

---

## 2) Recommended overall processing strategy (both for training and for inference compatibility)

1. Precompute per-case (or per-sample) files that contain:
   - Full 3D volumes: image (D,H,W) and mask (D,H,W), plus `affine` and voxel spacing.
   - 3D GT representations: SDF, occupancy volume (binary mask), and optionally extracted `mesh.ply`
   - Each 2D slice extracted across chosen planes saved as `.npz` or `.pt` with keys (`image`, `mask`, `pointmap`, `affine`, `slice_index`, `gt_3d_ref`/mesh pointer)

2. Use volumetric SDF as the canonical 3D ground-truth for slat/voxel supervision and/or mesh extraction for mesh-based decoders. Computing SDF is in `TS_nnUNet_Dataset` (see `compute_sdf` usage) and expected for `mask_sdf` channels.

3. For single-slice conditioning tasks: create `pointmap` per 2D slice using the world coordinates (via `affine`) so that `pointmap` matches the coordinate frame used by the `SSIPointmapNormalizer`.

4. Ensure all transforms match downstream `PreProcessor` and `PointPatchEmbed` expectations:
   - Resize and center pad to square, then resize to `PointPatchEmbed.input_size` (recommended 256 or 512 depending on your config).
   - Convert single-channel MRI to 3-channel (duplicate) or use 3-channel color mapping if you prefer.
   - Use the `SSIPointmapNormalizer` (or `ObjectApparentSizeSSI`) for consistent normalization across data.

---

## 3) 2D slice extraction: details & options

### A. Which plane(s) to extract
- Axial (default for many cardiac MRI sequences) - good start.
- Consider adding Coronal and Sagittal to increase diversity and robustness.
- For 3D MR volumes where in-plane resolution >> through-plane, resampling to isotropic spacing first (or using preprocessed isotropic outputs) is easier.

### B. Which slice indices to select
- Central slice of the organ: compute 3D mask bounding box and pick the central index (recommended for faster convergence and alignment).
- Sample multiple slices uniformly across the mask bounding box (for each case extract N slices per volume, e.g., 3–10 depending on volume depth).
- Avoid slices with very sparse mask pixels: use a mask-occupancy threshold (e.g., 0.01–0.05 of pixels inside the mask).
- For single-slice training, consider randomly sampling slices during training as augmentation.

### C. Practical slice extraction code (example using nibabel)

```python
import nibabel as nib
import numpy as np

# Load the 3D scan and its segmentation mask
nii_img = nib.load('volume.nii.gz')
img = nii_img.get_fdata()  # shape (D, H, W) or (H, W, D) depending on file
affine = nii_img.affine  # 4x4

mask_nii = nib.load('mask.nii.gz')
mask = mask_nii.get_fdata().astype(np.uint8)  # same shape as img

# Suppose img and mask are (D, H, W). Choose axial slices (z direction)
z = 32  # example slice index
img_slice = img[z, :, :]  # 2D
mask_slice = mask[z, :, :]
```

### D. Convert single-channel MRI to 3-channel RGB/gray
- Duplicate the single channel to 3 channels (RGB) to match the RGB pipeline:

```python
img_slice_3ch = np.stack([img_slice, img_slice, img_slice], axis=0)  # shape (3,H,W)
```

- If you want to normalize to [0,1] before converting, use dataset statistics or per-volume z-score.
- The repo often uses `img_transform` and `IMAGENET_NORMALIZATION` only when needed. For medical images you should not use ImageNet normalizations; instead, z-score per volume or match nnUNet normalization (nnUNet normalizes by dataset-specific intensity normalizations). 

### E. Recommended normalization (for Cardiac MRI)
- If you have preprocessed nnUNet data, the images are often already z-scored. Use the same scale.
- Otherwise:
  - Clip intensities to a fixed range (optional), e.g., [-1000, 1000] for CT — for MR, empirically compute percentiles and clip (e.g., 0.5th - 99.5th percentiles).
  - Standardize per-volume: (I - mean(volume)) / std(volume)
  - Save scale values if needed for downstream normalization.

---

## 4) Pointmap creation from a 2D slice (the most crucial step)

Pointmaps are 3-channel maps containing 3D coordinates for each pixel in camera-space (or world space depending on convention). The model expects a pointmap of shape (3, H, W) and the `PointPatchEmbed` and `SSIPointmapNormalizer` will handle normalization — but you should provide metric coordinates if available.

When your NIfTI data includes `affine`, compute the 3D world coordinates of every pixel in the slice as follows:

- For a 2D axial slice at index `z` (0-based index): voxel indices are (i, j, k) where k is `z` and i,j are pixel coords. For each pixel (i, j):
  - Build the homogeneous coordinate: [i, j, k, 1]
  - `world_coords = affine @ [i, j, k, 1]`  → `[x_mm, y_mm, z_mm]`

- If you want `pointmap` in camera coordinates (view-space), you can transform points with a canonical camera transform: for example, set camera's look-at transform such that view is orthographic along the slice direction. In many cases providing world coordinates is sufficient and `SSIPointmapNormalizer` will compute normalization.

Code snippet (NIfTI -> pointmap) — RANDOM AXIS sampling per user requirements:

```python
import numpy as np
import nibabel as nib
import random

nii_img = nib.load('volume.nii.gz')
img = nii_img.get_fdata()
affine = nii_img.affine  # 4x4

# Choose a random axis (0=x, 1=y, 2=z) and sample a slice index along it
axis = random.choice([0, 1, 2])
if axis == 2:
  H, W = img.shape[1], img.shape[2]
elif axis == 1:
  H, W = img.shape[0], img.shape[2]
else:
  H, W = img.shape[0], img.shape[1]

slice_idx = random.randint(0, img.shape[axis] - 1)

if axis == 2:
  img_slice = img[slice_idx, :, :]
  mask_slice = mask[slice_idx, :, :]
elif axis == 1:
  img_slice = img[:, slice_idx, :]
  mask_slice = mask[:, slice_idx, :]
else:
  img_slice = img[:, :, slice_idx]
  mask_slice = mask[:, :, slice_idx]

# Build grid of pixel coords and compute voxel coordinates order-aware
xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
if axis == 2:
  k = np.full_like(xv, slice_idx)
  voxel_coords = np.stack([xv, yv, k, np.ones_like(xv)], axis=-1)
elif axis == 1:
  k = np.full_like(xv, slice_idx)
  voxel_coords = np.stack([k, xv, yv, np.ones_like(xv)], axis=-1)
else:
  k = np.full_like(xv, slice_idx)
  voxel_coords = np.stack([yv, k, xv, np.ones_like(xv)], axis=-1)

world_coords = (affine @ voxel_coords.reshape(-1, 4).T).T.reshape(H, W, 4)
xyz = world_coords[..., :3]

# Set NaN outside mask and convert to CHW
xyz_masked = np.where(mask_slice[..., None], xyz, np.nan)
pointmap_tensor = xyz_masked.astype(np.float32).transpose(2, 0, 1)  # shape (3, H, W)
```
# Transpose to CHW and convert to float32
pointmap_tensor = xyz_masked.astype(np.float32).transpose(2, 0, 1)  # shape (3, H, W)
```

**Important**: If original `affine` is unknown or `None` (e.g., preprocessed nnUNet data), you can approximate coordinates as:
- x_norm = (i / (W - 1) - 0.5) * width_mm  (width_mm = W * pixel_spacing_x)
- y_norm = (j / (H - 1) - 0.5) * height_mm
- z_mm = slice_index * spacing_z

However, using metric `affine` is strongly recommended for physically meaningful reconstructions and accurate pose computations.

### Converting depth-only to (x,y,z)
If you only have a per-pixel depth `d` (distance along optical axis), you can convert to `x,y,z` by choosing an intrinsics matrix or using `infer_intrinsics_from_pointmap` from `sam3d_objects/pipeline/utils/pointmap.py`.

### NaN safe usage, resizing and interpolation:
- When resizing `pointmap`, preserve NaNs appropriately. The repo already contains `resize_all_to_same_size` which handles NaNs (see `img_and_mask_transforms.resize_all_to_same_size`). Use the same behavior when resizing.
- Where the mask is False, set pointmap values to `NaN`.

---

## 5) Pose handling for medical-shape-only reconstruction (important)

In the original SAM3D pipeline (and TRELLIS frameworks), the geometry model often requires a pose (translation/scale/rotation) because the trained datasets include scene placement information. For medical shape reconstruction (single-organ, centered task), you do not typically need pose information — you only need the canonical object shape.

There are three practical options you can choose depending on how much of the existing pipeline you want to keep unchanged:

1) Disable pose conditioning in the model & pipeline (recommended for simplicity)
  - Set `include_pose=False` in the model configuration (for `SparseStructureFlow`, `SLatFlowModel`, or `MOT` modules) so the network will not accept or predict pose tokens.
  - Initialize `InferencePipeline` with `pose_decoder_name='default'` to avoid pose post-processing and decoding:
    - Example: `pipeline = InferencePipeline(..., pose_decoder_name='default', ...)`
  - The pipeline will still perform shape-only generation and decoding.

2) Keep the model architecture unchanged but supply uniform default pose to the dataset
  - For minimal code changes, keep `include_pose=True` and provide a default dataset field `pose_target` for every sample. Use `ScaleShiftInvariant` or the convention that the pipeline expects (see `sam3d_objects/data/dataset/tdfy/pose_target.py`).
  - A default pose example (ScaleShiftInvariant):
    ```python
    pose_target = {
      'x_instance_scale': torch.ones(1, 1, 3) * 1.0,        # unit scale, per-dim
      'x_instance_rotation': torch.tensor([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
      'x_instance_translation': torch.zeros(1, 1, 3),        # zero translation
      'x_scene_scale': torch.ones(1, 1, 3) * 1.0,           # scene scale 1.0
      'x_scene_center': torch.zeros(1, 1, 3),               # zero center shift
      'x_translation_scale': torch.ones(1, 1, 1),           # translation scale
      'pose_target_convention': 'ScaleShiftInvariant'
    }
    ```
  - Set `pose_weight = 0.0` in flow model/training config to avoid pose supervision losses interfering.

3) Keep pose decoding but use a `default`/`zero` pose decoder at inference and training
  - Use `pose_decoder_name='default'` to remove pose decoding entirely (returns empty dict). `get_pose_decoder('default')` is available in the pipeline (`sam3d_objects/pipeline/inference_utils.py`).
  - Alternatively, use `pose_decoder_name='ZeroPredictionScaleShiftInvariant'` to decode a zero-pose prediction (identity pose) without learning the translation/rotation.

Recommendation:
- For medical single-organ reconstruction, prefer approach (1). Update your model configs to set `include_pose=False` and `pose_weight=0.0` — this avoids extra tokens, reduces attention complexity, and eliminates pose loss terms.
- If you prefer minimal changes, use approach (2): supply default `pose_target` in the dataset and set `pose_weight=0.0`.

Implementation hints:
- `include_pose` is used by `SparseStructureFlowModel` and can be toggled when instantiating or in config files.
- The pipeline `InferencePipeline` accepts `pose_decoder_name` and will apply the chosen pose decoder after `ss_generator` output; use `'default'` to suppress pose outputs.

Edge cases:
- Some parts of the repo use `pose` tokens for layout post-processing (e.g., `layout_post_optimization_utils`). If you keep the postprocessing but want no pose, set `pose_decoder_name='default'` and provide transformation-free defaults (identity/quaternion=1,0,0,0) if the downstream code expects the attributes.
- Make sure your `GT 3D` mesh/voxel coordinate system matches the shape frame expected when disabling pose: typically the canonical object coordinate system with center at origin and scale normalised in the same way that `SSIPointmapNormalizer` uses.

---

---

## 5) 3D shape conversion: SDFs, occupancy grids, and meshes

### Voxel-based SDF (recommended baseline)
- SDF is commonly used for volumetric supervision. The repo uses `compute_sdf` in `TS_nnUNet_Dataset` and expects SDF channels for each class.
- For binary GT, compute signed distance transform for the foreground vs background.

Example (compute SDF):

```python
# If compute_sdf imported from repo's utils: from utils.sdf_fn import compute_sdf
# compute_sdf should accept a binary mask (ndarray) and returns float32 signed distance field
sdf = compute_sdf(mask_volume)  # shape (D, H, W) floats
```

- Save the SDF arrays as `.npy` or incorporate into the dataset sample.

### Occupancy/voxel grid (binary)
- For decoders that produce volumetric outputs, you can use occupancy grids or binary masks.
- Resample your GT to the expected `resolution` used by the decoder; e.g., if decoder's `resolution` is 64, resample to 64^3 grid.
- To resample while preserving geometry, use `torchio.Resample` or `scipy.ndimage.zoom` with order=0 for the mask and order=1 (or 3) for intensity.

### Mesh extraction (for mesh-based decoders and evaluation)
- Use `skimage.measure.marching_cubes` to extract a mesh from binary or SDF data.
- Apply `pymeshfix` and `trimesh` to fix and simplify the mesh.

Basic steps:
```python
from skimage import measure
from pymeshfix import MeshFix

verts, faces, normals, values = measure.marching_cubes(mask_volume.astype(np.float32), level=0.5, spacing=(sz, sy, sx))

# fix mesh
meshfix = MeshFix(verts, faces)
meshfix.repair()
verts_fixed, faces_fixed = meshfix.v, meshfix.f

# optional save as .ply
import trimesh
trimesh.Trimesh(verts_fixed, faces_fixed).export('mesh.ply')
```

**Note**: If you rely on `SparseFeatures2Mesh` (used in `SLatMeshDecoder`), it expects dense volumetric features encoded as a sparse tensor with SDF and optional deformation fields.

### Gaussian splatting / Gaussians representation
- If you want the `SLatGaussianDecoder` to be supervised / trained, fit Gaussian splats to the GT mesh or pointcloud (advanced). The repo provides `representations/gaussian` utilities which will help create the Gaussian representation. However, for a first pass, use SDF or mesh ground truth instead.

---

## 6) Storing the processed dataset and dataset loader integration

You can choose the storage format for the preprocessed data. Two typical approaches are recommended:

A. Per-case volume + derived GT (SDF / mesh) plus per-slice indices (space-saver / dynamic slice creation)
- Keep the original volume and GTs with metadata and generate per-slice samples on the fly during dataset iteration (recommended if disk space / processing time is limited).
- Advantage: smaller footprint, dynamic sampling of slices (data augmentation and randomization)
- Implementation: Write a `TS_SAM3D_Dataset` (or extend `TS_nnUNet_Dataset` as a wrapper) that on access reads a volume from its `.b2nd/.pkl` or `.nii.gz` file and extracts a slice's `image`, `mask`, `pointmap` and `gt_sdf` or `gt_mesh` before returning.

B. Pre-extracted per-slice files (.npz or .pt) (fast IO for large training runs)
- For heavy training where random-access speed matters, pre-extract and store per-slice `.npz` files with keys `image`, `mask`, `pointmap`, `affine`, `slice_idx`, and a pointer to the 3D GT (e.g., a `mesh.ply` or `sdf.npy` file). This avoids resampling at runtime and minimizes CPU load.

The repo `TS_nnUNet_Dataset` returns a dictionary with expected keys; either replica this naming or write a wrapper that maps your keys to the expected ones used by the pipeline and `PreProcessor`.

### Dataset config flags (recommended defaults)
- `original_nifti_dir`: '/mnt/nas1/disk01/weidongguo/dataset/TS'  # root path to the raw NIfTI
- `cache_slices`: True  # pre-extract and save per-slice `.npz` or `.pt` for faster training
- `slice_sampling_method`: 'random_axis'  # default sampling strategy per sample
- `num_slices_per_volume`: 3  # if sampling many slices per volume for training
- `slice_cache_dir`: <workspace>/dataset/slices  # folder to persist caches
 - `include_pose`: False  # disable dataset-supplied pose_target by default for shape-only medical tasks

The loader should check `cache_slices` first and fallback to per-volume dynamic sampling if False.

**Example: structure for `.npz`**:
```
case_000_slice_012.npz
  - image: float32, shape (3, H, W)
  - mask: uint8, shape (H, W)
  - pointmap: float32, shape (3, H, W)
  - affine: float32, shape (4, 4)
  - slice_idx: int
  - gt_sdf_path: 'case_000_sdf.npy'  # or save inside the NPZ
```

Add optional metadata fields:
- `cache_slices`: boolean (True if this sample was created by the offline caching step)
- `slice_sampling_method`: the sampling strategy used to extract this slice (e.g. `random_axis`, `uniform`, `central`, `all`)

Note: For multi-class masks (0..K), compute `mask_sdf` for each class separately, e.g.:
```python
mask_np = volume_mask  # shape (D,H,W), integer labels 0..K
sdf_channels = []
for c in range(1, K+1):
    sdf_c = compute_sdf((mask_np == c).astype(np.uint8))
    sdf_channels.append(sdf_c)
mask_sdf = np.stack(sdf_channels, axis=0)  # shape (K, D, H, W)
```

When training, you may either store `mask_sdf` per-case and link to it via `gt_sdf_path`, or pre-extract per-slice SDF patches for faster IO.

**`TS_SAM3D_Dataset` loader tips**:
- If using `.npz` slices, simply load the file and return:
```
return {
    'image': torch.tensor(image),
    'mask': torch.tensor(mask),
    'pointmap': torch.tensor(pointmap),
    'gt_sdf': torch.load(gt_sdf_path),
}
```
- If you create on-the-fly from volumes, reuse the same pipeline: 
  - read volume and mask
  - sample a slice index `z` and produce `image`, `mask_slice`, compute `pointmap` as above
  - return tensors with shapes consistent with pipeline

---

## 7) Use `SSIPointmapNormalizer` and `PreProcessor` configs (repo config) to match normalizations

- In your training config, the key flags are: `normalize_pointmap`, `pointmap_normalizer`, and `pointmap_transform`. Set `normalize_pointmap` to `True` and use `SSIPointmapNormalizer` with `ObjectCentricSSI` or `ObjectApparentSizeSSI` depending on your preference.
- The repo uses `NormalizedDisparitySpaceSSI` sometimes; choose that if your input data are virtually in disparity space.
- `PreProcessor._process_image_mask_pointmap_mess` takes care of normalization; stick to its transforms and set `pointmap_transform` and `img_transform` accordingly.

---

## 8) Resampling / isotropic spacing and resolution choices

- If your MRI volumes are anisotropic (e.g., 1.2mm × 1.2mm × 6.0mm), you should resample to isotropic spacing or a consistent `resolution` expected by training. Common choices: 1 mm or 0.5 mm isotropic depending on memory and expected fidelity.
- For SLat-based decoders, set the `resolution` to match: e.g., `resolution=64` means a 64^3 latent grid; the `MeshExtractor` uses `res=resolution * 4` internally which yields a higher resolution for final meshes.
- Use `torchio.Resample` or `scipy.ndimage.zoom` to resample volumes. For segmentation masks, use nearest neighbor interpolation (order=0). For SDF computation, do interpolation on the binary volume.

### Adaptive resolution per modality (recommended)

- Medical images vary by modality and scanner. Choose a target isotropic spacing intelligently using the raw volume spacing and modality:
  - CT/MRI (diagnostic resolution): use target spacing around 1.0 mm isotropic (default). When the in-plane resolution is much finer (e.g., <= 0.5 mm), you may select 0.5 mm if GPU/memory allows.
  - Ultrasound / low-res / compressed volumes: use target spacing between 1.0 and 2.0 mm (choose the least upsampling to avoid smoothing artifacts).
  - High-resolution scans (micro-CT): use 0.2 - 0.5 mm if needed for detail.

- Example logic (pseudo):
  1) Compute the original volume spacings (sx, sy, sz) from NIfTI header.
  2) If modality == CT or modality == MRI: target = clip(min(sx, sy, 1.0), 0.5, 2.0)
  3) If modality == Ultrasound: target = clip(min(sx, sy, sz, 1.5), 1.0, 2.5)
  4) For most usecases, a safe default is `1.0 mm`.

This logic lets you preserve in-plane detail while not needlessly up-sampling the through-plane axis.

### Augmentations (2D slice-level and joint 3D/2D)

- If `PreProcessor` or the dataset `img_and_mask_transforms` do not contain the augmentations needed, add per-slice augment transforms in the dataset loader as a second stage of transforms. Typical augmentations include:
  - Geometric transforms: small rotation (±10°), scale (0.9-1.1), random crop/pad, horizontal/vertical flips; these must be applied identically to `image`, `mask`, and `pointmap`.
  - Spatial transforms: for 3D volumes used to sample more slices, allow a small elastic deformation (torchio.RandomElasticDeformation) if shape anatomical plausibility remains.
  - Intensity transforms: additive Gaussian noise, per-volume z-score changes, contrast/brightness jitter.

- Pointmap handling: all geometric transforms must be applied to the pointmap coordinates. When using affine-based pointmaps (world coordinates), instead of resampling the 3D volume after transform, you can apply the 2D transform to the `pointmap` coordinates (rotate/translate/scale in world space) to remain accurate. Always set `NaN` where the mask is background.

- Implementation hint: create a small wrapper in the dataset or PreProcessor that accepts the 2D slice image, mask and pointmap and applies PyTorch-friendly transforms (or TorchIO 2D transforms) to the image/mask, and applies an equivalent rigid transform to the pointmap.

Example geometric transform on `pointmap` (rotate around center):

```python
# Given pointmap (3, H, W) float32, mask (H, W) bool, and a 2D rotation matrix R (3x3 in homogeneous coords)
# Convert (x,y,z) coords to homogeneous, apply rotation/translation, then project back and set NaN where mask==0
xyz = pointmap.transpose(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
xyz_h = (R @ xyz_h.T).T  # transform
xyz_trans = xyz_h[:, :3].reshape(H, W, 3).transpose(2, 0, 1)
xyz_trans[..., ~mask] = np.nan
```

Make sure transforms preserve `NaN` values for background pixels and are performed in a consistent, invertible way for debugging/visualization.

---

## 9) Example end-to-end snippet (Python) to generate (image, mask, pointmap, gt_sdf)

```python
# Dependencies: nibabel, numpy, torch, skimage, pymeshfix (optional), nnUNet preprocessed loader if used
from skimage import measure
import nibabel as nib
import numpy as np
import torch

# 1. Load NIfTI files (original volume and segmentation):
nii_img = nib.load('path/to/img.nii.gz')
nii_mask = nib.load('path/to/mask.nii.gz')
img = nii_img.get_fdata()
mask = nii_mask.get_fdata().astype(np.uint8)
affine = nii_img.affine

# 2. Decide slice index or indices:
z = img.shape[0] // 2  # central slice (assuming D,H,W order) - check axis ordering

# 3. Extract 2D slice and 3-channel image
img_slice = img[z]  # (H, W)
mask_slice = mask[z]  # (H, W)

# Convert MRI single-channel to 3-channel float (replicate)
image_3ch = np.stack([img_slice, img_slice, img_slice], axis=0).astype(np.float32)

# 4. Compute world coordinates per pixel and set NaN outside mask
H, W = img_slice.shape
xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
voxel_coords = np.stack([xv, yv, np.full_like(xv, z), np.ones_like(xv)], axis=-1)
world_coords = (affine @ voxel_coords.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]

# Set NaN outside mask
xyz_masked = np.where(mask_slice[..., None], world_coords, np.nan)
pointmap = xyz_masked.transpose(2, 0, 1).astype(np.float32)  # shape (3, H, W)

# 5. Compute 3D GT SDF (from 3D mask)
from utils.sdf_fn import compute_sdf
sdf_3d = compute_sdf(mask)  # (D, H, W) float
# Save or reduce to the desired grid resolution if required

# 6. Save per-slice NPZ / PT file
np.savez_compressed('case_000_slice_{}.npz'.format(z), image=image_3ch, mask=mask_slice, pointmap=pointmap, affine=affine, gt_sdf=sdf_3d)
```

Notes:
- If `affine` is missing or not trustworthy, use voxel coordinates scaled by spacing to approximate world coordinates.
- If your orientation is nonstandard (NIfTI axes), your affine to world is critical to produce correct pointmap; double-check orientation with `nibabel.aff2axcodes(affine)`.

---

## 10) Integration with pipeline & expected datastructures

### A. PreProcessor defaults
- `PreProcessor._process_image_mask_pointmap_mess` expects `rgb_image` and `rgb_image_mask` as images and masks in channel-first tensors (C,H,W, where C==3). It also normalizes `pointmap` using `pointmap_normalizer` if present; this returns `pointmap`, `pointmap_scale`, and `pointmap_shift`.
- `PointPatchEmbed` requires resizing to `PointPatchEmbed.input_size` (e.g., 256). Your `resize_and_make_square()` and other transforms should enforce those sizes and maintain pointmap shape.

### B. Example integration points
- To feed slice-based inputs to the `InferencePipelinePointMap`, you can use `pipeline.compute_pointmap` or pass precomputed `pointmap` to `pipeline.run(...)`.
- For training, create a dataset that returns the desired dictionaries and use the repo's training utilities (`flow_matching`, training harness) to configure loss and optimizer.

Recommended dataset preprocessing flow (explicit):
1. Read raw NIfTI and load `img`, `mask`, `affine` (resampled to isotropic spacing if needed)
2. Pick slice with `slice_sampling_method` (e.g., `random_axis`), compute `pointmap` from `affine` and mask
3. Call `PreProcessor._process_image_mask_pointmap_mess(rgb_image, rgb_mask, pointmap)` to apply joint transforms and normalization
4. Return the processed keys: `image`, `mask`, `pointmap`, `pointmap_scale`, `pointmap_shift`, `affine`, and `gt_sdf` or pointer to `gt_sdf`

---

## 11) Choosing the training target representation & metrics

- For mesh-based decoders (`SLatMeshDecoder`): use a high-resolution mesh (extracted via marching cubes) for evaluation. For training, you can use a volumetric SDF or TSDF as which the model can reconstruct and later extract mesh with `SparseFeatures2Mesh`.
  - Prioritize mesh-based decoder (`SLatMeshDecoder`) for clinical surface quality. Use volumetric SDF to supervise the decoder and compute final meshes via marching cubes for evaluation.
- For gaussian-based decoders (`SLatGaussianDecoder`): fitting gaussian splats to meshes is advanced. For initial fine-tuning, use SDF/mesh supervision and consider gaussian-specific fitting later.
- Metrics: IoU (voxel overlap), Chamfer Distance, normalized surface distance, and post-fit IoU after ICP are good choices. The repo already includes functions like `compute_iou` and `layout_post_optimization_utils` for post-processing.

### Recommended metric libraries and examples
Instead of re-implementing these metrics, consider the following libraries:

- `pytorch3d` for chamfer distance and mesh/pointcloud utilities: https://github.com/facebookresearch/pytorch3d
- `surface-distance` (medpy-family) for Hausdorff (HD95) and average symmetric surface distance (ASSD)
- `scipy.spatial.distance.directed_hausdorff` for a baseline directed Hausdorff function
- `torchmetrics` for Dice and classic segmentation metrics in PyTorch

Example (PyTorch3D Chamfer usage):
```python
from pytorch3d.loss import chamfer_distance
loss_chamfer, _ = chamfer_distance(pred_points, gt_points)
```

Example (HD95 using `surface-distance`):
```python
from surface_distance import compute_surface_distances, compute_robust_hausdorff
distances = compute_surface_distances(gt_mask.astype(np.bool), pred_mask.astype(np.bool), spacing_mm)
hd95 = compute_robust_hausdorff(distances, 95)
```

---

## 12) Tips, pitfalls, and recommendations

- Affine metadata: Ensure `affine` is accessible in your dataset preprocessing stage. NNUNet preprocessed volumes may have lost affine; re-associate it from original NIfTI. The pipeline uses identity `affine` as placeholder in `TS_nnUNet_Dataset`.

- NaN handling: Always set invalid pointmap entries to NaN (or a constant invalid token), to ensure `PointPatchEmbed` marks invalid points and uses `invalid_xyz_token`.

- Down/resampling & anisotropy: If you resample volumes, record new `affine` and adjust spacing accordingly; mismatched spacing will cause inaccuracies in shape scale or pose.

- SDF & mesh resolution: Make sure ground-truth shapes are resampled to the same `resolution` used by decoders to avoid mismatch.

- Data augmentation: Since MR is modality-specific, use `torchio` transforms to keep sensible augmentation (intensity, elastic deformation, small rotations, and cropping). Example `GetTrainTransforms` in `ts_nnunet_dataloader.py` is helpful.

- Use `SSIPointmapNormalizer`/`ObjectCentricSSI`: For the pointmaps, prefer `ObjectCentricSSI` if your data provides direct object coordinates or `NormalizedDisparitySpaceSSI` if you want to use disparity remapping.

- If you have decimal class labels (segmentation multi-class), compute `mask_sdf` per-class and possibly compute one-hot category masks (see how `TS_nnUNet_Dataset` computes `mask_sdf` and `categorical_mask`).

---

## 13) Recommendation for an initial, minimal pipeline (step-by-step):

1. For each case (original `.nii.gz` + `.nii.gz` label):
   a. Optionally resample to isotropic spacing, e.g., 1mm^3 using `torchio.Resample`.
   b. Compute `sdf_3d = compute_sdf(mask_volume)` and store it.
   c. Extract a set of slices (e.g., 3 slices per volume) that contain the organ (mask occupancy threshold) in axial plane.
   d. For each slice: compute `image` (3ch), `mask`, `pointmap` (from `affine`) and store as `.npz` or .pt file.
   e. Generate a pointer to the 3D GT (e.g., `gt_sdf.npy` or `gt_mesh.ply`) in the saved per-slice file.

2. Implement a `TS_SAM3D_Dataset` wrapper that returns a dict of keys required by the repo preprocessor: `image`, `mask`, `pointmap`, `affine`, `gt_sdf`.

3. Configure `PreProcessor` with `pointmap_normalizer=SSIPointmapNormalizer()` and `normalize_pointmap=True`. Configure `img_transform` to resize to the expected `PointPatchEmbed.input_size`.

4. Configure your SAM3D training harness to use the dataset loader and to compute the required supervision losses (SDF MSE, IoU, Chamfer). Optionally use the `flow_matching` wrapper for slat generator optimization that is already present in the repo.

5. Monitor training for NaNs; use `grad_clip` if necessary.

---

## 14) Example dataset loader snippet using nnUNet preprocessed data (`TS_nnUNet_Dataset`)

If you prefer to reuse the included `TS_nnUNet_Dataset` logic (fast approach):
- The `TS_nnUNet_Dataset` already loads image and mask and computes `mask_sdf`. It returns `img` (C,D,H,W) and `mask_sdf`. 
- Extend it to output per-slice items: implement a loader that samples slices and computes `pointmap` with `affine` (reconstructed from original NIfTI or approximated by known spacing).

**Important detail**: `TS_nnUNet_Dataset.__getitem__` uses identity affine as placeholder. This is not suitable for accurate metric pointmaps — reattach `affine` from original NIfTI when available.

---

## 15) Additional references & repo integrations

- `PreProcessor` (_sam3d_objects/data/dataset/tdfy/preprocessor.py_) handles joint transforms and pointmap normalization.
- `SSIPointmapNormalizer` and `NormalizedDisparitySpaceSSI` are in `img_and_mask_transforms.py` (detailed logic on pointmap normalization).
- `PointPatchEmbed` (in `sam3d_objects/model/backbone/dit/embedder/pointmap.py`) consumes (B, 3, H, W) with NaNs and generates the per-window point tokens.
- `InferencePipelinePointMap` (`sam3d_objects/pipeline/inference_pipeline_pointmap.py`) holds the inference-time flow and pipe that is used for running `pointmap` conditioned generation — very helpful to debug expected keys and types.
- SLat decoders: `SLatGaussianDecoder` and `SLatMeshDecoder` accept volumetric or gaussian outputs; refer to `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae`.

---

## 16) TRELLIS and other third-party pipelines

- The repository does not currently include TRELLIS-based preprocessing modules or references. If you have a separate TRELLIS pipeline that you use for 3D preprocessing (meshing / gaussian fitting), ensure the outputs are standardized (world coords, consistent spacing) before integrating them.
- Standard tooling for mesh and voxelization: `skimage.measure.marching_cubes`, `pymeshfix`, `trimesh` for mesh repair/analysis, `torchio` for resampling.

---

## 17) Final checklist before starting training

- [ ] Each sample has `image`, `mask`, `pointmap` (NaN for background), `affine`, and a pointer to `gt_3d` (mesh or SDF) available.
- [ ] `normalize_pointmap=True` with `SSIPointmapNormalizer` configured.
- [ ] `PointPatchEmbed.input_size` matches the size you produce (e.g., 256 or 512); pointmap resize is handled by transforms.
- [ ] GT SDF/mesh/voxel are resampled to the same resolution expected by the SLAT decoders.
- [ ] `affine` and spacing are correct for metric conversions.
- [ ] Augmentation strategies (TorchIO transforms) applied for robust generalization.
- [ ] Pre-extract or compute `gt_sdf` saved; test `TS_SAM3D_Dataset` read path and sample shapes with the pipeline (`preprocess_image`, `sample_slat`, `decode`)

---

## 18) Quick troubleshooting & debugging tips

- Use a synthetic small dataset to run through the full pipeline (preprocess → InferencePipelinePointMap.preprocess_image → sample_slat → decode). This helps find shape mismatches early.
- Visualize `pointmap`, `mask` and `image` overlays with `SceneVisualizer` (see `sam3d_objects/utils/visualization/scene_visualizer.py`).
- If pointmaps do not align with masks, check `affine` or axis orientation (NIfTI `affine` can indicate axis permutation; use `nibabel.aff2axcodes` to confirm ordering).
- If training loss diverges, verify the SDF/voxel GT is correct (visualize slices or re-construct mesh and compare to original label), and ensure `pointmap` normalization is correct (SSIPointmapNormalizer outputs scale and shift).

---

## 19) Final notes and next steps

- If you'd like, I can generate a `TS_SAM3D_Dataset` class in `sam3d_objects/data/dataset/ts/` that extends `TS_nnUNet_Dataset` and returns per-slice dictionaries matching `PreProcessor` expectations and paired `gt_sdf` pointers. 
- I can also prepare example scripts for: 
  - Pointmap extraction from raw NIfTI with the proper `affine` handling,
  - A minimal dataset wrapper to hook into the repo's training harness,
  - A utility to compute GT SDF and save meshes using `marching_cubes`.
    - `scripts/reprocess_ts_nifti.py` - reprocess raw NIfTI and build per-case SDF and per-slice cache (`.npz`) files.
    - `sam3d_objects/data/dataset/ts/ts_sam3d_dataloader.py` - the per-slice dataset implementation that returns samples compatible with `PreProcessor`.
    - `scripts/test_ts_sam3d_dataset.py` - small test harness for the dataset and cache.

  Example commands:
  ``bash
  python scripts/reprocess_ts_nifti.py --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS --out_dir ./dataset/ts_processed --axes 0,1,2 --spacing 1.0 --classes 5
  python scripts/test_ts_sam3d_dataset.py --nifti_dir ./dataset/ts_processed --cache_dir ./dataset/ts_processed
  ```

If you confirm, I'll implement those example scripts and dataset code next.

---

Appendix: Short Code snippets and references (already included above) and repo references:
- `TS_nnUNet_Dataset`:
  - File: `sam3d_objects/data/dataset/ts/ts_nnunet_dataloader.py`
- `PreProcessor` & `SSIPointmapNormalizer`:
  - Files: `sam3d_objects/data/dataset/tdfy/preprocessor.py`, `sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py`
- `PointPatchEmbed` (pointmap embedder):
  - File: `sam3d_objects/model/backbone/dit/embedder/pointmap.py`
- `InferencePipelinePointMap`:
  - File: `sam3d_objects/pipeline/inference_pipeline_pointmap.py`
- Slat decoders: `SLatMeshDecoder`, `SLatGaussianDecoder`:
  - Files: `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_mesh.py`, `decoder_gs.py`

