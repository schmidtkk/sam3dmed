# SAM3D Medical Fine-tuning — Minimal Implementation Plan

Purpose: Fine-tune SAM3D-Object for medical imaging (e.g., MR/CT/Ultrasound), enabling reconstruction of full 3D organ shapes from a single 2D slice click.

---

## Overview & Scope
- Task: Given a 2D user click (single slice), produce a 3D reconstruction of the organ (mesh / gaussian / occupancy). Use LoRA for parameter-efficient fine-tuning.
- Minimal approach: fine-tune the structured latent (slat) generator/decoder, conditioning on single-slice pointmap or segmentation mask. Prioritize mesh-based decoding (SLatMeshDecoder) for higher fidelity surface reconstructions where clinically relevant. For medical tasks where object placement in a scene is irrelevant, prefer shape-only training without pose conditioning (see `include_pose` and `pose_decoder_name` settings below). 
 - Minimal approach: fine-tune the structured latent (slat) generator/decoder, conditioning on single-slice pointmap or segmentation mask. Mesh-only initial training: fine-tune `SLatMeshDecoder` with LoRA adapters on attention/FFN modules and supervise with SDF and mesh losses. Gaussian representations are optional later experiments only. For medical tasks where object placement in a scene is irrelevant, prefer shape-only training without pose conditioning (see `include_pose` and `pose_decoder_name` settings below).

---

## Short Plan (High-level steps)
1. Prepare medical dataset → one sample per 2D slice (random axis sampling among x,y,z) + 2D click / mask + 3D GT shapes/voxels/meshes
 2. Implement dataset class & PreProcessor config to return keys expected by pipeline (`image`, `mask` or `pointmap`, `gt_3d`). If your pipeline configuration disables pose tokens (recommended), then `pose_target` is not required. If the pipeline expects pose tokens, provide a default `pose_target` (see Pose handling).
3. Integrate LoRA adapters into model (Transformer Q/K/V/out and FeedForward) for low-cost fine-tuning
4. Fine-tune `slat_generator` (and optionally `ss_generator`), freeze base weights, only train LoRA and decoder heads
5. Validate on held-out volumes: 2D mask accuracy, Chamfer/IoU on 3D outputs

---

## Data Preprocessing Steps (Detailed)
1. Volume -> slices (recommended: re-preprocess raw NIfTI and deprecate nnUNet preprocessed)
   - The raw NIfTI dataset location is: `/mnt/nas1/disk01/weidongguo/dataset/TS` — **use these raw files for reprocessing** to preserve `affine` and spacing.
   - Convert DICOM/NIfTI to standardized format (.nii / .npy), resample to isotropic voxel size (e.g., 1mm^3) using `torchio.Resample` to avoid anisotropy issues.
   - Extract slices along a random axis per sample (choose from x=0, y=1, z=2). Implement `slice_sampling_method=random_axis` and produce exactly one sample per 2D slice per sample. This is the core dataset behavior for SAM3D fine-tuning.

2. Create single-slice training examples (one sample per 2D slice)
   - For each slice, supply
     - `image`: 2D grayscale (H,W) mapped to 3 channels (or single channel depending on model): ensure values normalized approximately like training images (see `img_and_mask_transforms` for scaling parameters)
     - `mask`: binary segmentation if you have segmentation
     - `pointmap` or `clicked point`: encode user click as a small sparse depth map or 2D coordinate -> convert to `pointmap` if 3D depth available
     - `gt_3d`: ground-truth 3D shape (mesh, occupancy voxel grid, or point cloud). Convert to mesh or occupancy grid used by decoder (`slat_decoder_mesh`, `slat_decoder_gs`).
   - `pose_target`: Optional. We recommend `include_pose=False` for medical tasks. If you prefer to keep the model architecture unchanged, provide a uniform default pose (identity/zero translation/unit scale) and set `pose_weight=0.0` during training.
   - Augmentations & PreProcessor: If `PreProcessor` does not provide sufficient augmentations, the dataset loader should apply an additional per-slice augmentation wrapper. These augmentations must be applied consistently to `image`, `mask`, and `pointmap`: rotate/scale/translate pointmaps in world coordinates, perform intensity jitter on images, and use nearest neighbor for masks. Keep geometric augmentation conservative (small angles and scales) for medical contexts.

3. Preprocessing details (observed in repo files)
   - Use `PreProcessor` (sam3d_objects/data/dataset/tdfy/preprocessor.py) to handle joint transforms, normalization, and pointmap normalization.
   - For CT: apply windowing; clip HU to (-1000, 2000) or clinical standard ranges.
   - For MRI: intensity standardization per volume (z-score normalization) before passing to PreProcessor.
   - For Ultrasound: consider speckle filtering and intensity normalization.
   - Keep masks consistent: Binary, same size as `image`. Use `img_and_mask_transforms` to pad/resize.
   - Compute `pointmap`: For a clicked pixel, set (x, y, z) if 3D depth available or set z to slice's index/coordinate. Use `PointPatchEmbed`'s expected format: `(B, 3, H, W)`.
      - Compute `pointmap` using the raw NIfTI `affine`: transform voxel indices (i,j,k) based on the slice axis into world coordinates (mm) and set background to NaN. Ensure `pointmap` is computed before calling `PreProcessor` transforms.

   ### Pose handling choices (explicit)
   When preparing your dataset and run configs, select one of these options: 

   1) Best practice — disable pose conditioning:
      - Configure generator/flow models by setting `include_pose=False` in the corresponding configs for `SparseStructureFlow`, `SLatFlowModel`, `MOT` model classes or wrapper configs. This removes the pose token from latents and avoids pose training.
      - Initialize `InferencePipeline` with `pose_decoder_name='default'` to avoid pose decoding: `pipeline = InferencePipeline(..., pose_decoder_name='default', ...)` 
      - Set `pose_weight = 0` or remove pose losses in training.

   2) Minimal changes — provide default (identity) poses in dataset:
      - Provide `pose_target` with a canonical identity pose if the model was trained to expect a pose token and you prefer not to change model config.
      - Example `pose_target` (ScaleShiftInvariant):
        ```python
        pose_target = {
            'x_instance_scale': torch.ones(1, 1, 3) * 1.0,
            'x_instance_rotation': torch.tensor([1.0, 0.0, 0.0, 0.0]),
            'x_instance_translation': torch.zeros(1, 1, 3),
            'x_scene_scale': torch.ones(1, 1, 3) * 1.0,
            'x_scene_center': torch.zeros(1, 1, 3),
            'x_translation_scale': torch.ones(1, 1, 1),
            'pose_target_convention': 'ScaleShiftInvariant'
        }
        ```
      - For training set `pose_weight` to 0.0 during optimization to suppress pose loss while keeping the model's pose branch inactive.

   3) Keep the model and training unchanged — use a zero-pose decoder during inference to suppress pose output:
      - Use `pose_decoder_name='ZeroPredictionScaleShiftInvariant'` or `'default'` in `InferencePipeline` — this will keep the training pipeline unmodified but not produce learned poses at inference time.

   Recommendation: For medical shape-only tasks, option (1) is recommended for a simpler and more stable training process.

4. Important: pointmap NaNs handling
   - The repo handles NaNs in pointmaps (see `img_and_mask_transforms.py` and `PointPatchEmbed`). Ensure background is NaN and valid points are finite.
    - Ensure that `resize_all_to_same_size` is used to perform NaN-safe interpolation of pointmaps; do not in-place resize with direct interpolation without the NaN-mask technique.

---

## Fine-Tuning Implementation Details
1. Target layers & LoRA placement
   - Inject LoRA adapters into these modules:
     - `MOTMultiHeadSelfAttention`/`MultiHeadAttention`: patch `to_qkv`, `to_q`, `to_kv`, `to_out` linear layers and feed-forward's linear matrices.
     - Pointmap embedder `point_proj` and MLP layers in `FeedForward` modules as needed.
   - If you prefer a short path, rely on an external PEFT library (`peft` / `loralib`) with an adapter/wrapper to hook into PyTorch modules — otherwise add a small `LoRALinear` implementation in `model/backbone/utils` and modify module definitions.

2. Freeze strategy
   - Freeze full model parameters except LoRA adapters and any new output heads (e.g., `out_layer` or decoder MLPs). This is a low-cost approach that improves stability in small medical datasets.

3. Training details
   - Optimizer: AdamW or Adam for LoRA params. Example: lr=1e-3 (LoRA), lr=1e-5 for any learned final heads if also fine-tuning them.
   - Batch size: small (1-8) based on GPU memory; gradient accumulation is recommended for stability.
   - Losses:
     - Reconstruction: volumetric occupancy/MSE, Chamfer distance over meshes, voxel IoU
     - Pose loss: MSE for translations/scale/quaternions or use `layout_post_optimization_utils.compute_loss` for mask matching
     - Optional 2D supervision: MSE or BCE for segmentation mask reconstruction
     - Optionally joint losses from `flow_matching` modules — repo has flow matching generator for training (`flow_matching/model.py`) and `loss` wrapper

4. Training schedule
   - Small-scale: 1-5 epochs on a curated medical training set; evaluate frequently using validation 3D metrics.
   - A recommended curriculum: start with 2D mask supervision + 3D coarse occupancy, then introduce dense 3D mesh losses.

---

## Needed Preprocessing (explicit list)
- Convert medical volumes to isotropic voxel grids, registered to a common coordinate system if possible
- Create 2D slice images with consistent resolution (recommended 256x256 or 512x512) and pixel normalization
 - Create 2D slice images with consistent resolution (recommended 256x256 or 512x512) and pixel normalization. We recommend extracting a single slice per sample using random axis sampling, and optionally saving them to disk with a caching option.
- Generate `mask` / `pointmap` for training: for pointmap, attach z coordinate (depth or slice index), background as NaN
- Produce GT meshes/voxel grids with consistent scale and coordinate conventions; ensure `pose_target` uses expected `pose_target_convention` (default `ScaleShiftInvariant` / `DisparitySpace` — see `pose_target.py`)
 - Produce GT meshes/voxel grids with consistent scale and coordinate conventions; from the multi-class label NIfTI (0=bg, 1..5 foreground classes), compute per-class SDF channels via `compute_sdf` on the per-class binary mask. Store per-class SDF as `(C, D, H, W)` or a pointer file.
- Ensure `PreProcessor` is updated with `pointmap_normalizer` and `img_transform` that match medical signal statistics
- Optional: add per-modality augmentations (rotation, small scale, intensity jitter). For medical, keep geometric augmentations conservative.

---

## Implementation Steps (Actionable & Minimal)
1. Setup
   - Follow `doc/setup.md` to install dependencies & download checkpoints
   - Create an environment (conda or venv) and install `requirements*.txt`

2. Minimal dataset code & reprocessing
   - Implement a preprocessing script `scripts/reprocess_ts_nifti.py` that:
      - Reads raw NIfTI files from `/mnt/nas1/disk01/weidongguo/dataset/TS`
      - Resamples to isotropic voxel spacing and canonical orientation
      - Computes per-class SDF for classes 1..K and saves as `case_x_sdf.npy` (C x D x H x W)
      - (Optional) Extracts meshes for each class and saves `case_x.ply`
      - Extracts per-slice `.npz` files with keys `image`, `mask`, `pointmap`, `affine`, `slice_idx`, `gt_sdf_path` using random axis selection and caching.

   - Implement `sam3d_objects/data/dataset/ts/TS_SAM3D_Dataset` (or extend `TS_nnUNet_Dataset`) that:
      - Returns exactly one sample per slice (per-slice mode)
      - Allows configuration of `slice_sampling_method` with `random_axis` (default), `uniform`, `central`, or `all`
      - Accepts `cache_slices=True/False` to use pre-extracted `.npz` files for faster loading
      - Computes `pointmap` using the `affine` and attaches `pointmap_scale` and `pointmap_shift` by calling `PreProcessor._process_image_mask_pointmap_mess`
      - Returns keys: `image`, `mask`, `pointmap`, `pointmap_scale`, `pointmap_shift`, `affine`, `mask_sdf` (per-class channels), `gt_mesh_path` (optional)

3. LoRA injection
   - Implement a `loralib`-style wrapper or `LoRALinear` to wrap linear layers in `MOTMultiHeadSelfAttention` and `MultiHeadAttention`.
   - Provide a small config to enable LoRA with low rank r (e.g., r=4
   - Implement a lookup util to register LoRA on all `nn.Linear` modules in the target transformer blocks.

4. Train harness
   - Create a `scripts/finetune_medical.py` entrypoint using the repo's training utilities: initialize pipeline with `instantiate_and_load_from_pretrained()` in `inference_pipeline.py`, replace dataset loader with the new medical dataset, apply optimizer only to LoRA params and heads, then call training loops using the `flow_matching` training wrapper if desired.

5. Quick debugging & validation (enhanced for TS dataset)
   - Reprocess one or two volumes and run the `scripts/reprocess_ts_nifti.py` to extract per-slice `.npz`.
   - Load `TS_SAM3D_Dataset` with `cache_slices=True`, iterate for a few samples and ensure the following:
       - `image` shapes `C,H,W` are correct and normalized
       - `mask` is 2D binary with one channel
       - `pointmap` shape `3,H,W` contains finite values where `mask==1` and `NaN` elsewhere
       - `mask_sdf` shape equals `C_gt, D, H, W` or pointer path exists
       - The dataset can run through `InferncePipelinePointMap.preprocess_image` to produce the expected `PreProcessor` outputs
   - Run a single `sample_slat` and `slat_decoder_mesh` to verify end-to-end shapes.

   ## Repo & pipeline configuration changes (shape-only setup)
   To make the repo operate in a shape-only (no-pose) configuration for medical tasks, update the generator/sparse-structure flow configs and the pipeline to remove pose tokens and decoding:

   1) Set `include_pose=false` in the model config (e.g., `sparse_structure_flow` / `slat_generator`); this removes the extra pose token from the latent stream and reduces attention overhead.

   2) Set `pose_decoder_name='default'` in the `InferencePipeline` instantiation to avoid pose decoding at inference time:
   ```python
   pipeline = InferencePipeline(..., pose_decoder_name='default', ...)
   ```

   3) Set `pose_weight=0.0` in training configs to ensure any downstream pose losses are disabled.

   After these changes, the model will operate in shape-only mode and will only predict canonical object shape in its local coordinate frame.


---

## Practical Advice & Pitfalls (from repo review)
- The repo expects `pointmap` with NaN for background and normalized shifting/scale: ensure that `pointmap` values are normalized via `SSIPointmapNormalizer` and `PreProcessor`.
- The `PoseTarget` conventions must be matched (e.g., `ScaleShiftInvariant`). Use `PoseTargetConverter` to maintain consistent pose encoding.
- Token lengths and position embeddings: when adding a new condition/modal, ensure `latent_mapping` and `pos_emb` lengths align for merge/split in `mot_sparse_structure_flow` (`merge_latent_share_transformer` / `split_latent_share_transformer`).
- Watch for memory constraints: combined tokens lead to large attention. Consider `latent_share_transformer` to share transformer blocks or chunk attention.
- Training stability: the repo uses diffusion/flow matching and may require LR tuning, gradient clipping, or smaller batch sizes; monitor for NaNs or exploding loss.
- Checkpoints and loading: use `inference_pipeline.instantiate_and_load_from_pretrained()` which respects the repo's IO utilities; for LoRA you can save separate adapter states.
- NaNs in pointmap/point map resizing: see `img_and_mask_transforms` for special handling. Avoid `torch.nn.functional.resize` applied to NaN values without masking.

---

## Evaluation / Validation Checklist
- 2D: Mask IoU, pixel-wise MSE for mask predictions
- 3D: Chamfer distance, voxel IoU, mesh-to-mesh distance if ground truth mesh is available
- Optional metrics to add: 3D Dice (voxel-wise), Hausdorff distance / HD95 (surface-based), and per-voxel boundary metrics (ASSD / MSD)
Instead of implementing complex metrics from scratch, rely on stable, tested libraries:

- PyTorch3D (Facebook Research): Chamfer distance and point/mesh utilities. Good for batched pointcloud/mesh comparisons.
- SciPy / scikit-image: baseline functions for voxel-based metrics and directed Hausdorff distance.
- surface-distance (medpy-family or `surface-distance` package): HD95, ASSD computations.
- torchmetrics: 2D/3D Dice and standard segmentation metrics if you want torch-native logging.

Example: Using PyTorch3D to compute Chamfer distance (pip install pytorch3d)
```python
from pytorch3d.loss import chamfer_distance
# p1 and p2 are pointclouds (B, N1, 3) and (B, N2, 3)
loss_chamfer, matching = chamfer_distance(p1, p2)
```

Example: Using `surface-distance` to compute HD95
```python
from surface_distance import compute_surface_distances, compute_robust_hausdorff
distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm)
hd95 = compute_robust_hausdorff(distances, 95)
```
- Pose/Scale: MSE on `pose_target` components if supervised
- Visual: render comparison using `plot_tdfy_scene` / `layout_post_optimization_utils.apply_transform` for alignment

---

## Example Commands (Hints)
- Setup: (run from repo root)
```bash
# create env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

- Quick test inference (single-slice forward):
```bash
python demo.py --config checkpoints/<TAG>/pipeline.yaml --image <slice_image> --mask <mask> --pointmap <pointmap>
```

- Fine-tune (pseudocode): use `scripts/finetune_medical.py` (TBD) with the dataset and LoRA on
```bash
python scripts/finetune_medical.py --config checkpoints/<TAG>/pipeline.yaml --dataset medical_train --lr 1e-3 --lora_rank 4
```

# Reprocess raw NIfTI and build caches:
```bash
python scripts/reprocess_ts_nifti.py --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS --out_dir ./dataset/ts_processed --axes 0,1,2 --spacing 1.0 --classes 5
```

# Quick test dataset loader:
```bash
python scripts/test_ts_sam3d_dataset.py --nifti_dir ./dataset/ts_processed --cache_dir ./dataset/ts_processed
```

---

## Next steps (Optional)
- I can generate the PR/prototype code for the minimal components:
  - a small `medical` dataset `sam3d_objects/data/dataset/medical/`,
  - `scripts/finetune_medical.py`, and
  - a `LoRALinear`/adapter implementation hooking into `MultiHeadAttention`.

Please tell me which optional items you want me to implement next.

---

Generated by repository scan and the `demand.md` file; if you want a runnable demo patch (dataset + training script + small run/test harness), request "generate patch for medical finetuning" and I will create it.