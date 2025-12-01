```shell
# SAM3D Medical Fine-Tuning Pipeline

Complete step-by-step guide for medical image segmentation with SAM3D + LoRA.

## Dataset Structure

This tutorial uses the **TS (TotalSegmentator) Heart Dataset** located at:
```
/mnt/nas1/disk01/weidongguo/dataset/TS
```

**Dataset organization**:
```
TS/
├── TS_heart_cropped_resize_train/  (596 cases)
│   ├── s0331/
│   │   ├── s0331-image.nii.gz   # 3D CT volume (64x64x64)
│   │   └── s0331-label.nii.gz   # Multi-class segmentation (6 classes: 0-5)
│   ├── s0332/
│   └── ...
├── TS_heart_cropped_resize_test/   (150 cases)
│   ├── s0004/
│   │   ├── s0004-image.nii.gz
│   │   └── s0004-label.nii.gz
│   └── ...
└── TS_heart_cropped_resize/        (746 cases total, combined)
```

**Volume properties**:
- **Shape**: 64×64×64 voxels (isotropic)
- **Spacing**: 3mm isotropic (from affine matrix)
- **Classes**: 6 (background=0, 5 cardiac structures)
- **Format**: NIfTI compressed (.nii.gz)

## Prerequisites

Activate the conda environment:
```shell
conda activate sam3d-objects
```

Ensure CUDA is available:
```shell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 1: Download Pretrained Checkpoints

Download SAM3D pretrained weights from Hugging Face:
```shell
python scripts/download_hf_checkpoints.py \
    --repo facebook/sam-3d-objects \
    --out checkpoints/hf
```

**Note**: If the repo is gated, authenticate first:
```shell
export HF_TOKEN=your_token_here
# or use: huggingface-cli login
```

Expected files in `checkpoints/hf/`:
- `ss_encoder.safetensors` - SAM3D image encoder
- `ss_generator.ckpt` - SAM3D generator
- `slat_decoder_*.ckpt` - SLAT decoders (mesh/gaussian)

## Step 2: Preprocess Data

Convert raw NIfTI medical images to preprocessed cache:
```shell
# For TS Heart dataset (training set)
python scripts/reprocess_ts_nifti.py \
    --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_train \
    --out ./dataset/ts_processed \
    --classes 5 \
    --extract_mesh \
    --spacing 3.0

# For test set
python scripts/reprocess_ts_nifti.py \
    --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_test \
    --out ./dataset/ts_processed_test \
    --classes 5 \
    --extract_mesh \
    --spacing 3.0
```

**Key arguments**:
- `--original_nifti_dir`: Directory with raw `.nii` or `.nii.gz` files (each case in subfolder with `*-image.nii.gz` and `*-label.nii.gz`)
- `--out`: Output directory for processed `.npz` slice caches
- `--classes`: Number of foreground segmentation classes (5 for TS Heart: LA, LV, RA, RV, Myocardium)
- `--extract_mesh`: Extract surface meshes using marching cubes
- `--spacing`: Isotropic voxel spacing in mm (3.0 for TS Heart dataset)

**Output structure**:
```
dataset/ts_processed/
├── s0001.nii_axis0_slice0050.npz
├── s0001.nii_axis1_slice0128.npz
├── s0002.nii_axis2_slice0064.npz
└── ...
```

Each `.npz` contains: `image`, `mask`, `pointmap`, `affine`, `slice_idx`, `axis`, `gt_sdf_path`

## Step 3: Train with LoRA

### Quick start (simplified wrapper):
```shell
./scripts/run_medical_pipeline.sh \
    --gpu 1 \
    --batch_size 4 \
    --epochs 50 \
    --preprocess_crop_size 256,256
```

### Full control (Hydra config):
```shell
python scripts/train_medical_hydra.py \
    data.preprocess_crop_size=[256,256] \
    training.epochs=50 \
    training.batch_size=4 \
    lora.rank=8 \
    lora.alpha=16
```

**Common options**:
- `--gpu <id>`: GPU device index
- `--batch_size <n>`: Batch size (default: 4)
- `--epochs <n>`: Training epochs (default: 50)
- `--preprocess_crop_size H,W`: Normalize slice size to (H, W)
- `--lora_rank <n>`: LoRA rank (default: 8)
- `--resume <path>`: Resume from checkpoint

**Resume training**:
```shell
./scripts/run_medical_pipeline.sh \
    --gpu 1 \
    --resume checkpoints/medical/epoch_10.pt \
    --epochs 100
```

**Training outputs**:
- `checkpoints/medical/best.pt` - Best model (lowest loss)
- `checkpoints/medical/epoch_*.pt` - Epoch checkpoints
- `outputs/YYYY-MM-DD/*/train.log` - Training logs

## Step 4: Evaluation

Evaluate trained model on test set:
```shell
python scripts/eval_medical.py \
    --checkpoint checkpoints/medical/best.pt \
    --data_root dataset/ts_processed \
    --output_dir results/evaluation \
    --batch_size 8
```

**Metrics computed**:
- **Dice Score**: Volumetric overlap (higher is better, 0-1)
- **HD95**: Hausdorff distance 95th percentile (lower is better, mm)
- **Chamfer Distance**: Surface reconstruction quality (lower is better)
- **Surface Dice**: Boundary accuracy at 1mm tolerance

**Outputs**:
- `results/evaluation/metrics.json` - Per-case and aggregate metrics
- `results/evaluation/visualizations/` - Prediction overlays (if `--save_viz`)

## Step 5: Visualization

Visualize fine-tuned model predictions:
```shell
python scripts/visualize_finetuned.py \
    --lora_checkpoint checkpoints/medical/best.pt \
    --image /path/to/test_image.png \
    --mask_dir /path/to/masks \
    --mask_index 0 \
    --output_dir results/visualizations
```

**For medical volumes**:
```shell
python scripts/visualize_comparison.py \
    --checkpoint checkpoints/medical/best.pt \
    --volume /path/to/test.nii.gz \
    --slice_idx 64 \
    --axis 2 \
    --output results/comparison.png
```

**Outputs**:
- PNG files with prediction overlays
- 3D mesh `.obj` files (if applicable)
- Slice-by-slice comparison grids

## Troubleshooting

**DataLoader size error**:
```
RuntimeError: stack expects each tensor to be equal size
```
→ Increase `--preprocess_crop_size` or rerun preprocessing with consistent size

**CUDA out of memory**:
→ Reduce `--batch_size` or `--preprocess_crop_size`

**nvdiffrast warning**:
```
Cannot import nvdiffrast
```
→ Optional dependency for GPU rasterization, can be safely ignored

**Missing checkpoints**:
→ Ensure Step 1 completed successfully and files exist in `checkpoints/hf/`

## Advanced: Hydra Configuration

All training parameters in `configs/train.yaml` can be overridden via CLI:
```shell
python scripts/train_medical_hydra.py \
    data.slice_cache_dir=dataset/custom \
    data.preprocess_crop_size=[192,192] \
    training.learning_rate=1e-4 \
    training.weight_decay=0.01 \
    lora.rank=16 \
    lora.alpha=32 \
    loss.w_mask_loss=1.0 \
    loss.w_sdf_loss=0.5
```

See `configs/train.yaml` for full parameter list.

---

**Quick Reference**:
```shell
# 1. Download checkpoints
python scripts/download_hf_checkpoints.py --repo facebook/sam-3d-objects --out checkpoints/hf

# 2. Preprocess data (TS Heart dataset)
python scripts/reprocess_ts_nifti.py --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_train --out dataset/ts_processed --classes 5 --spacing 3.0

# 3. Train
./scripts/run_medical_pipeline.sh --gpu 0 --batch_size 4 --epochs 50 --preprocess_crop_size 256,256

# 4. Evaluate
python scripts/eval_medical.py --checkpoint checkpoints/medical/best.pt --data_root dataset/processed --output_dir results

# 5. Visualize
python scripts/visualize_finetuned.py --lora_checkpoint checkpoints/medical/best.pt --image test.png --output_dir viz
```
```
