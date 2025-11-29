# Quickstart: Training (Simplified)

This quickstart shows an easy way to run preprocessing, training and evaluation using the provided scripts.

## 1) Preprocess your data (NIfTI files)

```bash
conda activate sam3d-objects
python scripts/reprocess_ts_nifti.py \
  --original_nifti_dir /path/to/raw_nifti \
  --out_dir /path/to/preprocessed \
  --classes 1 \
  --axes 0,1,2 \
  --extract_mesh \
  --save_metadata
```

## 2) Train with the dataset using Hydra (preferred)

```bash
# Using the hydra YAML configuration (recommended):
python scripts/train_medical_hydra.py \
  data.use_dataset=true \
  data.data_root=/path/to/preprocessed \
  training.batch_size=4 \
  training.epochs=50 \
  training.lr=1e-3 \
  device=cuda \
  checkpoint.dir=./checkpoints/medical \
  lora.rank=4 \
  lora.alpha=8.0 \
  training.num_workers=4 \
  data.augment=true
```

You can still call the original `train_medical.py` for compatibility, but we recommend migrating to the Hydra YAML-based runner.

If you prefer to run the full pipeline (preprocess -> train -> eval) in one step, use the helper script:

```bash
./scripts/run_medical_pipeline.sh --raw_nifti /path/to/raw_niftis --out /path/to/preprocessed --epochs 20 --batch_size 4
```

### For quick debugging (no data)
Run the dummy training harness (default):

```bash
python scripts/train_medical.py --batch_size 4 --epochs 2
```

This uses the internal dummy dataset and is useful for testing loops, LoRA injection, and checkpointing on a developer machine.
