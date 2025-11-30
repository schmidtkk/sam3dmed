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
# Using the hydra YAML configuration (recommended). Hyperparameters like epochs, batch_size and lr
# are set inside `configs/train.yaml`. Use CLI overrides for rare cases.
python scripts/train_medical_hydra.py \
  data.use_dataset=true \
  data.data_root=/path/to/preprocessed \
  device=cuda \
  checkpoint.dir=./checkpoints/medical

Note: You can select the model in the Hydra config using the `model` section.
For example, to use the SLatMeshDecoder and set small debug params:

```bash
python scripts/train_medical_hydra.py model.name=slat_mesh model.params.resolution=4 model.params.model_channels=32 device=cuda
```
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
### Evaluate & visualize
To evaluate and produce visualizations (HTML/PNG) of predicted meshes or overlays:

```bash
# Evaluate and save predictions + overlays
python scripts/eval_medical.py --checkpoint ./checkpoints/medical/best.pt --data_root /path/to/preprocessed --output_dir ./results --save_predictions --visualize --visualize_format html
```

This will compute evaluation metrics and create an HTML visualization per-sample under `./results/visualizations/` when Plotly and PyTorch3D are available, or a 2D overlay PNG fallback otherwise.

```

This uses the internal dummy dataset and is useful for testing loops, LoRA injection, and checkpointing on a developer machine.
