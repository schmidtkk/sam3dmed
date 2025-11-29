conda activate sam3d-objects

STATUS NOTE — Read before starting implementation
===============================================

- Default environment activation (run once per terminal session):
  conda activate sam3d-objects

- Current default settings for the Medical SAM3D fine-tuning project:
  - Data resampling: adaptive per modality, default 1.0 mm isotropic
  - Input 2D image size: 256 x 256 (configurable)
  - Slice sampling: single-slice per sample, `slice_sampling_method=random_axis`
  - Caching: `cache_slices=True` by default
  - Pose: `include_pose=False` (shape-only)
  - LoRA: `rank=4`, `alpha=8`, freeze base weights, only train adapters & decoder heads
  - Initial training mode: MESH-ONLY (SLatMeshDecoder) — train mesh decoder supervised by SDFs + mesh losses; Gaussian decoder training is optional for later experiments
  - Primary metrics: 3D IoU/Dice, Chamfer Distance, HD95
  - Augmentations: Per-slice geometric/intensity augmentations enabled in training

  Test & environment instructions:
  1. Activate environment:
    conda activate sam3d-objects
  2. Install dependencies (if not already installed):
    pip install -r requirements.txt
    pip install -r requirements.dev.txt
  3. Optional metric libs (recommended):
    pip install pytorch3d surface-distance
  4. Run tests:
    LIDRA_SKIP_INIT=1 pytest -q tests/test_metrics.py tests/test_augmentations.py

  Note: If you do not use pytest, you can run the quick check with Python:
    LIDRA_SKIP_INIT=1 python -c "from sam3d_objects.utils.metrics import compute_dice; import numpy as np; print(compute_dice(np.ones((2,2,2)), np.ones((2,2,2))))"

## Implementation Progress

### Completed
- [x] Phase 1: Docs & plan finalization
- [x] Phase 2: Metric wrappers (`sam3d_objects/utils/metrics.py`)
- [x] Phase 3: Dataset augmentations (`sam3d_objects/data/dataset/ts/slice_augmentations.py`)

### In Progress
- [ ] Phase 4: Reprocessing script (`scripts/reprocess_ts_nifti.py`)

### Not Started
- [ ] Phase 5: TS_SAM3D_Dataset loader finalization
- [ ] Phase 6: LoRA adapters
- [ ] Phase 7: Training harness
- [ ] Phase 8: Evaluation pipeline
- [ ] Phase 9: CI tests
- [ ] Phase 10: Final review

Follow these settings unless an experiment requires change; update this file to reflect changes in defaults.
