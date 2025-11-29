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
- [x] Phase 4: Reprocessing script (`scripts/reprocess_ts_nifti.py`)
- [x] Phase 5: TS_SAM3D_Dataset loader finalization

### Completed
- [x] Phase 6: LoRA adapters
  - `sam3d_objects/model/lora.py`: LoRALinear class, inject_lora, freeze_base_params, merge utilities
  - Tests: `tests/test_lora.py` (18 tests pass)
- [x] Phase 7: Training harness
  - `scripts/train_medical.py`: MedicalTrainer with LoRA, SDF+Chamfer+mesh_reg losses
  - Tests: `tests/test_train_medical.py` (17 tests pass)

### Completed
- [x] Phase 8: Evaluation pipeline
  - `scripts/eval_medical.py`: MedicalEvaluator with Dice, HD95, Chamfer metrics
  - Tests: `tests/test_eval_medical.py` (14 tests pass)
- [x] Phase 9: CI configuration
  - `.github/workflows/test.yml`: GitHub Actions for linting and testing
  - `pyproject.toml`: pytest + ruff + coverage configuration
  - `tests/conftest.py`: Shared test fixtures
  - All 69 tests pass, ruff linting clean
- [x] Phase 10: Final review ✅

---

## Final Implementation Summary

### Core Components
| Component | File | Description |
|-----------|------|-------------|
| **Metrics** | `sam3d_objects/utils/metrics.py` | Dice, Chamfer, HD95 (PyTorch3D/SciPy fallback) |
| **Augmentations** | `sam3d_objects/data/dataset/ts/slice_augmentations.py` | SliceAugmentor with geometric/intensity transforms |
| **Preprocessing** | `scripts/reprocess_ts_nifti.py` | Adaptive spacing, SDF computation, mesh extraction |
| **Dataset** | `sam3d_objects/data/dataset/ts/ts_sam3d_dataloader.py` | TS_SAM3D_Dataset with caching, filtering |
| **LoRA** | `sam3d_objects/model/lora.py` | LoRALinear, injection, freezing, merging |
| **Training** | `scripts/train_medical.py` | MedicalTrainer with mixed precision, gradient accumulation |
| **Evaluation** | `scripts/eval_medical.py` | MedicalEvaluator with 3D volumetric metrics |

### Test Coverage (69 tests)
| Module | Tests |
|--------|-------|
| Metrics | 4 |
| Augmentations | 9 |
| Dataset | 7 |
| LoRA | 18 |
| Training | 17 |
| Evaluation | 14 |

### Quick Start
```bash
# Activate environment
conda activate sam3d-objects

# Run all tests
LIDRA_SKIP_INIT=1 pytest tests/ -v

# Run linting
ruff check sam3d_objects/ scripts/ tests/

# Format code
ruff format sam3d_objects/ scripts/ tests/
```

Follow these settings unless an experiment requires change; update this file to reflect changes in defaults.
