# LoRA Fine-tuning Implementation Review

✅ Summary

- LoRA (Low-Rank Adaptation) support was implemented and integrated for mesh-only medical fine-tuning of SAM3D.
- Key file: `sam3d_objects/model/lora.py` — implements the LoRALinear wrapper and a number of helper utilities.
- Training integration is implemented in `scripts/train_medical.py` via `MedicalTrainer._setup_lora()` which uses `setup_lora_for_medical_finetuning()`.
- Checkpointing saves LoRA-only weights to keep checkpoints compact (via `get_lora_state_dict` and `load_lora_state_dict`).
- Tests: `tests/test_lora.py` includes a thorough test suite covering injection, merge, freeze, load/save, and param counts.

---

## What was refactored / Added

1. `sam3d_objects/model/lora.py`
   - New file with complete LoRA utilities:
     - `LoRALinear` nn.Module — wraps an `nn.Linear` to add a LoRA path.
     - `_find_modules_by_name` — finds target linear layers by name pattern.
     - `inject_lora` — performs in-place injection of `LoRALinear` into specific linear layers.
     - `freeze_base_params` — freezes base parameters (all except LoRA parameters).
     - `get_lora_params` — returns trainable LoRA params for optimizer creation.
     - `get_lora_state_dict` / `load_lora_state_dict` — save/load LoRA-only weights to keep checkpoints compact.
     - `merge_lora_weights` — merges LoRA deltas into base linear weights and replaces `LoRALinear` modules with merged `nn.Linear` for inference.
     - `count_parameters` — utility to show `total`, `trainable`, and `lora` counts.
     - `setup_lora_for_medical_finetuning` — high-level convenience that injects LoRA, freezes base params and optionally unfreezes output layers.

2. `scripts/train_medical.py`
   - The `MedicalTrainer._setup_lora()` method now calls `setup_lora_for_medical_finetuning(...)` to ensure a consistent LoRA setup.
   - Training uses `trainable_params = [p for p in self.model.parameters() if p.requires_grad]` (Train only LoRA + optional output layers).
   - Checkpoint code `save_checkpoint()` now stores `lora_state_dict` and `load_checkpoint()` `load_lora_state_dict()` to restore LoRA-only weights.

3. `scripts/run_medical_pipeline.sh` and `markdown/quickstart_training.md`
   - Pipeline simplified to directly call `scripts/train_medical.py --use_dataset`, using the standard CLI interface instead of embedding an inline python block.

4. Tests
   - `tests/test_lora.py` added/updated to exercise functionality across LoRA basics and edge cases:
     - `LoRALinear` initialization and forward pass
     - Merge functionality (ensures merged model equals LoRA-injected model output)
     - `inject_lora` and `setup_lora_for_medical_finetuning` behaviors
     - Freeze/unfreeze and state-dict loader/saver utilities

---

## Design and Implementation Details

### LoRALinear
- Wraps a frozen `nn.Linear` (used for base projection layers in the model) with a low-rank path implemented as `A` and `B` matrices:
  - `A` has shape (rank, in_features)
  - `B` has shape (out_features, rank)
- Computes output as:
  - `result = W x + (alpha / r) * B @ (A @ x)
  - A dropout path is optionally included.
- Initialization:
  - `A` uses Kaiming uniform initialization (to match common LoRA practice for well-conditioned initial transforms).
  - `B` initialized to zeros to ensure initial behavior equals `original_linear` behavior.
- Methods & properties:
  - `merge_weights()` returns a fresh `nn.Linear` that has `W + delta_w` applied, useful for inference and removing LoRA overhead.
  - `weight` property returns computed merged weight on the fly but does not update the base weight in the model until `merge_weights()` is used.

### Injection / Discovery
- `_find_modules_by_name(model, target_names)` recursively walks named children and finds `nn.Linear` modules whose attribute name contains any of the `target_names` patterns (e.g., `to_qkv`, `to_out`).
- `inject_lora(model, target_modules, rank, alpha, dropout)` replaces each matching `nn.Linear` with `LoRALinear` wrapper.
  - Important unsafe assumption: this approach looks up linear modules by name matching substring patterns; it relies on consistent naming across model backbones (e.g., `to_qkv` or `to_out`).

### Freeze / Optimizer
- `freeze_base_params(model)` sets `requires_grad=False` for all parameters except `lora_A` and `lora_B`.
- `get_lora_params(model)` fetches trainable LoRA params for constructing optimizers.
- `setup_lora_for_medical_finetuning` bundles the above and optionally unfreezes common output/head layers so they can be trained alongside LoRA.

### Checkpoints
- `get_lora_state_dict(model)` extracts only the LoRA parameters from the model state dict (compact and efficient).
- `load_lora_state_dict(model, lora_state)` loads LoRA-only weights back into a model where LoRA is already injected.
- `MedicalTrainer.save_checkpoint()` and `load_checkpoint()` are wired to use these functions to keep checkpoint size small and load safely.

### Merging for Inference
- `merge_lora_weights(model)` replaces `LoRALinear` modules with merged `nn.Linear` modules (i.e., `W <- W + delta_w`) for inference-only performance.
  - This is useful for shipping or evaluating a model without adapter overhead.

---

## Integration Points

- `scripts/train_medical.py`:
  - Calls `setup_lora_for_medical_finetuning()` in `MedicalTrainer._setup_lora()`.
  - Optimizers are created using only trainable params.
  - Checkpointing uses `get_lora_state_dict()`.
  - Loading uses `load_lora_state_dict()`.

- `scripts/eval_medical.py`:
  - The evaluator expects the model to be merged or LoRA layers to be present; `inject_lora()` + `load_lora_state_dict()` is used before evaluation if a LoRA-only checkpoint is loaded.

- Dataset & Pipeline:
  - No coupling changes: `TS_SAM3D_Dataset` still provides samples, collate function remains the same.
  - `scripts/run_medical_pipeline.sh` now calls `train_medical.py --use_dataset` to keep pipeline simple.

---

## Tests
- `tests/test_lora.py` covers:
  - `LoRALinear` forward, output shapes, `merge_weights`.
  - `inject_lora` module injection and behavior.
  - `freeze_base_params` ensures only LoRA params are trainable.
  - `get_lora_state_dict` / `load_lora_state_dict` round-trip.
  - `merge_lora_weights` produces the same forward outputs as original LoRA-wrapped model.
- The `train_medical` tests (`tests/test_train_medical.py`) validate that the training harness honors LoRA freeze, checkpointing, and validation behavior.

All tests pass in the current repo state (69 tests including LoRA tests).

---

## Edge Cases & Risks
- Name matching and injection approach relies on the attribute naming of modules (e.g., `to_qkv`, `to_out`) — if the model structure or naming changes, `inject_lora()` may not find intended modules and will warn. Consider more robust matching strategies or configuration via exact names.
- `load_lora_state_dict` does not strictly require or validate that LoRA modules are present; it logs `warning` for missing names. This is useful for compatibility, but no runtime errors are raised. That design choice is convenient but may hide silent mismatches.
- When `merge_lora_weights` replaces modules, it mutates the original model — ensure you don’t accidentally call it during training unless intended. Consider an `inplace=False` option returning a merged copy.
- LoRA training uses `state_dict` with default torch serialization; for production, `weights_only=True` safeguards could be considered.

---

## Recommended Improvements
1. Expand `inject_lora` with either a config with explicit module names or allow a callback for module selection.
2. Add per-module LoRA config (e.g., different ranks for certain layers).
3. Add integration tests for an entire LoRA training + merge cycle to guard against regression (train, save, load with LoRA, run inference with merged weights).
4. Provide a small helper to produce a `merged_model()` copy rather than in-place merging.
5. Add an optional `strict` argument for `load_lora_state_dict` to fail loudly when mismatch occurs.
6. Add docs for how to export a merged model (and optionally export a `torchscript` or `ONNX` shippable artifact).

---

## Example: Typical Usage

1. Inject LoRA and freeze base params (Trainer handles this automatically):
```python
from sam3d_objects.model.lora import setup_lora_for_medical_finetuning

setup_lora_for_medical_finetuning(model, rank=4, alpha=8, dropout=0.0, unfreeze_output_layers=True)
```

2. Create optimizer using only trainable params (LoRA + optional heads):
```python
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
```

3. Save a LoRA-only checkpoint:
```python
from sam3d_objects.model.lora import get_lora_state_dict
ckpt = {
    'epoch': 10,
    'lora_state_dict': get_lora_state_dict(model)
}
torch.save(ckpt, 'lora_checkpoint.pt')
```

4. Load LoRA-only checkpoint (ensure LoRA injected):
```python
from sam3d_objects.model.lora import inject_lora, load_lora_state_dict
inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4, alpha=8)
checkpoint = torch.load('lora_checkpoint.pt', map_location='cpu')
load_lora_state_dict(model, checkpoint['lora_state_dict'])
```

5. Optional: Merge LoRA weights for inference:
```python
from sam3d_objects.model.lora import merge_lora_weights
merge_lora_weights(model)  # replaces LoRALinear with merged nn.Linear
```

---

## Final Notes
- The LoRA implementation is compact, well-tested, and fits the training harness flow.
- It maintains a clean separation of concerns (LoRA utilities are confined to `lora.py` and do not bloat the trainer logic).
- The training harness `scripts/train_medical.py` properly handles using LoRA for parameter-efficient tuning and persists LoRA-only checkpoints.

If you’d like, I can:
- Add an example `train_medical` run command in `README.md` with specific flags for LoRA training.
- Add a unit test that verifies `merge_lora_weights` + inference on a small model exactly matches using LoRA wrappers.
- Add `strict` mode to `load_lora_state_dict` and add checks for param shape mismatches.

Would you like any of these additions? For the next step I can implement one of the recommended improvements or add a usage snippet to the repo README if you'd prefer.