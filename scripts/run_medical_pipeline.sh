#!/usr/bin/env bash
set -euo pipefail

# Run preprocessing, training and evaluation for SAM3D medical fine-tuning.
# Usage:
#   ./scripts/run_medical_pipeline.sh --raw_nifti /path/to/raw --out /path/to/preprocessed \
#       --epochs 20 --batch_size 4 --device cuda --resume ./checkpoints/medical/best.pt
#
# Options:
#   --raw_nifti <path>    Path to raw original NIfTI files (required)
#   --out <path>          Output directory for preprocessed dataset (required)
#   --classes <n>         Number of foreground classes (default: 1)
#   --epochs <n>          Number of epochs (default: 50)
#   --batch_size <n>      Batch size (default: 4)
#   --lr <float>          Learning rate (default: 1e-3)
#   --device <str>        Device (cuda|cpu) (default: cuda)
#   --no_preprocess       Skip preprocessing step if preprocessed data already exists
#   --only_preprocess     Only run preprocessing and exit
#   --only_train          Only run training (requires preprocessed data)
#   --only_eval           Only run evaluation (requires checkpoint and preprocessed data)
#   --resume <path>       Resume training from checkpoint
#   --help                Show help and exit

# Defaults
EPOCHS=50
BATCH_SIZE=4
LR=1e-3
DEVICE="cuda"
CLASSES=1
NO_PREPROCESS=0
ONLY_PREPROCESS=0
ONLY_TRAIN=0
ONLY_EVAL=0
RESUME=""
NUM_WORKERS=4
LORA_RANK=4
LORA_ALPHA=8.0
CHECKPOINT_DIR="./checkpoints/medical"
EVAL_DIR="./results"

PROGNAME=$(basename "$0")

function usage() {
  sed -n '1,120p' "$0" | sed -n '1,120p'
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --raw_nifti) RAW_NIFTI="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --classes) CLASSES="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --no_preprocess) NO_PREPROCESS=1; shift 1;;
    --only_preprocess) ONLY_PREPROCESS=1; shift 1;;
    --only_train) ONLY_TRAIN=1; shift 1;;
    --only_eval) ONLY_EVAL=1; shift 1;;
    --resume) RESUME="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --lora_rank) LORA_RANK="$2"; shift 2;;
    --lora_alpha) LORA_ALPHA="$2"; shift 2;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2;;
    --eval_dir) EVAL_DIR="$2"; shift 2;;
    --help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Validate required args
if [[ -z "${RAW_NIFTI:-}" ]] && [[ $ONLY_TRAIN -eq 0 ]] && [[ $ONLY_EVAL -eq 0 ]]; then
  echo "Error: --raw_nifti required unless running only train/eval" >&2
  usage
  exit 1
fi
if [[ -z "${OUT_DIR:-}" ]]; then
  echo "Error: --out required" >&2
  usage
  exit 1
fi

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  echo "Activating conda environment: sam3d-objects"
  # shellcheck disable=SC1091
  source ~/.bashrc || true
  conda activate sam3d-objects || true
else
  echo "Warning: conda not found; ensure python environment has dependencies installed"
fi

# Ensure out dirs
OUT_DIR=$(realpath "$OUT_DIR")
mkdir -p "$OUT_DIR"
CHECKPOINT_DIR=$(realpath "$CHECKPOINT_DIR")
EVAL_DIR=$(realpath "$EVAL_DIR")
mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR"

# 1) Preprocessing
if [[ $NO_PREPROCESS -eq 0 && $ONLY_TRAIN -eq 0 && $ONLY_EVAL -eq 0 ]]; then
  echo "[Pipeline] Running preprocessing: $RAW_NIFTI -> $OUT_DIR"
  python3 scripts/reprocess_ts_nifti.py \
    --original_nifti_dir "$RAW_NIFTI" \
    --out_dir "$OUT_DIR" \
    --classes "$CLASSES" \
    --axes 0,1,2 \
    --extract_mesh \
    --save_metadata \
    --spacing 1.0 || { echo "Preprocessing failed"; exit 1; }
  echo "[Pipeline] Preprocessing finished"
  if [[ $ONLY_PREPROCESS -eq 1 ]]; then
    echo "Exiting as only_preprocess flag was set"; exit 0
  fi
elif [[ $NO_PREPROCESS -eq 1 ]]; then
  echo "[Pipeline] Skipping preprocessing per --no_preprocess"
fi

# 2) Training: use inline Python to import train_medical as module and run with real dataset
if [[ $ONLY_EVAL -eq 0 ]]; then
  echo "[Pipeline] Starting training (epochs=$EPOCHS, batch_size=$BATCH_SIZE)"
  # Call the train_medical script directly. This supports --use_dataset to switch
  # to the real dataset loader (TS_SAM3D_Dataset). Using direct CLI keeps the script
  # simple and avoids inline Python.
  python3 scripts/train_medical.py \
    --use_dataset \
    --data_root "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --num_workers "$NUM_WORKERS" \
    --augment \
    ${RESUME:+--resume "$RESUME"} || { echo "Training failed"; exit 1; }
  echo "[Pipeline] Training complete"
fi

# 3) Evaluation
if [[ -n "$RESUME" || $(ls -A "$CHECKPOINT_DIR" 2>/dev/null | wc -l) -gt 0 ]]; then
  # pick best checkpoint if available
  if [[ -z "$RESUME" ]]; then
    if [[ -f "$CHECKPOINT_DIR/best.pt" ]]; then
      RESUME="$CHECKPOINT_DIR/best.pt"
    else
      RESUME=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -n 1 || true)
    fi
  fi
  if [[ -z "$RESUME" ]]; then
    echo "No checkpoint found for evaluation"; exit 1
  fi
fi

if [[ $ONLY_PREPROCESS -eq 0 && $ONLY_TRAIN -eq 0 && $ONLY_EVAL -eq 0 ]] || [[ $ONLY_EVAL -eq 1 ]]; then
  echo "[Pipeline] Running evaluation against checkpoint: $RESUME"
  python3 scripts/eval_medical.py --checkpoint "$RESUME" --data_root "$OUT_DIR" --output_dir "$EVAL_DIR" --device "$DEVICE" --save_predictions || { echo "Evaluation failed"; exit 1; }
  echo "[Pipeline] Evaluation complete; results in $EVAL_DIR"
fi

echo "[Pipeline] All done"
exit 0
