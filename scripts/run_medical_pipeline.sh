#!/usr/bin/env bash
set -euo pipefail

# Skip LIDRA initialization (internal module removed from this repo)
export LIDRA_SKIP_INIT=1

# Medical Fine-Tuning Pipeline for SAM3D
# NOTE: This script assumes the `sam3d-objects` environment is already activated.
#
# Usage:
#   ./scripts/run_medical_pipeline.sh [OPTIONS]
#
# Options:
#   --gpu <id>            GPU index (default: 1)
#   --batch_size <n>      Override batch size
#   --epochs <n>          Override epochs
#   --lora_rank <n>       Override LoRA rank
#   --data_root <path>    Path to preprocessed data
#   --resume <path>       Resume from checkpoint
#   --help                Show this help
#
# All other config is in configs/train.yaml
# Hydra overrides can be passed as additional arguments.

# ============================================================
# Defaults (override via CLI or modify train.yaml)
# ============================================================
GPU_ID="${GPU_ID:-1}"
DATA_ROOT="${SAM3D_DATA_ROOT:-./dataset}"
SLICE_CACHE="${SAM3D_SLICE_CACHE:-./dataset/ts_processed}"
RESUME=""

# Hydra overrides collected here
HYDRA_OVERRIDES=()

# ============================================================
# Parse CLI arguments
# ============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU_ID="$2"; shift 2;;
    --batch_size) HYDRA_OVERRIDES+=("training.batch_size=$2"); shift 2;;
    --epochs) HYDRA_OVERRIDES+=("training.epochs=$2"); shift 2;;
    --lora_rank) HYDRA_OVERRIDES+=("lora.rank=$2"); shift 2;;
    --data_root) DATA_ROOT="$2"; shift 2;;
    --slice_cache) SLICE_CACHE="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --preprocess_crop_size) PREPROCESS_CROP_SIZE="$2"; shift 2;;
    --help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      # Pass unknown args as Hydra overrides
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

# ============================================================
# Setup environment
# ============================================================
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU: $GPU_ID"

# Resolve paths
DATA_ROOT=$(realpath "$DATA_ROOT" 2>/dev/null || echo "$DATA_ROOT")
if [[ -d "$SLICE_CACHE" ]]; then
  SLICE_CACHE=$(realpath "$SLICE_CACHE")
fi

# ============================================================
# Training
# ============================================================
echo "[Pipeline] Starting training with Hydra config"

# Build command
CMD=(
  python -u
  scripts/train_medical_hydra.py
  "data.data_root=$DATA_ROOT"
  "data.slice_cache_dir=$SLICE_CACHE"
)

# Add resume if specified
if [[ -n "$RESUME" ]]; then
  CMD+=("checkpoint.resume=$RESUME")
fi

# Add any CLI overrides
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${HYDRA_OVERRIDES[@]}")
fi

# Add preprocess_crop_size override if provided in CLI: e.g., --preprocess_crop_size 256,256
if [[ -n "${PREPROCESS_CROP_SIZE:-}" ]]; then
  # Ensure format [H,W]
  PREPROCESS_CROP_SIZE_STR="[${PREPROCESS_CROP_SIZE}]"
  CMD+=("data.preprocess_crop_size=${PREPROCESS_CROP_SIZE_STR}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "[Pipeline] Training complete"
