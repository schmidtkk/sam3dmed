#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Test Script for Medical Fine-Tuning Pipeline
# =============================================================================
# This script runs quick sanity tests for both Stage 1 and Stage 2 training.
# It verifies that the pipeline runs end-to-end without errors.
#
# Usage:
#   ./scripts/test_medical_pipeline.sh [OPTIONS]
#
# Options:
#   --gpu <id>            GPU index (default: 1)
#   --stage <mode>        Test stage: stage1, stage2, both (default: both)
#   --epochs <n>          Number of epochs for test (default: 1)
#   --batch_size <n>      Batch size for test (default: 2)
#   --help                Show this help
#
# Example:
#   ./scripts/test_medical_pipeline.sh --gpu 1 --stage both
#   ./scripts/test_medical_pipeline.sh --stage stage2 --epochs 2

# Skip LIDRA initialization
export LIDRA_SKIP_INIT=1

# ============================================================
# Defaults
# ============================================================
GPU_ID="${GPU_ID:-1}"
TEST_STAGE="both"
EPOCHS=1
BATCH_SIZE=2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# Parse CLI arguments
# ============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU_ID="$2"; shift 2;;
    --stage) TEST_STAGE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --help)
      sed -n '1,24p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ============================================================
# Setup
# ============================================================
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "SAM3D Medical Pipeline Test"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Test Stage: $TEST_STAGE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "============================================================"
echo ""

# Check that checkpoints exist
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/hf"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
  echo "Please download checkpoints first."
  exit 1
fi

# Check key checkpoints
for ckpt in slat_decoder_mesh.ckpt slat_decoder_mesh.yaml ss_generator.ckpt ss_generator.yaml; do
  if [[ ! -f "$CHECKPOINT_DIR/$ckpt" ]]; then
    echo "ERROR: Missing checkpoint: $CHECKPOINT_DIR/$ckpt"
    exit 1
  fi
done

echo "[OK] All required checkpoints found"
echo ""

# ============================================================
# Test Functions
# ============================================================
test_stage2() {
  echo "============================================================"
  echo "Testing Stage 2: SLatMeshDecoder Training"
  echo "============================================================"
  
  local output_dir="$PROJECT_ROOT/checkpoints/test_stage2"
  rm -rf "$output_dir"
  mkdir -p "$output_dir"
  
  echo "Output: $output_dir"
  echo ""
  
  python -u scripts/train_medical.py \
    training.mode=stage2_only \
    training.epochs=$EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.num_workers=0 \
    data.slice_cache_dir="$PROJECT_ROOT/dataset/ts_processed" \
    checkpoint.dir="$output_dir" \
    checkpoint.save_every=1 \
    checkpoint.validate_every=1 \
    hydra.run.dir="$output_dir/hydra"
  
  # Verify outputs
  if [[ -f "$output_dir/final.pt" ]]; then
    echo ""
    echo "[PASS] Stage 2 test completed successfully!"
    echo "  - Checkpoint saved: $output_dir/final.pt"
    return 0
  else
    echo ""
    echo "[FAIL] Stage 2 test failed - no checkpoint saved"
    return 1
  fi
}

test_stage1() {
  echo "============================================================"
  echo "Testing Stage 1: SS Generator Training"
  echo "============================================================"
  
  local output_dir="$PROJECT_ROOT/checkpoints/test_stage1"
  rm -rf "$output_dir"
  mkdir -p "$output_dir"
  
  echo "Output: $output_dir"
  echo ""
  
  python -u scripts/train_medical.py \
    training.mode=stage1_only \
    stage1.enabled=true \
    training.epochs=$EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.num_workers=0 \
    data.slice_cache_dir="$PROJECT_ROOT/dataset/ts_processed" \
    checkpoint.dir="$output_dir" \
    checkpoint.save_every=1 \
    checkpoint.validate_every=1 \
    hydra.run.dir="$output_dir/hydra"
  
  # Verify outputs
  if [[ -f "$output_dir/final.pt" ]]; then
    echo ""
    echo "[PASS] Stage 1 test completed successfully!"
    echo "  - Checkpoint saved: $output_dir/final.pt"
    return 0
  else
    echo ""
    echo "[FAIL] Stage 1 test failed - no checkpoint saved"
    return 1
  fi
}

test_two_stage() {
  echo "============================================================"
  echo "Testing Two-Stage Training"
  echo "============================================================"
  
  local output_dir="$PROJECT_ROOT/checkpoints/test_two_stage"
  rm -rf "$output_dir"
  mkdir -p "$output_dir"
  
  echo "Output: $output_dir"
  echo ""
  
  python -u scripts/train_medical.py \
    training.mode=two_stage \
    stage1.enabled=true \
    training.epochs=$EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.num_workers=0 \
    data.slice_cache_dir="$PROJECT_ROOT/dataset/ts_processed" \
    checkpoint.dir="$output_dir" \
    checkpoint.save_every=1 \
    checkpoint.validate_every=1 \
    hydra.run.dir="$output_dir/hydra"
  
  # Verify outputs
  if [[ -f "$output_dir/final.pt" ]]; then
    echo ""
    echo "[PASS] Two-stage test completed successfully!"
    echo "  - Checkpoint saved: $output_dir/final.pt"
    return 0
  else
    echo ""
    echo "[FAIL] Two-stage test failed - no checkpoint saved"
    return 1
  fi
}

# ============================================================
# Run Tests
# ============================================================
PASSED=0
FAILED=0

case "$TEST_STAGE" in
  stage1)
    if test_stage1; then
      ((PASSED++))
    else
      ((FAILED++))
    fi
    ;;
  stage2)
    if test_stage2; then
      ((PASSED++))
    else
      ((FAILED++))
    fi
    ;;
  two_stage)
    if test_two_stage; then
      ((PASSED++))
    else
      ((FAILED++))
    fi
    ;;
  both)
    echo "Running Stage 2 test first (faster)..."
    echo ""
    if test_stage2; then
      ((PASSED++))
    else
      ((FAILED++))
    fi
    
    echo ""
    echo "Running Stage 1 test..."
    echo ""
    if test_stage1; then
      ((PASSED++))
    else
      ((FAILED++))
    fi
    ;;
  all)
    echo "Running all training modes..."
    echo ""
    
    if test_stage2; then ((PASSED++)); else ((FAILED++)); fi
    echo ""
    if test_stage1; then ((PASSED++)); else ((FAILED++)); fi
    echo ""
    if test_two_stage; then ((PASSED++)); else ((FAILED++)); fi
    ;;
  *)
    echo "Unknown stage: $TEST_STAGE"
    echo "Valid options: stage1, stage2, two_stage, both, all"
    exit 1
    ;;
esac

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [[ $FAILED -gt 0 ]]; then
  echo "Some tests FAILED!"
  exit 1
else
  echo "All tests PASSED!"
  exit 0
fi
