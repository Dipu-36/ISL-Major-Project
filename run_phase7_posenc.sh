#!/usr/bin/env bash
# Phase 7: Positional Encoding + Keyframe Sampling + EMA Smoothing + eta_min scheduler
# Identical hyperparameters to Phase 6 so changes are the only variable.
# Resumes from Phase 6 best checkpoint; optimizer/scheduler start fresh
# (because --unfreeze-decoder-layers 2 triggers fresh-init in the training code).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python3.10"
CHECKPOINT="artifacts/phase6_50k/ce_extended_50000train_e20_seed42/best_model.pt"
OUTPUT_DIR="artifacts/phase7_posenc_keyframes"
LOG_FILE="${OUTPUT_DIR}/run.log"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Phase 6 checkpoint not found at $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Phase 7: Positional Encoding + Keyframe Sampling ==="
echo "Python     : $PYTHON"
echo "Checkpoint : $CHECKPOINT"
echo "Output dir : $OUTPUT_DIR"
echo "Log file   : $LOG_FILE"
echo "Starting at: $(date)"

nohup "$PYTHON" -m training.controlled_generalization_test \
    --single-run \
    --root . \
    --output-dir "$OUTPUT_DIR" \
    --train-samples-large 50000 \
    --epochs 20 \
    --val-samples 500 \
    --batch-size 32 \
    --lr 2.5e-4 \
    --decoder-lr 2.5e-5 \
    --unfreeze-decoder-layers 2 \
    --semantic-lambda 0.0 \
    --contrastive-weight 0.0 \
    --metric-eval-batches 16 \
    --seed 42 \
    --checkpoint-path "$CHECKPOINT" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Launched PID $PID — tailing log (Ctrl-C to detach, training continues)"
echo "$PID" > "${OUTPUT_DIR}/train.pid"
tail -f "$LOG_FILE"
