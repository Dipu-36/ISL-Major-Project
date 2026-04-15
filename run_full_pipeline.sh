#!/usr/bin/env bash
# Full training pipeline: Stage A → Stage B → Stage C with visualization
# This script runs the complete benchmark-clean pipeline

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="logs/full_pipeline_${TIMESTAMP}.log"

mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "======================================================================"
log "Starting Full Training Pipeline"
log "Timestamp: $TIMESTAMP"
log "======================================================================"

# ========================================================================
# Stage 0: Verify prerequisites
# ========================================================================
log ""
log "=== STAGE 0: Verifying Prerequisites ==="

# Check cache
if [[ ! -f "artifacts/dataset_cache/isl_tf96.pt" ]]; then
    log "ERROR: Dataset cache not found. Run cache build first."
    exit 1
fi
log "✓ Dataset cache found"

# Check splits
if [[ ! -f "artifacts/splits.json" ]]; then
    log "ERROR: splits.json not found"
    exit 1
fi
log "✓ Splits file found"

# Count samples
TRAIN_COUNT=$(python3 -c "import json; d=json.load(open('artifacts/splits.json')); print(len(d.get('train_uids', d.get('train', []))))")
VAL_COUNT=$(python3 -c "import json; d=json.load(open('artifacts/splits.json')); print(len(d.get('val_uids', d.get('val', []))))")
log "✓ Training samples: $TRAIN_COUNT"
log "✓ Validation samples: $VAL_COUNT"

# ========================================================================
# Stage A: WPP Pretraining
# ========================================================================
log ""
log "=== STAGE A: WPP Encoder Pretraining ==="
log "Running: train_stage_a_v3.py (full data)"

"$PYTHON" training/train_stage_a_v3.py \
    --run-name stageA_v3 \
    --batch-size 128 \
    --max-epochs 60 \
    --probe-every 5 \
    --seed 42 \
    --num-workers 4 \
    2>&1 | tee "logs/stageA_${TIMESTAMP}.log"

if [[ $? -ne 0 ]]; then
    log "✗ Stage A training failed"
    exit 1
fi
log "✓ Stage A training complete"

# Generate Stage A visualizations
log "Generating Stage A visualizations..."
"$PYTHON" training/visualize_stage_a.py \
    --run-dir artifacts/phase9_encoder_pretrain/stageA_v3 \
    2>&1 | tee -a "$MAIN_LOG"

STAGEA_SUMMARY="artifacts/phase9_encoder_pretrain/stageA_v3/training_summary.txt"
if [[ -f "$STAGEA_SUMMARY" ]]; then
    log ""
    log "Stage A Summary:"
    cat "$STAGEA_SUMMARY" | tee -a "$MAIN_LOG"
fi

# ========================================================================
# Stage B: Gate Check
# ========================================================================
log ""
log "=== STAGE B: Gate Evaluation ==="
log "Running: probe_stage_b_v2.py"

"$PYTHON" probe_stage_b_v2.py \
    --checkpoint artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt \
    2>&1 | tee "logs/stageB_${TIMESTAMP}.log"

GATE_FILE="artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json"
if [[ ! -f "$GATE_FILE" ]]; then
    log "ERROR: Gate file not found"
    exit 1
fi

GATE_PASSED=$($PYTHON -c "import json; d=json.load(open('$GATE_FILE')); print(str(d.get('gate_passed', False)).lower())")
if [[ "$GATE_PASSED" != "true" ]]; then
    log "✗ Stage B gate FAILED"
    log "Review: $GATE_FILE"
    exit 1
fi
log "✓ Stage B gate PASSED"

# ========================================================================
# Stage C: Translation Training
# ========================================================================
log ""
log "=== STAGE C: Translation Training ==="
log "Running: train_stage_c_v2.py (full data)"

"$PYTHON" training/train_stage_c_v2.py \
    --encoder-checkpoint artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt \
    --run-name stageC_v2_benchmark \
    --batch-size 32 \
    --seed 42 \
    --num-workers 4 \
    2>&1 | tee "logs/stageC_${TIMESTAMP}.log"

if [[ $? -ne 0 ]]; then
    log "✗ Stage C training failed"
    exit 1
fi
log "✓ Stage C training complete"

# Generate Stage C visualizations
log "Generating Stage C visualizations..."
"$PYTHON" training/visualize_stage_c.py \
    --run-dir artifacts/phase9_decoder_finetune/stageC_v2_benchmark \
    2>&1 | tee -a "$MAIN_LOG"

STAGEC_SUMMARY="artifacts/phase9_decoder_finetune/stageC_v2_benchmark/training_summary.txt"
if [[ -f "$STAGEC_SUMMARY" ]]; then
    log ""
    log "Stage C Summary:"
    cat "$STAGEC_SUMMARY" | tee -a "$MAIN_LOG"
fi

# ========================================================================
# Final Summary
# ========================================================================
log ""
log "======================================================================"
log "FULL PIPELINE COMPLETE"
log "======================================================================"
log ""
log "Results Location:"
log "  Stage A: artifacts/phase9_encoder_pretrain/stageA_v3/"
log "  Stage B: artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json"
log "  Stage C: artifacts/phase9_decoder_finetune/stageC_v2_benchmark/"
log ""
log "Visualizations:"
log "  Stage A graphs: artifacts/phase9_encoder_pretrain/stageA_v3/graphs/"
log "  Stage C graphs: artifacts/phase9_decoder_finetune/stageC_v2_benchmark/graphs/"
log ""
log "Main log: $MAIN_LOG"
log "======================================================================"

# Extract and display final BLEU score
FINAL_BLEU=$($PYTHON -c "
import json
with open('artifacts/phase9_decoder_finetune/stageC_v2_benchmark/history.json') as f:
    history = json.load(f)
    best_epoch = max(range(len(history)), key=lambda i: history[i]['bleu'])
    print(f'{history[best_epoch][\"bleu\"]:.4f}')
")

log ""
log "FINAL RESULT: Best BLEU-4 = $FINAL_BLEU"
log "Target: > 1.47 (iSign baseline)"

if (( $(echo "$FINAL_BLEU > 1.47" | bc -l) )); then
    log "✓ TARGET ACHIEVED!"
else
    log "Target not yet achieved. Review results and consider improvements."
fi

log "======================================================================"
