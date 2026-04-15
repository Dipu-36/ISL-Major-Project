#!/usr/bin/env bash
# run_stageC_v2.sh — Stage C v2: Multi-task translation training with WPP aux
#
# Prerequisites:
#   - run_stageB_v2.sh must have exited 0 (gate passed)
#   - artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt must exist
#   - artifacts/wpp_vocab.json must exist
#
# Training adds the WPP auxiliary loss to prevent encoder drift during
# translation fine-tuning.  T5 decoder is frozen for the first 5 epochs
# then the top-2 blocks are unfrozen.
#
# Outputs: artifacts/phase9_decoder_finetune/<RUN_NAME>/
#   best_bleu_model.pt  best_loss_model.pt  history.json
#
# Detached mode:
#   DETACH=1 RUN_NAME=stageC_v3_retry1 ./run_stageC_v2.sh
#
# In detached mode the script starts the Python job under nohup, writes
# logs/stageC_<RUN_NAME>_<timestamp>.log, stores the PID in
# artifacts/phase9_decoder_finetune/<RUN_NAME>/stagec.pid, and exits
# immediately so the job survives logout.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python"
CHECKPOINT="artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt"
GATE_FILE="artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json"
RUN_NAME="${RUN_NAME:-stageC_v2}"
DETACH="${DETACH:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/stageC_${RUN_NAME}_${TIMESTAMP}.log"
RUN_DIR="artifacts/phase9_decoder_finetune/${RUN_NAME}"
PID_FILE="${RUN_DIR}/stagec.pid"

mkdir -p "$RUN_DIR"
mkdir -p logs

# Check gate
if [[ ! -f "$GATE_FILE" ]]; then
  echo "[run_stageC_v2] ERROR: Stage B v2 gate file not found: $GATE_FILE"
  echo "[run_stageC_v2] Run ./run_stageB_v2.sh first."
  exit 1
fi

GATE_PASSED=$("$PYTHON" -c "import json; d=json.load(open('$GATE_FILE')); print(str(d.get('gate_pass', d.get('gate_passed', False))).lower())")
if [[ "$GATE_PASSED" != "true" ]]; then
  echo "[run_stageC_v2] ABORT — Stage B v2 gate has not passed."
  echo "[run_stageC_v2] Inspect $GATE_FILE and re-run Stage A."
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_stageC_v2] ERROR: Stage A checkpoint not found: $CHECKPOINT"
  exit 1
fi

CMD=(
  "$PYTHON" -u training/train_stage_c_v2.py
  --encoder-checkpoint "$CHECKPOINT"
  --run-name "$RUN_NAME"
  --batch-size "$BATCH_SIZE"
  --seed 42
  --num-workers "$NUM_WORKERS"
)

echo "[run_stageC_v2] Starting Stage C v2 (multi-task translation) ..."

if [[ "$DETACH" == "1" ]]; then
  echo "[run_stageC_v2] Launching detached run: $RUN_NAME"
  echo "[run_stageC_v2] Log file: $LOG_FILE"
  nohup env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$PID_FILE"
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[run_stageC_v2] ERROR: Detached process died immediately. Check $LOG_FILE"
    exit 1
  fi
  # disown may fail if job control is disabled; nohup is sufficient for survival.
  disown "$PID" 2>/dev/null || true
  echo "[run_stageC_v2] Detached PID: $PID"
  echo "[run_stageC_v2] PID file: $PID_FILE"
  echo "[run_stageC_v2] Monitor with: tail -f $LOG_FILE"
  exit 0
fi

PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "[run_stageC_v2] ✓ Stage C v2 COMPLETE"
  echo "[run_stageC_v2] Review BLEU in: ${RUN_DIR}/history.json"
  echo "[run_stageC_v2] If BLEU > 1.0, run Stage D: ./run_stageD.sh (update checkpoint path)"
else
  echo ""
  echo "[run_stageC_v2] ✗ Stage C v2 exited with error $EXIT_CODE"
fi

exit $EXIT_CODE
