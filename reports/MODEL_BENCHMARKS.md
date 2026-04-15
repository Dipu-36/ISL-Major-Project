# Model Benchmarks & Test Results

**Last Updated:** April 16, 2026

---

## Summary (At-a-glance)

- **WPP Avg AUC:** ≥ 0.58 — Achieved (v3 run)
- **Words ≥ 0.62 AUC:** ≥ 25 / 150 — Achieved (gate passed)
- **Mean L1 Isotropy:** ≤ 0.50 — Achieved
- **Best BLEU-4 (validation):** 1.39 (epoch 18) — current best (target: 1.47 baseline)
- **Best ROUGE-L (validation):** 10.26 (epoch 17)
- **Generic ratio:** ~0.10 (low hallucination)
- **Dominant ratio:** 0.00 (no mode collapse)

---

## Sources

- Hyperparameters and tuning notes: [reports/HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)
- Validation & evaluation protocol: [reports/MODEL_VALIDATION_PROTOCOL.md](MODEL_VALIDATION_PROTOCOL.md)
- Architecture snapshot: [ARCHITECTURE.md](ARCHITECTURE.md)
- Benchmark protocol & targets: [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)

---

## Detailed Metrics

- Stage A (WPP pretraining):
  - `avg_auc` target ≥ 0.58 — achieved (v3 run)
  - `words >= 0.62 AUC` target ≥ 25 — achieved
  - `mean_l1_isotropy` target ≤ 0.50 — achieved

- Stage C (translation):
  - Best validation BLEU-4: **1.39** (epoch 18)
  - Best validation ROUGE-L: **10.26** (epoch 17)
  - Note: target BLEU-4 to beat iSign baseline = **1.47** (see [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md))

---

## Artifact Locations (examples)

- Encoder pretrain artifacts: `artifacts/phase9_encoder_pretrain/` (look for run names `stageA_v3_*`)
- Decoder finetune artifacts: `artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/` (contains `best_bleu_model.pt`, `history.json`)
- Full evaluation outputs: `artifacts/final_test_results/`

(Use the `run_full_pipeline.sh` driver to reproduce end-to-end runs; see [ARCHITECTURE.md](ARCHITECTURE.md#appendix-quick-commands) and `run_full_pipeline.sh`.)

---

## Notes on Significance & Robustness

- Current differences (e.g., BLEU 1.39 vs target 1.47) are modest; apply bootstrap resampling (see [reports/MODEL_VALIDATION_PROTOCOL.md](MODEL_VALIDATION_PROTOCOL.md#comparison-protocol)) to test statistical significance before claiming parity with baseline.
- Checkpoint selection: prefer `best_bleu_model.pt` for reporting; include `best_loss_model.pt` as sanity check.

---

## How to Update This File

1. Run validation/evaluation after a completed training run.
2. Append new row(s) to the "Detailed Metrics" section with run name, epoch, and metrics.
3. Add artifact path(s) under "Artifact Locations" if different.

---

## Quick Commands (reproduce evaluation)

```bash
# Evaluate a final checkpoint on the test split and write full outputs
python evaluate_test_set.py \
  --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/best_bleu_model.pt \
  --split test \
  --num-samples all \
  --output-dir artifacts/final_test_results/

# Compute bootstrap p-value between two runs (example)
python tools/bootstrap_bleu_compare.py \
  --predA artifacts/runA/preds.json \
  --predB artifacts/runB/preds.json \
  --refs data/test_refs.json \
  --n 1000
```
