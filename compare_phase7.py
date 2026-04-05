#!/usr/bin/env python3
"""Phase 7 comparison and analysis script.

Usage (after Phase 7 training completes):
    python compare_phase7.py

Reads Phase 6 and Phase 7 report.json files and prints:
  - Metrics table (before vs after)
  - Training curve summary (per-epoch progression)
  - 10 sample prediction examples (last epoch)
  - Outcome classification (A / B / C)
  - Next-step recommendation
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PHASE6_REPORT = Path(
    "artifacts/phase6_50k/ce_extended_50000train_e20_seed42/report.json"
)
PHASE7_REPORT = Path(
    "artifacts/phase7_posenc_keyframes/ce_extended_50000train_e20_seed42/report.json"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_report(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def last_epoch(report: dict[str, Any]) -> dict[str, Any]:
    return report["history"][-1]


def best_rouge_epoch(report: dict[str, Any]) -> dict[str, Any]:
    return max(report["history"], key=lambda e: e["rouge_l"])


def best_bleu_epoch(report: dict[str, Any]) -> dict[str, Any]:
    return max(report["history"], key=lambda e: e["bleu"])


def fmt_delta(before: float, after: float, higher_is_better: bool = True) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    direction = "↑" if (delta > 0) == higher_is_better else ("↓" if delta != 0 else "→")
    return f"{sign}{delta:.4f} {direction}"


def fmt_delta2(before: float, after: float, higher_is_better: bool = True) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    direction = "↑" if (delta > 0) == higher_is_better else ("↓" if delta != 0 else "→")
    return f"{sign}{delta:.2f} {direction}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PHASE6_REPORT.exists():
        print(f"ERROR: Phase 6 report not found at {PHASE6_REPORT}")
        sys.exit(1)
    if not PHASE7_REPORT.exists():
        print(f"ERROR: Phase 7 report not found at {PHASE7_REPORT}")
        print("Training may still be in progress. Check artifacts/phase7_posenc_keyframes/run.log")
        sys.exit(1)

    r6 = load_report(PHASE6_REPORT)
    r7 = load_report(PHASE7_REPORT)

    e6_last = last_epoch(r6)
    e7_last = last_epoch(r7)
    e6_best_rouge = best_rouge_epoch(r6)
    e7_best_rouge = best_rouge_epoch(r7)
    e6_best_bleu = best_bleu_epoch(r6)
    e7_best_bleu = best_bleu_epoch(r7)

    ckpt6 = r6.get("checkpoint", {})
    ckpt7 = r7.get("checkpoint", {})

    # ── Table 1: Final-epoch comparison ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("TABLE 1: Metrics Comparison — Final Epoch (Phase 6 → Phase 7)")
    print("=" * 72)
    hdr = f"{'Metric':<32}  {'Phase 6':>10}  {'Phase 7':>10}  {'Delta':>18}"
    print(hdr)
    print("-" * 72)

    rows = [
        ("val_loss",              e6_last["val_loss"],                   e7_last["val_loss"],                   False),
        ("BLEU (final epoch)",    e6_last["bleu"],                       e7_last["bleu"],                       True),
        ("ROUGE-L (final epoch)", e6_last["rouge_l"],                    e7_last["rouge_l"],                    True),
        ("CHRF (final epoch)",    e6_last["chrf"],                       e7_last["chrf"],                       True),
        ("BLEU (best epoch)",     e6_best_bleu["bleu"],                  e7_best_bleu["bleu"],                  True),
        ("ROUGE-L (best epoch)",  e6_best_rouge["rouge_l"],              e7_best_rouge["rouge_l"],              True),
        ("dominant_ratio",        e6_last["dominant_prediction_ratio"],  e7_last["dominant_prediction_ratio"],  False),
        ("unique_ratio",          e6_last["unique_prediction_ratio"],    e7_last["unique_prediction_ratio"],    True),
        ("generic_ratio",         e6_last["generic_sentence_ratio"],     e7_last["generic_sentence_ratio"],     False),
        ("avg_output_len",        e6_last["avg_output_length"],          e7_last["avg_output_length"],          None),
        ("avg_target_len",        e6_last["avg_target_length"],          e7_last["avg_target_length"],          None),
        ("input_diff_ratio",      e6_last["input_difference_ratio"],     e7_last["input_difference_ratio"],     True),
    ]

    for label, v6, v7, hib in rows:
        if hib is None:
            delta_str = f"{v7 - v6:+.2f}"
        elif isinstance(v6, float) and abs(v6) < 1.0 and abs(v7) < 1.0:
            delta_str = fmt_delta(v6, v7, hib)
        else:
            delta_str = fmt_delta2(v6, v7, hib)
        print(f"  {label:<30}  {v6:>10.4f}  {v7:>10.4f}  {delta_str:>18}")

    # ── Table 2: ROUGE-L plateau check ────────────────────────────────────────
    print()
    print("=" * 72)
    print("TABLE 2: ROUGE-L Trajectory (Phase 7 — all epochs)")
    print("=" * 72)
    print(f"  {'Epoch':>6}  {'ROUGE-L':>8}  {'BLEU':>6}  {'val_loss':>10}  {'unique':>8}")
    print("  " + "-" * 44)
    prev_rl = 0.0
    for ep in r7["history"]:
        marker = " *** NEW BEST" if ep["rouge_l"] > prev_rl else ""
        print(f"  {ep['epoch']:>6}  {ep['rouge_l']:>8.2f}  {ep['bleu']:>6.4f}"
              f"  {ep['val_loss']:>10.4f}  {ep['unique_prediction_ratio']:>8.3f}{marker}")
        if ep["rouge_l"] > prev_rl:
            prev_rl = ep["rouge_l"]

    # ── Sample predictions (last epoch) ──────────────────────────────────────
    sample_pred_epochs = r7.get("sample_predictions", [])
    if sample_pred_epochs:
        last_pred_epoch = sample_pred_epochs[-1]
        examples = last_pred_epoch.get("examples", [])
        print()
        print("=" * 72)
        print(f"TABLE 3: 10 Sample Predictions — Epoch {last_pred_epoch['epoch']}")
        print("=" * 72)
        for i, ex in enumerate(examples[:10]):
            generic_tag = " [GENERIC]" if ex.get("is_generic") else ""
            print(f"  [{i}] REF : {ex.get('reference', '')}")
            print(f"       PRED: {ex.get('prediction', '')}{generic_tag}")
            print()

        # Qualitative analysis
        generic_count = sum(1 for ex in examples[:10] if ex.get("is_generic"))
        print(f"  Generic predictions: {generic_count}/10")
        print()
        preds = [ex.get("prediction", "") for ex in examples[:10]]
        unique_preds = len(set(preds))
        print(f"  Unique predictions : {unique_preds}/10")

    # ── Outcome classification ────────────────────────────────────────────────
    rouge_delta = e7_best_rouge["rouge_l"] - e6_best_rouge["rouge_l"]
    bleu_delta = e7_best_bleu["bleu"] - e6_best_bleu["bleu"]
    rouge_plateau_broken = e7_best_rouge["rouge_l"] > ckpt6.get("best_rouge_l", e6_best_rouge["rouge_l"]) + 0.5
    bleu_significant = bleu_delta > 0.02

    print()
    print("=" * 72)
    print("OUTCOME CLASSIFICATION")
    print("=" * 72)
    print(f"  Phase 6 best ROUGE-L : {e6_best_rouge['rouge_l']:.2f}")
    print(f"  Phase 7 best ROUGE-L : {e7_best_rouge['rouge_l']:.2f}  (delta {rouge_delta:+.2f})")
    print(f"  Phase 6 best BLEU    : {e6_best_bleu['bleu']:.4f}")
    print(f"  Phase 7 best BLEU    : {e7_best_bleu['bleu']:.4f}  (delta {bleu_delta:+.4f})")
    print(f"  ROUGE-L plateau broken: {'YES' if rouge_plateau_broken else 'NO'}")
    print(f"  BLEU significant gain : {'YES' if bleu_significant else 'NO'}")
    print()

    if rouge_delta > 1.0 and bleu_delta > 0.02:
        outcome = "A"
        label = "STRONG IMPROVEMENT"
    elif rouge_delta > 0.3 or bleu_delta > 0.01:
        outcome = "B"
        label = "MODERATE IMPROVEMENT"
    else:
        outcome = "C"
        label = "NO IMPROVEMENT / REGRESSION"

    print(f"  >>> OUTCOME: ({outcome}) {label}")
    print()

    # ── Next-step recommendation ──────────────────────────────────────────────
    print("=" * 72)
    print("NEXT STEP RECOMMENDATION")
    print("=" * 72)

    if outcome == "A":
        print("""
  Strong improvement confirmed. Positional encoding + keyframe selection
  are contributing meaningfully to translation quality.

  Recommended next steps (in order):
    1. Increase encoder depth: num_layers 2 → 4
         model/temporal_encoder.py: TemporalVisualEncoder(num_layers=4)
    2. Add semantic grounding loss (λ=0.2):
         --semantic-lambda 0.2  (with --use-warmup)
    3. Add contrastive loss (weight=0.1):
         --contrastive-weight 0.1
    Run each addition in isolation first to identify its contribution.
""")
    elif outcome == "B":
        print("""
  Moderate improvement. The structural fixes are helping but gains are
  limited — the model likely needs more capacity or better loss shaping.

  Recommended next steps:
    1. Tune keyframe selection alpha:
         Try exp_smoothing_alpha in {0.3, 0.7} — current is 0.5.
    2. Add semantic loss (λ=0.1) to improve semantic alignment:
         --semantic-lambda 0.1
    3. Verify positional encoding is not collapsing:
         Add a diagnostic: print std of pe offsets before/after training.
    4. Consider increasing d_model from 256 → 512 in TemporalVisualEncoder.
""")
    else:
        print("""
  No improvement detected. The preprocessing or architecture changes may
  not be integrating correctly.

  Diagnostic steps:
    1. Verify positional encoding is being applied:
         Add assertion: check that TemporalVisualEncoder.pos_enc.pe is
         non-zero and has the expected shape [1, 512, 256].
    2. Check encoder output distribution:
         Log mean/std of encoded features per epoch.
    3. Inspect attention patterns:
         Examine whether attention weights show temporal structure
         (nearby frames attending to each other) vs flat/uniform.
    4. Verify keyframe sampling is actually selecting different frames:
         Compare selected_idx distributions across samples.
    5. Revert exp_smoothing_alpha=0.5 → try 0.8 (less smoothing).
    6. Check that the checkpoint loading did not corrupt the model state.
""")

    print("=" * 72)


if __name__ == "__main__":
    main()
