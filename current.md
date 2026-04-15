# ISL Project Current State (April 15, 2026)

## 1) Project Goal
Build an Indian Sign Language (ISL) pose-to-English translation system that beats the iSign benchmark baseline.

Current benchmark target to beat:
- SignPose2Text (T5-base + MediaPipe-75): BLEU-4 = 1.47, ROUGE-L = 16.67

## 2) What Has Been Done So Far

### A. Root-cause investigation completed
The team traced low BLEU/hallucination behavior across older phases and concluded:
- The issue was not only a late-phase training instability.
- The core failure mode was weak encoder lexical grounding, allowing decoder language-prior takeover.

Main analysis docs:
- reports/ROOT_CAUSE_ANALYSIS.md
- reports/PIPELINE_REDESIGN.md
- reports/handoff.md

### B. Pipeline redesign implemented (Phase 9 path)
The redesigned flow is now in place:
1. Stage A v3: encoder pretraining with WPP (Word Presence Prediction).
2. Stage B v2 gate: AUC-based gate on lexical signal.
3. Stage C v2: translation fine-tuning with ongoing WPP auxiliary loss.

Status:
- Stage B gate passed.
- Stage C v2 is running on top of Stage A-pretrained encoder.

### C. Stage C engineering improvements already applied
In Stage C training code, previous reliability/perf fixes were added, including:
- Better progress visibility during long phases.
- Safer restart behavior and history schema updates.
- More robust handling around grad accumulation logging and helper scope issues.

## 3) Current Active Training Situation

Active run:
- Name: stageC_v2_bs8_safe_restart
- Log: logs/stageC_stageC_v2_bs8_safe_restart_20260413_213408.log
- Artifacts dir: artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart
- Decoder backbone: t5-small (from config/arch_v2_redesign.py)

Observed state at time of writing:
- Process is still alive (PID recorded in artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/stagec.pid and running).
- history.json currently contains 20 completed epochs.
- Log shows validation progress continuing after epoch 20 (next epoch in progress).

Latest completed epoch snapshot (epoch 20):
- Train CE: 3.6354
- Val CE: 3.6705
- Train WPP aux: 0.1831
- BLEU: 1.31
- ROUGE-L: 10.2
- Dominant prediction ratio: 0.00
- Generic ratio: 0.104
- Aux weight: 0.505

Best-so-far within current history.json:
- Best BLEU epoch: 18
- Best BLEU: 1.39
- ROUGE-L at best-BLEU epoch: 10.044
- Best ROUGE-L epoch: 17
- Best ROUGE-L: 10.259

Checkpoint files already present:
- best_bleu_model.pt
- best_loss_model.pt
- history.json

## 4) Current Interpretation
- Training is much more stable than older collapse/hallucination-heavy phases.
- BLEU is near the benchmark floor but still below the 1.47 target.
- ROUGE-L remains well below the 16.67 benchmark target.
- Sample outputs still include generic fluent sentences that are often semantically incorrect.

So, this run is a meaningful improvement over earlier pathological behavior, but not yet a benchmark-beating final model.

## 5) Known Risks / Caveats To Keep In Mind
1. Preprocessing masking risk
- In preprocessing/pipeline.py, short-clip padding behavior may still mark padded frames as valid in attention mask (known repo note).
- This can leak synthetic repeated-frame signal into training.

2. Metric interpretation risk
- Some older scripts/runs used non-benchmark-comparable subsets or decode limits.
- For final claims, ensure full validation decoding and consistent split/eval settings.

3. Decoder prior risk remains
- Even with WPP auxiliary loss, generic fluent outputs can still appear.
- Continue monitoring dominant_ratio, generic_ratio, and sample predictions each epoch.

## 6) What The Next Agent Should Do Immediately

### If current Stage C run is still active
1. Do not interrupt unless clearly stalled.
2. Wait for run completion and full history update.
3. Select candidate checkpoint primarily by BLEU, then inspect qualitative sample outputs.

### After completion (mandatory triage)
1. Generate plots for this run:
- python training/visualize_stage_c.py --run-dir artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart
2. Compare this run against a prior Stage C run (example):
- python training/compare_stage_c_runs.py --run1 artifacts/phase9_decoder_finetune/stageC_v2_bs8_stable1 --run2 artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart
3. Produce a short promote/not-promote decision for Stage D based on:
- BLEU and ROUGE-L trend
- Generic/dominant ratios
- Sample prediction quality from history.json (sample_preds_detailed)

### Promotion rule of thumb
Promote to Stage D only if this run shows clear quality gain over previous Stage C runs on both metrics and qualitative grounding.

## 7) Key Paths For Fast Onboarding
- reports/handoff.md
- reports/ROOT_CAUSE_ANALYSIS.md
- reports/PIPELINE_REDESIGN.md
- config/arch_v2_redesign.py
- training/train_stage_c_v2.py
- artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/history.json
- logs/stageC_stageC_v2_bs8_safe_restart_20260413_213408.log

## 8) One-line Status
Project is in late Stage C (WPP-regularized translation) with active training and partial recovery from prior failure modes, but has not yet surpassed benchmark-level translation quality.
