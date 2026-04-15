# Phase 3 Report - Current Project Situation, Codebase Analysis, and Benchmark Comparison

Date: 2026-04-13
Project: ISL Project (WPP pipeline: Stage A v3 -> Stage B v2 -> Stage C v2)

---

## 1) Executive Summary

The project has moved from representation collapse to a stable, measurable translation pipeline.

Current state:
- Stage A v3 (WPP encoder pretraining): successful and strongly above gate thresholds.
- Stage B v2 (probe gate): passed.
- Stage C v2 (translation + WPP auxiliary): completed on full train split with best BLEU-4 = 1.659 and best ROUGE-L = 11.10.

Headline benchmark verdict versus the provided iSign PDF (pose track):
- BLEU-4: beaten (1.659 vs 1.47 baseline, +12.9%).
- ROUGE-L: not yet beaten (11.10 vs 16.67 baseline, -33.4%).

So, the project has surpassed the pose baseline on BLEU, but has not yet surpassed the strongest reported ROUGE-L.

---

## 2) Evidence Collected from Current Artifacts

### 2.1 Stage A v3 (WPP pretraining)

Primary artifact:
- artifacts/phase9_encoder_pretrain/stageA_v3/final_probe.json

Observed results:
- avg_auc: 0.7062
- words_above_auc_0.60: 132
- words_above_auc_0.65: 114
- isotropy_l1: 0.2128
- gate_pass: true
- best_val_wpp_loss: 0.23098

Interpretation:
- Encoder now carries strong lexical discriminative signal.
- Isotropy is comfortably under collapse threshold.

### 2.2 Stage B v2 (formal gate)

Evidence from archived Stage B log:
- archive/logs/stageB/stageB_v2_20260408_205513.log

Observed gate summary:
- Words with AUC >= 0.62: 98/148 (need >= 25)
- Average AUC: 0.647
- Isotropy mean pairwise cosine: 0.227 (need <= 0.5)
- Gate decision: PASSED

Interpretation:
- Stage B v2 criteria were clearly passed.

### 2.3 Stage C v2 (translation)

Primary artifacts:
- artifacts/phase9_decoder_finetune/stageC_v2_bs8_stable1/training_summary.txt
- artifacts/phase9_decoder_finetune/stageC_v2_bs8_stable1/history.json

Observed results (full split run):
- Training samples: 103,276
- Validation samples: 12,240 (effective decoded val set in logs/history ~12,201 after filtering)
- Total epochs: 43
- Best BLEU-4: 1.659 (epoch 35)
- Final BLEU-4: 1.578
- Best ROUGE-L: 11.10
- Final ROUGE-L: 10.93
- Average dominant ratio: 0.023
- Average generic ratio: 0.119

Interpretation:
- Stage C is no longer in the prior severe hallucination regime.
- Decoder pathology metrics improved to acceptable internal thresholds.
- Translation overlap quality improved but remains below top ROUGE-L benchmark.

---

## 3) Comparison Against Experiment Matrix (reports/EXPERIMENT_MATRIX.md)

P1 status:
- P1-1 Build WPP vocab: DONE (artifacts/wpp_vocab.json present)
- P1-2 Stage A v3: DONE and PASS
  - avg_auc >= 0.58 condition met (0.7062)
  - words threshold condition met (formal Stage B log shows 98 words >= 0.62)
- P1-3 Stage B v2: DONE and PASS
- P1-4 Stage C v2: DONE and PASS for matrix criteria
  - BLEU >= 1.0 met (best 1.659)
  - generic_ratio < 0.3 met (avg 0.119; best epochs much lower)

P2 status:
- P2-1 full 103k Stage C run: DONE
- P2-2 Stage D joint fine-tuning: BLOCKED (run_stageD.sh not present in repo)
- P2-3 Stage A with 100k: effectively exceeded by current Stage A run on full 103,276

P3 ablations:
- Not fully executed in this latest chain.
- Remain useful for causal confirmation and publication-grade claims.

---

## 4) Comparative Analysis with Provided PDF

PDF used:
- reports/iSign A Benchmark for Indian Sign Language Processing.pdf

From the extracted baseline table (SignPose-to-Text):
- T5(base)+Mediapipe(75): BLEU-4 = 1.47, ROUGE-L = 16.67

### 4.1 Direct metric comparison (current best Stage C run)

| Metric | iSign pose baseline (paper) | Current project best | Delta | Verdict |
|---|---:|---:|---:|---|
| BLEU-4 | 1.47 | 1.659 | +0.189 (+12.9%) | Beaten |
| ROUGE-L | 16.67 | 11.10 | -5.57 (-33.4%) | Not beaten |

### 4.2 What this means

- Positive: The pipeline now clears a non-trivial BLEU threshold and exceeds the paper's reported pose BLEU baseline.
- Gap: ROUGE-L is still materially lower than the strongest paper baseline; lexical/sequence overlap quality remains the main gap.

### 4.3 Fairness caveats for benchmark claims

Before claiming final SOTA-like improvement, validate these are identical:
- exact split protocol,
- preprocessing settings,
- tokenization/evaluation setup,
- filtering of invalid/empty samples,
- checkpoint selection policy (best BLEU vs final).

Current evidence strongly supports "BLEU baseline beaten" but only "partial benchmark parity" overall.

---

## 5) Codebase Analysis - High-Impact Improvement Opportunities

### Priority 1 (consistency and correctness)

1. Unify gate threshold usage across code and docs.
- config/arch_v2_redesign.py defines gate AUC threshold = 0.62.
- training/train_stage_a_v3.py prints and computes internal gate at AUC >= 0.60.
- This mismatch can produce conflicting PASS/FAIL interpretations.

2. Fix Stage A launcher status message.
- run_stageA_v3.sh currently prints "Stage B gate PASSED" when Stage A exits 0.
- But Stage B probe is a separate step; message is misleading and can cause false confidence.

3. Standardize Stage B artifact schema.
- stage_b_v2_probe.json should consistently carry stage_b fields (gate_passed, words_above_auc_threshold, criteria block).
- Current file contents appear mixed with final_probe-style fields in some runs.

### Priority 2 (pipeline reliability)

4. Enforce hard gate by default in Stage C training script.
- training/train_stage_c_v2.py proceeds with warning if gate file is missing.
- For reproducible experiments, default should abort unless explicitly overridden by a flag (for diagnostics only).

5. Add missing launcher scripts referenced by docs/runbooks.
- run_stageB_v2.sh is referenced but missing at repo root.
- run_stageD.sh is referenced in reports and scripts but missing.
- This creates operational friction and increases manual error risk.

6. Update stale documentation/comments in code.
- train_stage_c_v2.py top comments mention "unfreeze top-2" while retry constants currently unfreeze only top-1.
- Keep code comments synchronized with active constants to avoid configuration drift.

### Priority 3 (metric improvement path to beat ROUGE-L)

7. Add decoding controls focused on hallucination suppression and coverage.
- Evaluate stronger constrained decoding settings (length penalty sweep, repetition penalties, bad phrase constraints from known attractors).
- Track impact specifically on ROUGE-L and semantic faithfulness.

8. Add signer-aware normalization experiment (already listed as diagnostic D4).
- Per-signer normalization may reduce style/noise variance and improve lexical alignment.

9. Run ablations with fixed protocol and publish a single consolidated scorecard.
- WPP aux on/off,
- decoder unfreeze depth (top-1 vs top-2),
- vocab size (150 vs 300),
- 2-layer vs 4-layer encoder.
- This will clarify what most improves ROUGE-L without sacrificing BLEU.

10. Add benchmark comparability guardrails.
- Emit sacrebleu signature, sample counts, split hash, and evaluation config into each run summary.
- This makes paper comparison auditable and publication-safe.

---

## 6) Risks and Open Questions

1. ROUGE-L gap risk:
- Model may still over-index on frequent news-style templates while missing exact content overlap.

2. Comparability risk:
- If paper evaluation protocol differs subtly, direct claims may be contested.

3. Operational risk:
- Missing Stage B/Stage D launchers and inconsistent gate thresholds can cause accidental invalid runs.

---

## 7) Recommended Immediate Next Steps

1. Code hygiene sprint (1 day):
- align gate threshold to 0.62 everywhere,
- fix launcher messaging,
- add run_stageB_v2.sh and run_stageD.sh,
- harden Stage C gate enforcement with explicit override flag.

2. Metric sprint (2-3 runs):
- run a controlled decoding/aux-weight/unfreeze-depth sweep,
- prioritize ROUGE-L lift while preserving BLEU >= 1.47.

3. Benchmark claim package:
- generate one final comparison table including BLEU, ROUGE-L, and evaluation signatures,
- state claim scope explicitly: "pose BLEU beaten; ROUGE-L not yet beaten" unless new run changes this.

---

## 8) Final Status Statement (as of 2026-04-13)

- Core redesigned pipeline is functioning and no longer in representation collapse.
- P1 goals are achieved according to internal experiment criteria.
- Compared to the provided iSign benchmark PDF, BLEU has been beaten on the pose track baseline, but ROUGE-L has not yet been beaten.
- The project is in a strong "partial benchmark breakthrough" phase, with ROUGE-L optimization now the main frontier.
