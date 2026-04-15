# ISL Translation Project — Saturday Presentation Report

**Project:** Pose-based Indian Sign Language to English Translation  
**Date:** April 8, 2026  
**Prepared for:** Interim technical presentation  
**Current status:** Root cause isolated, redesigned pipeline implemented, Stage A and Stage B passed, Stage C first run partially successful, Stage C retry currently running.

---

## 1. Executive Summary

This work focused on fixing the real root issue in the training pipeline rather than continuing to optimize a failing setup.

The main finding is that the original Stage A objective was wrong for this dataset. The previous encoder pretraining strategy tried to align noisy pose clips directly with frozen T5 sentence embeddings using sentence-level contrastive learning. That setup produced geometrically spread representations in v2, but still failed to learn semantic content. In other words, the encoder stopped collapsing but still did not become meaning-aware.

To address that, the pipeline was redesigned around **Word Presence Prediction (WPP)**. Instead of forcing the encoder to solve full sentence-level cross-modal alignment too early, the new Stage A teaches the encoder a more local and tractable objective: predict which content words are present in the clip. This redesign succeeded in Stage A and Stage B.

The translation stage has improved compared with the old lineage, but it is not yet clean enough to claim success. The first Stage C run achieved a measurable reduction in collapse and generic output frequency, and BLEU crossed the minimal numerical gate, but it still generated too many generic or hallucinated sentences. Therefore, Stage C is currently classified as **partial success**, not final success, and a stricter retry run is in progress.

This is a strong interim result for a Saturday presentation because it shows:

1. A clear diagnosis of why the old pipeline failed.
2. A principled redesign that is implemented, not hypothetical.
3. Concrete evidence that semantic signal can now be learned at the representation level.
4. Honest reporting that full translation quality is not solved yet.

---

## 2. What Was Done

### 2.1 Audit and Cleanup

The project tree was audited to separate active components from stale lineage artifacts.

What was cleaned:

1. Obsolete Phase 7 and Phase 8 launchers were archived into `archive/old_launchers/`.
2. Redundant, empty, and superseded log files were moved into `archive/logs/`.
3. A dedicated `reports/` directory was created for structured reports.
4. Existing legacy checkpoints remained preserved in `_backup_legacy/` for lineage comparison.

What was intentionally preserved:

1. Existing Stage A v1 and v2 runs for controlled comparison.
2. Existing ablation outputs and diagnostic reports.
3. Existing preprocessing, model, and training utilities that remained useful.

Reason for this cleanup:

The goal was not cosmetic cleanup. The goal was to create a reproducible working space where each new experiment has a clean run directory, explicit configuration, clear logs, and traceable comparison against prior results.

---

### 2.2 Root Cause Analysis

The original problem was not just representation collapse.

The full diagnosis showed six interacting issues:

1. **Sentence-level alignment noise** in iSign means a pose clip and its sentence label are not always a clean positive pair.
2. **Frozen T5 sentence embeddings** are too abstract and too hard a target for a small visual encoder trained on noisy pose data.
3. **No lexical intermediate supervision** means the encoder is never directly rewarded for recognizing specific sign content.
4. **Stage A v2 fixed geometry but not semantics**. This proved that collapse was only a symptom.
5. **Bridge-level alignment remained near random chance**, which means the encoder still was not generating useful content-aware features for the decoder.
6. **The decoder language-model prior still dominated**, producing fluent but generic or hallucinated outputs.

This led to the key design decision:

> Stop asking the encoder to solve full sentence-level alignment before it learns lexical structure.

---

### 2.3 Pipeline Redesign

The redesigned pipeline has four stages:

1. **Stage A v3:** encoder pretraining with WPP
2. **Stage B v2:** formal probe gate based on AUC and isotropy
3. **Stage C v2 / retry:** translation training with WPP auxiliary preservation
4. **Stage D:** deferred until Stage C is semantically reliable

The key architectural change is that Stage A no longer uses T5 at all. Instead, the encoder is trained as a self-contained lexical predictor.

---

## 3. Full Pipeline Overview

### 3.1 Input and Preprocessing Pipeline

The input is a pose sequence from `.pose` files.

#### Raw pose structure

Per frame, the pipeline extracts:

1. Body: 33 joints
2. Left hand: 21 joints
3. Right hand: 21 joints

Total = 75 joints × 3 coordinates = 225 raw features per frame.

#### Preprocessing steps

Implemented in `preprocessing/pipeline.py`:

1. **Confidence-weighted joint masking**
   - Uses pose detector confidence
   - Confidence floor = `0.2`
   - Low-confidence joints are down-weighted, not hard-zeroed

2. **Signing-space normalization**
   - Center joints using shoulder midpoint
   - Normalize scale by shoulder width
   - Makes pose less sensitive to signer size and camera framing

3. **Temporal smoothing**
   - Exponential moving average
   - `alpha = 0.5`

4. **Motion-based keyframe sampling**
   - Retains `96` frames per sample
   - Uses frame-level motion magnitude to keep informative frames
   - Always keeps the first and last frame

5. **Velocity augmentation**
   - Adds frame-to-frame velocity features
   - Final feature dimension per frame = `450`
     - `225` position
     - `225` velocity

This preprocessing is important to mention in the presentation because the model is not trained on raw video. The entire system depends on a structured pose representation.

---

## 4. Model Architecture

### 4.1 Stage A v3 Encoder

Configured in `config/arch_v2_redesign.py` and implemented in `training/train_stage_a_v3.py`.

#### Encoder architecture

1. Input dimension: `450`
2. Temporal encoder: `TemporalVisualEncoder`
3. Number of layers: `4`
4. Hidden dimension (`d_model`): `256`
5. Attention heads: `8`
6. Feedforward dimension: `1024`
7. Dropout: `0.15`

#### WPP head

1. Mean-pool encoder output over valid frames
2. Linear classification head: `Linear(256, 150)`
3. Multi-label output over top-150 content words

#### Stage A parameter count

Observed in training log:

1. Trainable parameters: `3,313,558`

---

### 4.2 Stage C Translation Model

Implemented in `training/train_stage_c_v2.py`.

#### Components

1. Same WPP-trained `TemporalVisualEncoder`
2. Bridge layer:
   - `Linear(256 → 512)`
   - `GELU`
   - `LayerNorm(512)`
3. Decoder: `t5-small`
4. Auxiliary head: retained WPP head on encoder pooled output

#### Stage C model behavior

1. The encoder provides temporal pose features.
2. The bridge maps encoder tokens into T5 hidden space.
3. T5 decoder attends over bridged visual tokens.
4. WPP auxiliary loss keeps the encoder grounded in lexical content during translation training.

#### Stage C initial trainable parameters

Observed in the run log:

1. Initial trainable parameters: `3,446,166`

This is important: most of T5 remains frozen at the start, so the model is not simply free to fall back to the decoder language prior immediately.

---

## 5. Training Objectives and Losses

### 5.1 Stage A v3 Objective

#### Main loss

**WPP multi-label classification**

1. Loss: `BCEWithLogitsLoss`
2. Output size: `[B, 150]`
3. Positive class weighting: `pos_weight = 10.0`

Reason:

Positive word labels are sparse, so the loss up-weights positives to avoid a trivial always-negative predictor.

#### Regularization loss

**Soft VICReg on pooled encoder features**

1. Variance penalty weight: `0.25`
2. Covariance penalty weight: `0.02`
3. Target standard deviation (`gamma`): `1.0`

Reason:

Full-strength VICReg in Stage A v2 fixed collapse but destroyed useful coarse structure. The new version is intentionally weaker.

---

### 5.2 Stage C Objective

#### Main loss

**Translation cross-entropy from T5**

This uses T5’s normal sequence-to-sequence loss over target tokens.

#### Auxiliary loss

**WPP auxiliary preservation loss**

1. Same `BCEWithLogitsLoss`
2. Same positive weighting = `10.0`
3. Applied on encoder pooled features during translation training

#### Combined Stage C loss

$$
L = L_{CE} + \lambda_{WPP} \cdot L_{WPP}
$$

This is the key mechanism meant to stop the decoder from ignoring the encoder.

---

## 6. Hyperparameters

### 6.1 Shared / Architecture Hyperparameters

| Parameter | Value |
|---|---:|
| `SRC_DIM` | 450 |
| `TARGET_FRAMES` | 96 |
| `ENCODER_LAYERS` | 4 |
| `D_MODEL` | 256 |
| `NHEAD` | 8 |
| `DIM_FEEDFORWARD` | 1024 |
| `ENCODER_DROPOUT` | 0.15 |
| `T5_NAME` | t5-small |
| `BRIDGE_OUT` | 512 |

### 6.2 Stage A v3 Hyperparameters

| Parameter | Value |
|---|---:|
| `WPP_VOCAB_SIZE` | 150 |
| `WPP_MIN_WORD_FREQ` | 50 |
| `WPP_LR` | 3e-4 |
| `WPP_LR_MIN` | 1e-6 |
| `WPP_WARMUP_EPOCHS` | 3 |
| `WPP_MAX_EPOCHS` | 60 |
| `WPP_EARLY_STOP_PATIENCE` | 15 |
| `WPP_BATCH_SIZE` | 128 |
| `WPP_WEIGHT_DECAY` | 1e-2 |
| `FRAME_DROPOUT_RATE` | 0.15 |
| `JOINT_NOISE_STD` | 0.01 |
| `VICREG_COEFF_VAR` | 0.25 |
| `VICREG_COEFF_COV` | 0.02 |
| `VICREG_GAMMA` | 1.0 |

### 6.3 Stage B v2 Gate Thresholds

| Criterion | Threshold |
|---|---:|
| `GATE_MIN_AUC` | 0.62 |
| `GATE_MIN_WORDS_ABOVE_AUC` | 25 |
| `GATE_MAX_L1_ISOTROPY` | 0.50 |

### 6.4 Stage C v2 First Run Hyperparameters

Configured from `config/arch_v2_redesign.py` and the first implementation state:

| Parameter | Value |
|---|---:|
| Encoder LR | 1e-4 |
| Decoder LR | 1e-5 |
| WPP auxiliary init weight | 0.5 |
| WPP auxiliary end weight | 0.05 |
| WPP auxiliary schedule length | 30 epochs |
| Frozen decoder warmup | 5 epochs |
| Unfrozen decoder blocks | 2 |
| Weight decay | 1e-2 |
| Patience | 10 |

### 6.5 Stage C Retry Hyperparameters

These were introduced after the first Stage C run was classified as partial success.

| Parameter | Retry value |
|---|---:|
| Frozen decoder warmup | 8 epochs |
| Patience | 8 |
| Unfrozen decoder blocks | 1 |
| Decoder LR | 5e-6 |
| WPP auxiliary init weight | 0.8 |
| WPP auxiliary end weight | 0.2 |
| WPP auxiliary schedule length | 40 epochs |
| Decoder `lm_head` | kept frozen |

Reason for retry changes:

The first Stage C run improved metrics but still showed decoder-prior hallucination. The retry is designed to make decoder adaptation slower and lexical preservation stronger.

---

## 7. Optimizers and Training Mechanics

### 7.1 Stage A Optimizer

Optimizer: **AdamW**

Parameter groups:

1. Encoder parameters
   - LR = `3e-4`
   - Weight decay = `1e-2`
2. Classification head
   - LR = `3e-4`
   - Weight decay = `1e-3`

Other mechanics:

1. Gradient clipping = `1.0`
2. LR schedule = linear warmup then cosine decay
3. Early stopping on validation WPP loss

### 7.2 Stage C Optimizer

Optimizer: **AdamW**

Parameter groups in the current retry script:

1. Encoder group
   - LR = `1e-4`
   - Weight decay = `1e-2`
2. Bridge group
   - LR = `1e-4`
   - Weight decay = `1e-2`
3. WPP head group
   - LR = `5e-5`
   - Weight decay = `1e-3`
4. Decoder group
   - LR = `5e-6` in retry
   - Weight decay = `1e-2`

Other mechanics:

1. Gradient clipping = `1.0`
2. Separate encoder/decoder LR scheduling
3. Decoder remains fully frozen for the warmup period
4. Decoder is unfrozen gradually later

---

## 8. Results So Far

### 8.1 Legacy Stage A Failure Summary

#### Stage A v1

1. Collapse problem present
2. Stage B gate failed
3. Semantic content remained effectively random

#### Stage A v2

1. Geometric collapse fixed
2. Semantic content still failed
3. Stage B gate still failed
4. Bridge retrieval remained near random chance

Main conclusion:

> v2 proved that fixing geometry alone does not solve semantic learning.

---

### 8.2 Raw-Feature Baseline Probe

A quick raw-feature pilot was run before trusting the new gate thresholds.

Pilot setup:

1. 4k train samples
2. 1k validation samples
3. Logistic regression on mean-pooled raw cached pose features
4. Top-20 WPP words

Pilot result:

1. Average AUC = `0.692`
2. `13/20` words at AUC ≥ `0.62`
3. `15/20` words at AUC ≥ `0.60`

This is important for the presentation because it shows the redesigned task is not arbitrary. There is measurable lexical signal already present in the raw pose feature space.

---

### 8.3 Stage A v3 Result

From `artifacts/phase9_encoder_pretrain/stageA_v3/final_probe.json`:

1. Best validation WPP loss = `0.22866`
2. Average AUC = `0.6583`
3. Words with AUC ≥ `0.60` = `98`
4. Words with AUC ≥ `0.65` = `71`
5. Isotropy = `0.2269`
6. Gate pass = `True`

Interpretation:

Stage A v3 was a **clear success**. The encoder learned real lexical signal and passed the semantic gate that all previous versions failed.

---

### 8.4 Stage B v2 Result

From `artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json`:

1. Gate passed = `True`
2. Words above AUC threshold = `98/148`
3. Average AUC = `0.6471`
4. Retrieval R@1 = `0.6%`
5. Random baseline = `0.2%`
6. Retrieval lift = `3.0×`

Interpretation:

Stage B formally validates that the encoder is now meaning-aware enough to justify translation training.

---

### 8.5 Stage C First Run Result

Stage C first run: `artifacts/phase9_decoder_finetune/stageC_v2/`

#### Best BLEU checkpoint

1. Epoch = `25`
2. BLEU = `1.024`
3. ROUGE-L = `8.609`
4. Dominant ratio = `0.016`
5. Generic ratio = `0.148`
6. Validation CE = `3.69286`

#### Best validation CE checkpoint

1. Epoch = `21`
2. BLEU = `0.756`
3. ROUGE-L = `7.845`
4. Dominant ratio = `0.018`
5. Generic ratio = `0.121`
6. Validation CE = `3.68607`

#### Honest interpretation

Stage C first run is classified as **partial success**.

Why it is better than the old pipeline:

1. BLEU crossed the minimal gate of `1.0`
2. Dominant ratio became low
3. Generic ratio dropped substantially compared with the old hallucination regime

Why it is not final success:

1. Sample outputs remained generic and hallucinated
2. Several predictions still used unstable political/news templates
3. Qualitative grounding was still too weak to justify Stage D

So the correct research decision was not to proceed to Stage D, but to tighten Stage C.

---

### 8.6 Stage C Retry Status

Current retry run: `artifacts/phase9_decoder_finetune/stageC_v3_retry1/`

Current status at the time of this report:

1. The retry run is active
2. Early epochs have been written
3. As expected, before decoder unfreezing the model is still weak and too early to judge

Latest available retry status:

1. Epoch = `3`
2. Validation CE = `4.24081`
3. BLEU = `0.196`
4. Generic ratio = `0.326`

This should **not** be over-interpreted yet, because the decoder in the retry run remains frozen until epoch `8`.

---

## 9. Important Scripts and Artifacts

### Core scripts

1. `training/word_inventory.py`
2. `training/train_stage_a_v3.py`
3. `probe_stage_b_v2.py`
4. `training/train_stage_c_v2.py`
5. `run_stageA_v3.sh`
6. `run_stageB_v2.sh`
7. `run_stageC_v2.sh`

### Important artifacts

1. `artifacts/wpp_vocab.json`
2. `artifacts/phase9_encoder_pretrain/stageA_v3/final_probe.json`
3. `artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json`
4. `artifacts/phase9_decoder_finetune/stageC_v2/history.json`
5. `artifacts/phase9_decoder_finetune/stageC_v3_retry1/history.json`

---

## 10. What To Show in the Saturday Presentation

This should be treated as an **interim research presentation**, not the final project presentation.

That means the goal is not to present the entire codebase or every experiment. The goal is to show:

1. the problem you found,
2. the reasoning that led to the redesign,
3. the evidence that the redesign is materially better,
4. the honest remaining gap.

### Recommended presentation scope

Show roughly **60% to 70% of the full project story**, not 100%.

That is enough to demonstrate serious progress without spending your final presentation material too early.

### What you should definitely show

#### 1. Problem framing

Explain the task in one slide:

1. ISL pose-to-text translation
2. Input = pose sequence, not raw video
3. Problem = semantic failure, not just low BLEU

#### 2. Failure diagnosis from the old pipeline

Show one slide with evidence that:

1. v1 failed semantically
2. v2 fixed collapse but still failed semantics
3. bridge retrieval remained near random
4. therefore the issue was objective mismatch, not just geometry

#### 3. Root-cause explanation

Show one slide explaining:

1. sentence-level alignment noise
2. mismatch with frozen T5 sentence embeddings
3. no lexical intermediate supervision

This is probably the most important intellectual contribution to present.

#### 4. Redesigned pipeline

Show one architecture slide:

1. preprocessing
2. Stage A v3 WPP encoder
3. Stage B v2 formal gate
4. Stage C with WPP auxiliary

Keep it conceptual, not code-heavy.

#### 5. Hard results

Show a results slide with these numbers:

1. Raw-feature pilot average AUC = `0.692`
2. Stage A v3 average AUC = `0.658`
3. Stage B v2 passed with `98/148` words above threshold
4. Stage C first run = partial success, BLEU just above `1.0`, generic ratio reduced but hallucination not solved

#### 6. Honest status slide

State clearly:

1. Representation learning problem is substantially improved
2. Translation is improved but not solved
3. Retry run is in progress to reduce decoder prior dominance

This will make the presentation credible.

---

### What you should avoid showing now

Do **not** spend time on:

1. full file-tree walkthrough
2. every archived script or log
3. every ablation detail
4. every hyperparameter unless specifically asked
5. final benchmark claims against the full paper result
6. polished demo claims if qualitative outputs are still unstable

Those belong to the final presentation a month later.

---

## 11. Suggested Slide Structure

If you want a compact and effective presentation, use roughly 8 slides.

### Slide 1 — Title and objective

One-sentence objective:

> Build a pose-based ISL-to-English system that learns semantic signal before translation.

### Slide 2 — Why the original pipeline failed

1. v1 semantic failure
2. v2 fixed collapse but still failed Stage B
3. therefore root cause was deeper than geometry

### Slide 3 — Root cause

1. noisy sentence alignment
2. mismatch to frozen T5 sentence embeddings
3. lack of lexical supervision

### Slide 4 — New architecture

High-level diagram:

pose → preprocessing → 4-layer encoder → WPP pretraining → Stage B gate → translation with WPP auxiliary

### Slide 5 — Stage A and Stage B results

Show AUC, isotropy, and gate pass.

### Slide 6 — Stage C first run result

Be honest: partial success, not final success.

### Slide 7 — Current retry and next actions

Explain what was changed in the retry and why.

### Slide 8 — Conclusion

1. semantic representation learning is now working
2. translation still needs refinement
3. this is the correct direction for the final month

---

## 12. What To Say If Asked “How Much Is Actually Done?”

A strong and honest answer is:

> The representation-learning part is now solved well enough to pass a formal semantic gate. The translation stage has improved but is not yet reliable enough for a final claim, so the current focus is reducing decoder-prior hallucination while preserving the newly learned lexical signal.

That statement is accurate and defensible.

---

## 13. Final Recommendation for This Saturday

For this Saturday, present the work as:

**“A successful diagnosis-and-redesign milestone, with strong representation-level progress and an ongoing translation-stage stabilization effort.”**

That is the right level of ambition.

It shows serious technical progress, gives your audience confidence that the project is now on a principled path, and still leaves the full benchmark and final translation story for the final presentation next month.
