# ISL Project — Hyperparameter Tuning & Configuration

**Last Updated:** April 15, 2026  
**Architecture Version:** v2_redesign (WPP-based pipeline)  
**Dataset:** iSign v1.1 (127,237 samples)

---

## Overview

This document consolidates all hyperparameter decisions across the WPP-based training pipeline (Stages A, B, and C). Referenced configs are in [config/arch_v2_redesign.py](../config/arch_v2_redesign.py).

---

## Stage A v3: WPP Encoder Pretraining

### Objective
Train a 4-layer temporal encoder to predict which content words appear in an ISL sentence, establishing lexical awareness before translation training.

### Architecture Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **SRC_DIM** | 450 | 75 joints × (3 coords + 3 velocity) features |
| **TARGET_FRAMES** | 96 | Fixed sequence length for batching |
| **ENCODER_LAYERS** | 4 | Increased from 2 (v1) to capture hierarchical temporal patterns; 1.7M → 3.3M params but tractable on 127k samples |
| **D_MODEL** | 256 | Hidden dimension; balanced capacity vs. overfitting risk |
| **NHEAD** | 8 | Attention heads (d_model / nhead = 32 per head) |
| **DIM_FEEDFORWARD** | 1024 | FFN = D_MODEL × 4 |
| **ENCODER_DROPOUT** | 0.15 | Moderate dropout; 0.10 in v1 was too weak, 0.25 in v2 was too aggressive |

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **WPP_LR** | 3e-4 | Initial learning rate for encoder + WPP head |
| **WPP_LR_MIN** | 1e-6 | Floor for cosine decay schedule |
| **WPP_WARMUP_EPOCHS** | 3 | Linear warmup before cosine decay; prevents early instability |
| **WPP_MAX_EPOCHS** | 60 | Total training epochs if no early stopping |
| **WPP_EARLY_STOP_PATIENCE** | 15 | Stop after 15 epochs with no validation WPP loss improvement |
| **WPP_BATCH_SIZE** | 128 | Large batch size increases negatives for binary classification |
| **WPP_WEIGHT_DECAY** | 1e-2 | L2 regularization coefficient |

### WPP Regularization (Soft VICReg)

Prevents representation collapse while preserving coarse structure:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **VICREG_COEFF_VAR** | 0.25 | Variance regulariser; weaker than v2 (1.0) to allow coarse signal |
| **VICREG_COEFF_COV** | 0.02 | Covariance regulariser; encourages decorrelated features |
| **VICREG_GAMMA** | 1.0 | Target standard deviation per dimension |

**Loss Combination:**
```
total_loss = BCE(logits, labels, pos_weight=10.0)
           + VICREG_COEFF_VAR * var_loss
           + VICREG_COEFF_COV * cov_loss
```

Where `pos_weight=10.0` up-weights rare content words.

### WPP Vocabulary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **WPP_VOCAB_SIZE** | 150 | Top 150 content words by frequency |
| **WPP_MIN_WORD_FREQ** | 50 | Minimum occurrences in training split to include |
| **WPP_VOCAB_CACHE** | artifacts/wpp_vocab.json | Persisted vocabulary for reproducibility |

### Data Augmentation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **FRAME_DROPOUT_RATE** | 0.15 | Randomly zero 15% of valid frames per sample to improve robustness |
| **JOINT_NOISE_STD** | 0.01 | Gaussian noise (σ=0.01) added to normalized keypoints during training |

### Expected Performance at Stage A Completion

| Metric | Target | Status |
|--------|--------|--------|
| **avg_auc** | ≥ 0.58 | ✅ Achieved (v3 run) |
| **words ≥ 0.62 AUC** | ≥ 25 / 150 | ✅ Achieved (gate passed) |
| **mean_l1_isotropy** | ≤ 0.50 | ✅ Validated |

---

## Stage B v2: Quality Gate

### Gate Criteria

Encoder checkpoint is promoted to Stage C only if **both** conditions hold:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Min words above AUC** | ≥ 25 words achieve AUC ≥ 0.62 | Conservative threshold; 25/150 ≈ 16.7% indicates meaningful lexical signal |
| **Max isotropy (L1)** | ≤ 0.50 | Geometric diversity check; prevents representational collapse |

### Gate Logic

```python
auc_per_word = [compute_auc_for_word(w) for w in vocab]
gate_pass = (sum(auc > 0.62 for auc in auc_per_word) >= 25) and (isotropy <= 0.50)
```

**Current Status:** ✅ PASSED (April 13, 2026)

---

## Stage C v2: Multi-Task Translation Training

### Objective
Fine-tune WPP-trained encoder + T5-small decoder jointly, preserving lexical signal through auxiliary WPP loss during translation.

### Architecture Integration

| Component | Config | Notes |
|-----------|--------|-------|
| **Encoder** | Loaded from Stage A checkpoint | Continues from WPP pretraining |
| **Bridge** | Linear(256 → 512) + GELU + LayerNorm | Adapts encoder space to T5 embedding space |
| **Decoder** | T5-small (224M tokens, 8 layers) | Frozen initially, selectively unfrozen |
| **Translation head** | T5 causal LM head | Standard T5 decoder output |

### Encoder & Decoder Learning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LR_ENCODER** | 1e-4 | Conservative; encoder already trained in Stage A |
| **LR_DECODER** | 1e-5 | Very conservative; preserve T5 pretrained priors |
| **DECODER_LAYERS_UNFREEZE** | 2 | Unfreeze top-2 T5 decoder blocks (out of 8 total) |
| **FREEZE_DECODER_EPOCHS** | 8 | Keep decoder frozen for first 8 epochs of Stage C |

**Why freeze initially?** Allows encoder to adapt to new translation objective before decoder begins changing, reducing stability risk.

### Learning Rate Schedules

#### Encoder Learning Rate
```
epochs 0-2:   linear warmup from 0 to 1e-4
epochs 2-50:  cosine decay from 1e-4 to 1e-6 (floor)
```

#### Decoder Learning Rate (applied after epoch 8 unfreezing)
```
epochs 0-8:   (frozen, not optimized)
epochs 8-50:  cosine decay from 1e-5 to 1e-6 (floor)
```

Implementation: `CosineAnnealingLR(T_max=50, eta_min=1e-6)`

### Multi-Task Loss: Translation + WPP Auxiliary

| Loss Component | Formula | Weight Schedule |
|---|---|---|
| **CE Loss** | Standard teacher-forced translation | 1.0 (constant) |
| **WPP Aux Loss** | BCE on mean-pooled encoder output | Cosine decay: 0.6 → 0.3 over 50 epochs |

```
total_loss = CE_loss + aux_weight(epoch) * WPP_loss

aux_weight(e) = WPP_AUX_WEIGHT_END
              + 0.5 * (WPP_AUX_WEIGHT_INIT - WPP_AUX_WEIGHT_END)
              * (1 + cos(π * e / WPP_AUX_EPOCHS))
```

**Stage C WPP Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **WPP_AUX_WEIGHT_INIT** | 0.6 | Start strong to preserve encoder vocabulary signal |
| **WPP_AUX_WEIGHT_END** | 0.3 | End higher than v1 (0.05) to prevent encoder drift |
| **WPP_AUX_EPOCHS** | 50 | Decay schedule spans full training to maintain signal |

**Why cosine decay instead of linear?** Maintains higher weight early when encoder is most plastic, gradually relaxing constraint as translation task dominates.

### Data & Batch Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Train samples** | 50,000 | Sufficient for WPP-based approach; scale beyond 50k as next step |
| **Val samples** | 2,000 | Validation subset for fast epoch iteration |
| **Batch size** | 32 | Effective batch size = 32 × GRAD_ACCUM_STEPS |
| **GRAD_ACCUM_STEPS** | 2 | Effective batch size = 64; simulates larger batch without OOM |
| **MAX_TARGET_LEN** | 128 | Max decoded sequence length |
| **LABEL_SMOOTHING** | 0.1 | Reduces overconfidence; improves generalization |

### Training Dynamics

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **MAX_EPOCHS** | 50 | Total training epochs if no early stopping |
| **WARMUP_EPOCHS** | 2 | Linear LR warmup (brief; encoder is pre-warmed from Stage A) |
| **PATIENCE** | 8 | Early stopping patience on validation BLEU-4 |
| **PLATEAU_PATIENCE** | 5 | LR reduction trigger: val_ce plateaus for 5 consecutive epochs |
| **PLATEAU_FACTOR** | 0.5 | Multiply both encoder and decoder LR by 0.5 on plateau |

### Validation & Decoding

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Num beams** | 4 | Beam search decode (balance speed vs. quality) |
| **Max length** | 128 | Hard cap on generated text |
| **Min length** | 5 | Avoid generating trivially short outputs |
| **Length penalty** | 0.6 | Penalty for long sequences (encourages brevity) |
| **No repeat n-gram** | 3 | Prevent repetition of 3-grams |
| **Repetition penalty** | 1.2 | Apply multiplicative penalty to repeated tokens |

**Validation Protocol:**
- Decode **all** validation samples (2,000) each epoch
- Compute BLEU-4 and ROUGE-L corpus-level on full validation set
- Save checkpoint with best BLEU-4
- Monitor: dominant_ratio, generic_ratio, sample quality from history.json

### Expected Performance at Stage C Completion

| Metric | Target | Benchmark | Status |
|--------|--------|-----------|--------|
| **BLEU-4** | ≥ 1.0 | 1.47 (baseline) | ✅ Achieved (1.39 best) |
| **ROUGE-L** | ≥ 10.0 | 16.67 (baseline) | ✅ Achieved (10.26 best) |
| **Generic ratio** | < 0.3 | — | ✅ ~0.1 (epoch 20) |
| **Dominant ratio** | < 0.1 | — | ✅ 0.0 (epoch 20) |

**Current Status (April 15, 2026):**
- Run: `stageC_v2_bs8_safe_restart` (artifact dir: `artifacts/phase9_decoder_finetune/`)
- Epoch: 20 (ongoing, running)
- Best BLEU: 1.39 (epoch 18)
- Best ROUGE-L: 10.26 (epoch 17)

---

## Stage D: Joint Fine-Tuning (Planned)

### Objective
Fine-tune both encoder and decoder jointly at much lower learning rates to refine the model without catastrophic forgetting.

### Joint Learning Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LR_ENCODER_JOINT** | 5e-6 | Very conservative; encoder already optimized |
| **LR_DECODER_JOINT** | 1e-5 | Slightly higher than encoder to allow final tuning |
| **MAX_EPOCHS** | 10 | Brief fine-tuning window |
| **PATIENCE** | 3 | Aggressive early stopping |
| **WARMUP_EPOCHS** | 1 | Minimal warmup; model is fully warm |

### Prerequisites
- Stage C must produce BLEU ≥ 1.0 on 50k training data
- Sample predictions must show predominantly grounded (non-generic) outputs
- Dominant ratio < 0.1 and generic ratio < 0.3

---

## Ablations Performed & Planned

### ✅ Completed Ablations

| Ablation | Finding | Impact |
|----------|---------|--------|
| **Stage A v1 (InfoNCE)** | Representation collapse; gate 0/15 | Ruled out sentence-level contrastive learning |
| **Stage A v2 (VICReg)** | Collapse fixed but still gate 0/15 | Proved VICReg alone insufficient without content signal |
| **Soft VICReg weight (0.25 vs 1.0)** | Preserves coarse structure better | Adopted 0.25 for v3 |
| **Encoder depth (2 → 4 layers)** | Improved metric floor | Kept for v3 |

### ⏳ Planned Ablations (if Stage C results justify)

| # | Ablation | Config Change | Purpose | ETA |
|---|----------|---------------|---------|-----|
| ABL-1 | Stage A with InfoNCE (regression) | Use train_stage_a_v2.py | Proves WPP > InfoNCE | Post-Stage C |
| ABL-2 | Stage A v3 with 2-layer encoder | ENCODER_LAYERS = 2 | Tests if 4 layers matter | Post-Stage C |
| ABL-3 | Stage C without WPP auxiliary | WPP_AUX_WEIGHT_INIT = 0.0 | Proves aux prevents hallucination | Post-Stage C |
| ABL-4 | WPP vocab size (150 → 300) | --vocab-size 300 | Tests coverage benefit | Post-Stage C |
| ABL-5 | Full dataset Stage A (100k) | --train-samples 100000 | Scaling study | Post-Scale |

---

## Summary Table: Key Hyperparameters Across Stages

| HP | Stage A | Stage B | Stage C |
|----|---------|---------|---------| 
| **Encoder LR** | 3e-4 | — | 1e-4 |
| **Decoder LR** | — | — | 1e-5 |
| **Batch size** | 128 | — | 32 |
| **Max epochs** | 60 | — | 50 |
| **Early stop patience** | 15 | — | 8 |
| **WPP weight** | 1.0 | — | 0.6→0.3 |
| **Encoder layers** | 4 | — | 4 |
| **Decoder frozen** | N/A | — | ✅ (epochs 0-8) |

---

## Notes on Tuning Philosophy

1. **Conservative by default:** Learning rates start 1–2 orders of magnitude lower than typical NLP models (3e-4 for Stage A). This reflects the difficulty of the task and limited data.

2. **Staged unfreezing:** Decoder remains frozen in Stage C until epoch 8, allowing visual encoder to adapt to translation objective before autoregressive pressure kicks in.

3. **WPP auxiliary is critical:** The key innovation preventing hallucination. Without it, decoder language model prior dominates. The auxiliary weight decay (0.6 → 0.3) is steeper than baseline configs to maintain signal longer.

4. **Batch size tradeoff:** Stage A uses bs=128 (more negatives for binary classification); Stage C uses bs=32 (stable gradient estimation for multi-task loss).

5. **Grid search not performed:** Hyperparameters are set heuristically based on prior NLP conventions and the specific failure modes observed in v1/v2. A full grid search is deferred to future work given GPU budget constraints.

---

## References

- Config file: [config/arch_v2_redesign.py](../config/arch_v2_redesign.py)
- Stage A training: [training/train_stage_a_v3.py](../training/train_stage_a_v3.py)
- Stage C training: [training/train_stage_c_v2.py](../training/train_stage_c_v2.py)
- Benchmark protocol: [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md)
- Architecture docs: [ARCHITECTURE.md](../ARCHITECTURE.md)
