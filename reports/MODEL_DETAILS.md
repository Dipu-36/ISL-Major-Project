# Detailed Model Architecture & Results — Research-Ready Summary

**Last Updated:** April 16, 2026

---

## Abstract

This document collects a research-ready, self-contained description of the ISL translation pipeline: dataset, preprocessing, model architectures, training recipes (Stages A–D), hyperparameter choices, evaluation protocols, ablations, results, and reproducibility instructions. It aggregates implementation details from `config/arch_v2_redesign.py`, `training/`, and `preprocessing/` so a reviewer can reproduce experiments and a manuscript can cite precise settings.

---

## 1. Dataset

- Dataset: iSign v1.1 (127,237 samples)
- Input: per-sample `.pose` files extracted by MediaPipe Holistic
- Pose composition:
  - 33 body landmarks
  - 21 left-hand landmarks
  - 21 right-hand landmarks
  - Each landmark: (x, y, z) + confidence score
- Standard data splits and cached lists: `artifacts/splits.json`

---

## 2. Preprocessing

Key steps (see [ARCHITECTURE.md](../ARCHITECTURE.md#2-data-preprocessing-pipeline) and `preprocessing/pipeline.py`):

- Spatial normalization: center on shoulder midpoint and scale by shoulder width.
- Confidence-based weighting: multiply joint coordinates by clipped confidence scores.
- Temporal smoothing: exponential moving average (EMA) with alpha ≈ 0.5.
- Motion-based keyframe sampling: compute per-frame velocity magnitude and select top-K frames (K=96) while preserving first/last frames.
- Velocity augmentation: compute frame-to-frame difference and concatenate with positions.
- Final tensor shape: `[96 frames, 450 dims]` (75 joints × 3 coords × 2 (pos+vel)).

Rationale: these steps normalize spatial variance, reduce noise while preserving motion onsets, and compress variable-length sequences into fixed-length inputs preserving highest-motion frames.

---

## 3. Model Architecture (Layer-by-layer)

### TemporalVisualEncoder (encoder)

- Input projection: `nn.Linear(450, 256)`
- Positional encoding: sinusoidal (non-learned)
- Transformer encoder: 4 layers (`nn.TransformerEncoder`) with
  - `d_model = 256`
  - `nhead = 8`
  - `dim_feedforward = 1024`
  - `dropout = 0.15`
  - `norm_first = True` (pre-LN)
- Mean-pooling across valid frames used for WPP head during Stage A

Parameter count (approx): ~3.3M for encoder (4-layer configuration)

### WPP Head (Stage A)

- `nn.Linear(d_model, WPP_VOCAB_SIZE)` (binary classifier per word)
- Loss: BCEWithLogits (with `pos_weight` to upweight rare words)
- Auxiliary VICReg penalties: variance + covariance (soft weights to avoid destroying coarse semantic structure)

### Bridge

- `nn.Linear(d_model, T5_D_MODEL)` → `GELU()` → `LayerNorm(T5_D_MODEL)`
- Purpose: adapt encoder representation (256d) to T5 embedding space (512d)

### Decoder (Stage C)

- `t5-small` (AutoModelForSeq2SeqLM)
- Decoder frozen initially; top-`DECODER_LAYERS_UNFREEZE` (2) blocks unfrozen after `FREEZE_DECODER_EPOCHS` epochs

---

## 4. Training Recipes

### Stage A v3: WPP Pretraining

- Objective: multi-label WPP classification on mean-pooled encoder outputs
- Optimizer: AdamW, LR = `WPP_LR` (3e-4) with linear warmup (3 epochs) + cosine decay to `WPP_LR_MIN` (1e-6)
- Batch size: 128
- Weight decay: 1e-2
- Regularization: Soft VICReg (`VICREG_COEFF_VAR = 0.25`, `VICREG_COEFF_COV = 0.02`)
- Pos-weight for BCE: typically 10.0 to reduce FN on rare words
- Early stopping: patience = 15 epochs on validation WPP loss
- Gate measurement logged each epoch: per-word AUC distribution + isotropy

### Stage B v2: Quality Gate

- Gate passes if at least 25 vocabulary words achieve AUC ≥ 0.62 and mean L1 isotropy ≤ 0.50
- Command: `python probe_stage_b_v2.py --checkpoint <encoder_best.pt>`

### Stage C v2: Translation Finetuning

- Load Stage A encoder checkpoint
- Bridge → T5-small decoder (pretrained)
- Freeze decoder for first 8 epochs; unfreeze top-2 decoder blocks thereafter
- Learning rates: encoder `1e-4`, decoder `1e-5` (very conservative)
- WPP auxiliary loss integrated into translation training with cosine-decayed weight (init 0.6 → end 0.3 or config canonical values)
- Batch size: 32 (with grad-accum to simulate larger batch if needed)
- Beam search decoding for validation: `num_beams = 4`
- Early stopping based on validation BLEU-4 (patience = 8)

### Stage D: Joint Fine-tuning (planned)

- Conservative joint learning rates: encoder `5e-6`, decoder `1e-5`
- Short window: ≤ 10 epochs with aggressive early stopping to avoid catastrophic forgetting

---

## 5. Hyperparameters (canonical; export)

Refer to snapshot in [ARCHITECTURE.md](../ARCHITECTURE.md#architecture-snapshot-exported-constants). Key values:

- `ENCODER_LAYERS = 4`, `D_MODEL = 256`, `TARGET_FRAMES = 96`
- `WPP_VOCAB_SIZE = 150`, `WPP_LR = 3e-4`, `WPP_BATCH_SIZE = 128`
- `LR_ENCODER = 1e-4`, `LR_DECODER = 1e-5`, `DECODER_LAYERS_UNFREEZE = 2`
- `VICREG_COEFF_VAR = 0.25`, `VICREG_COEFF_COV = 0.02`

---

## 6. Evaluation Protocol & Metrics

Primary: BLEU-4 (corpus-level, SacreBLEU), ROUGE-L, CHRF
Secondary: per-word AUC (WPP), isotropy (mean pairwise cosine of encoder outputs), generic ratio, dominant ratio, input-difference ratio (zeroing visual input)

- Checkpoint selection: `best_bleu_model.pt` (highest BLEU) for final reporting
- Validation: decode full validation set each epoch (2k samples) and compute corpus metrics
- Test: final evaluation done once on held-out test split (`artifacts/splits.json['test']`) — do not iterate on test set

---

## 7. Results

### Achieved (snapshot)

- Stage A:
  - `avg_auc` ≥ 0.58 (achieved)
  - `≥25 words with AUC ≥ 0.62` (gate passed)
  - `isotropy (L1)` ≤ 0.50 (achieved)

- Stage C (best validation):
  - BLEU-4 = **1.39** (epoch 18)
  - ROUGE-L = **10.26** (epoch 17)
  - Generic ratio ≈ 0.10, dominant ratio = 0.00

### Interpretation

- The WPP objective successfully produces lexical signal not present in InfoNCE experiments (v1/v2). The encoder exhibits measurable discriminability for a subset of content words and avoids collapse (good isotropy).
- Translation quality (BLEU) improved but remains below the iSign baseline target of 1.47; gap is modest and worth further ablation and statistical testing.

---

## 8. Ablation Summary

Completed and planned ablations (see [reports/HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)):

- InfoNCE vs WPP (WPP superior)
- VICReg strength (soft vs strong) — soft preserves coarse lexical structure
- Encoder depth (2 vs 4) — deeper encoder improved lexical signal
- WPP auxiliary weight schedule (affects hallucinations)
- Planned: vocabulary size, dataset scale, removing WPP aux in Stage C

---

## 9. Reproducibility & Commands

Quick commands (from the repo):

```bash
# Stage A: Encoder pretraining (WPP)
python training/train_stage_a_v3.py --run-name stageA_v3_run1

# Stage B: Quality gate
python probe_stage_b_v2.py --checkpoint artifacts/phase9_encoder_pretrain/stageA_v3_run1/encoder_best.pt

# Stage C: Translation training
python training/train_stage_c_v2.py \
  --encoder-checkpoint artifacts/phase9_encoder_pretrain/stageA_v3_run1/encoder_best.pt \
  --run-name stageC_v2_run1

# Full pipeline wrapper
./run_full_pipeline.sh
```

Recommended environment: Python 3.9–3.10, CUDA 11.x, and a 32GB+ GPU (A100 recommended for faster validation runs).

---

## 10. Compute, Runtime & Logging

- Typical validation decode (2k samples, beam=4) on an A100 ~10s per 2k samples (beam decode is parallelized; actual runtime depends on batch size and tokenizer speed)
- All runs log `history.json`, per-epoch checkpoints, and `predictions.json` under `artifacts/` with run-specific subfolders

---

## 11. Limitations & Future Work

- Current hyperparameter choices are heuristic; full grid search deferred due to GPU budget
- BLEU remains below baseline target; planned ablations and larger-scale Stage A runs may close gap
- Model currently handles lexical signal but not temporal alignment (ordering). Future work: temporal-span supervision or frame-level anchors to recover ordering information

---

## 12. References

Key references and code pointers:

- `config/arch_v2_redesign.py` — canonical constants
- `training/train_stage_a_v3.py`, `training/train_stage_c_v2.py`
- `preprocessing/pipeline.py`
- Papers: Vaswani et al. (2017), Raffel et al. (2020), Bardes et al. (VICReg, 2022)

---

## Appendix: What to include in a paper's "Methods" section

- Exact dataset split (cite `artifacts/splits.json` with seed)
- Preprocessing normalization & sampling algorithm (include code snippets)
- Complete model diagram (encoder layers, bridge, decoder)
- Hyperparameter table (as exported in `ARCHITECTURE.md` snapshot)
- Evaluation protocol (BLEU/ROUGE/CHRF, secondary metrics, gating rules)
- Reproducibility instructions (commands above + artifact paths)


