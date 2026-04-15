# ISL Project — Evaluation Plan

**Date:** April 8, 2026  
**Scope:** All evaluation stages for the WPP-redesigned pipeline

---

## 1. Representation Quality Metrics (Stage A)

These metrics are computed by `training/train_stage_a_v3.py` (probes every 5 epochs) and `probe_stage_b_v2.py` (full evaluation at gate).

### 1.1 Primary: Per-word AUC (WPP task)

| Metric | What it measures | Expected pre-training | Expected post-training |
|--------|-----------------|----------------------|------------------------|
| Per-word AUC | Can a linear classifier separate pose clips containing word W from those without? | ~0.50 (random) | ≥0.62 for ≥25 words |
| Average AUC over vocabulary | Mean vocabulary discriminability | 0.50 | ≥0.58 |
| Words with AUC ≥ 0.65 | "Easy" words — strongly signaled in ISL | 0 | ≥10 |

### 1.2 Secondary: Geometry

| Metric | What it measures | Expected collapse | Expected after WPP | Target |
|--------|-----------------|-------------------|-------------------|--------|
| Mean pairwise cosine sim | Encoder collapse indicator | 0.60–0.90 | < 0.40 | < 0.30 |
| PC1 variance ratio | How much variance is in a single direction (≤0.15 = healthy) | 0.40–0.60 | < 0.25 | < 0.20 |
| Effective dims (90% var) | Number of meaningful dimensions | 5–15 | ≥30 | ≥50 |

### 1.3 Retrieval (cross-sample)

| Metric | What it measures | Random baseline | Pre-training | Target |
|--------|-----------------|-----------------|--------------|--------|
| R@1 (encoder output) | Self-retrieval: is this sample's representation unique? | 1/N% | ~2% | ≥5% |

---

## 2. Semantic Gate Metrics (Stage B v2)

Gate in `probe_stage_b_v2.py`. Logged to `stage_b_v2_probe.json`.

| Check | Pass criterion | Current status |
|-------|---------------|----------------|
| Content words | ≥25 / 150 words with AUC ≥ 0.62 | FAIL (0/15 in v1, 0/15 in v2) |
| Isotropy | mean pairwise cos ≤ 0.50 | FAIL in v1 (0.64), PASS in v2 (0.20) |
| **Combined gate** | Both criteria | **FAIL** |

New target: Both criteria pass with v3 WPP training.

---

## 3. Translation Metrics (Stage C)

Computed in `train_stage_c_v2.py`. Logged to `history.json`, `best_bleu_model.pt`.

### 3.1 Primary

| Metric | Hallucination threshold | Minimal success | Target |
|--------|------------------------|-----------------|--------|
| BLEU-4 (sacrebleu intl) | < 0.5 (pure hallucination) | ≥ 1.0 | ≥ 5.0 |
| ROUGE-L | < 8 (prior phases) | ≥ 10 | ≥ 18 |

### 3.2 Anti-hallucination

| Metric | Definition | Prior phase range | Target |
|--------|-----------|------------------|--------|
| Dominant_ratio | Fraction of outputs that are the most common prediction | 0.14–0.21 in phases 6–8 | < 0.10 |
| Generic_ratio | Fraction starting "he said that he..." | ~0.80 | < 0.05 |

### 3.3 Encoder health during Stage C

To verify the encoder doesn't lose WPP signal during translation training:

| Check | How | Frequency |
|-------|-----|-----------|
| WPP auxiliary loss | Logged per epoch | Every epoch |
| WPP probe on val | Run `probe_stage_b_v2.py` with Stage C checkpoint | After epoch 10, then end |
| Encoder isotropy | Logged in Stage A history | Inherited from Stage A |

---

## 4. Generalization Checks

| Check | Method | Pass criterion |
|-------|--------|---------------|
| Train vs val BLEU gap | Compare `bleu_train` vs `bleu_val` | Gap ≤ 10 BLEU points |
| Train vs val WPP loss | Compare per epoch | Gap ≤ 0.05 |
| Cross-domain vocabulary | Probe AUC on test UIDs (held out) | AUC ≥ 0.55 for ≥15 words |

---

## 5. Overfitting Checks

| Risk | Indicator | Mitigation |
|------|-----------|------------|
| Encoder overfit on WPP | Train WPP loss drops but val AUC plateaus | WPP_WEIGHT_DECAY=0.01, early stop |
| Decoder memorisation | dominant_ratio rises, unique_ratio falls | patience=10-epoch stop on BLEU |
| Stage C encoder drift | Val WPP AUC drops between Stage A and Stage C end | WPP aux loss; monitor via reprobe |

---

## 6. Ablations

Only three ablations are prioritised (most impactful, least redundant with already-done H1–H5 study):

| Ablation | What to test | When to run |
|----------|-------------|-------------|
| **ABL-1: WPP vs InfoNCE** | Run Stage A with InfoNCE objective (v2 config) vs WPP (v3 config). Compares avg AUC at epoch 20. | After v3 succeeds (to confirm WPP is the cause) |
| **ABL-2: 4-layer vs 2-layer encoder** | Run Stage A v3 with ENCODER_LAYERS=2. Compares max words @ AUC≥0.62. | After v3 succeeds |
| **ABL-3: WPP aux weight** | Run Stage C with aux_weight=0 (no auxiliary). Compares Stage C BLEU and generic_ratio. | After Stage C succeeds |

These three ablations answer the three design decisions that differ most from prior work.

---

## 7. Benchmark Comparison Plan

| System | BLEU | Notes |
|--------|------|-------|
| iSign paper baseline (reported) | TBD from paper | From ISLTranslate_Dataset_for_Translating_Indian_Sign_L.pdf |
| Prior Phase 8 (this repo) | 0.00–0.80 (artefact) | Hallucination — not a real score |
| Stage A v3 → Stage C v2 (target) | ≥ 5.0 | Expected if semantic gate passes |

A meaningful comparison to the paper requires running the full pipeline and comparing on the iSign test set (using `artifacts/splits.json` test_uids).

---

## 8. Metric Instrumentation Checklist

Before running Stage C, verify these are logged in `history.json`:

- [x] `bleu` — sacrebleu BLEU-4 (intl tokenization)
- [x] `rouge_l` — ROUGE-L F1 averaged over validation set
- [x] `dominant_ratio` — fraction of outputs that match the most common prediction
- [x] `generic_ratio` — fraction of outputs matching "he said that he..." pattern
- [x] `val_ce` — validation cross-entropy loss
- [x] `wpp_aux_loss` — WPP auxiliary loss (encoder health monitor)
- [x] `aux_weight` — current WPP weight (for schedule verification)
- [x] `sample_preds` — 5 decoded examples per epoch (spot-check for hallucination)
