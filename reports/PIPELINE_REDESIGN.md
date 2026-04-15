# ISL Project — Pipeline Redesign Plan

**Date:** April 8, 2026  
**Status:** Implementation complete. Ready to run.

---

## 1. Why the Previous Approaches Failed

See [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md) for full analysis. In brief:

1. Sentence-level InfoNCE against T5 embeddings carries near-zero gradient toward lexical content on noisy iSign alignment.
2. Fixing geometric collapse (VICReg in v2) is necessary but not sufficient — semantically random spread is equally useless.
3. No sub-sentence supervision signal means the encoder can only learn surface statistics (sequence length, motion energy).

---

## 2. Redesign Principles

These are the principles guiding the new pipeline — not copied from any paper:

| Principle | Rationale |
|-----------|-----------|
| **Direct vocabulary supervision** | Signs correspond approximately to words; predicting which words appear in a clip is a tractable proxy for "understanding signs" |
| **Multi-task loss during translation** | Preserves encoder vocabulary awareness under pressure from the generative CE loss |
| **Soft collapse prevention** | VICReg stays but at weight 0.25 (not 1.0); preserves coarse similarity structure |
| **Deeper encoder** | 4 layers (was 2); more capacity to model temporal sign structure at 450-dim input; still modest at ~3.3M params |
| **No T5 in pretraining** | Removes the misaligned alignment target; encoder is self-contained during Stage A |
| **Gate on content, not geometry** | AUC per word (not isotropy lift); 0.62 AUC means the model can predict that word better than random |

---

## 3. New Pipeline Overview

```
STAGE 0 (before training)
  ▼  python training/word_inventory.py
  ▼  → artifacts/wpp_vocab.json  (top-150 content words by frequency)

STAGE A v3  [encoder pretraining — ~6–10 GPU-hours]
  Input:  pose features [B, 96, 450]
  Model:  TemporalVisualEncoder (4 layers, d=256)
          + WPP classification head Linear(256, 150)
  Loss:   BCEWithLogitsLoss (multi-label WPP) + soft VICReg (var_weight=0.25)
  Output: artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt
  Goal:   ≥25 / top-150 words with AUC ≥ 0.62 on validation set

STAGE B v2  [probe gate — ~30 min]
  Evaluates: per-word AUC, encoder isotropy, retrieval R@1
  Gate pass: ≥25 words @ AUC≥0.62 AND isotropy ≤ 0.50
  Output: stage_b_v2_probe.json
  [STOP if gate fails]

STAGE C v2  [translation + WPP aux — ~12–18 GPU-hours]
  Input:  WPP-trained encoder + T5-small
  Loss:   CE_translation + aux_weight * WPP_aux
          aux_weight: 0.5 → 0.05 over 30 epochs (cosine decay)
  Decoder frozen for first 5 epochs, then top-2 blocks unfrozen
  Goal:   BLEU > 1.0, dominant_ratio < 0.3, generic_ratio < 0.1

STAGE D  [joint fine-tuning — same as v1, update checkpoint path]
  Uses: train_stage_d.py (existing, update encoder checkpoint source)
```

---

## 4. Key Differences from Previous Attempts

| Aspect | v1 (Stage A InfoNCE) | v2 (InfoNCE + VICReg) | v3 (WPP + soft VICReg) |
|--------|---------------------|----------------------|------------------------|
| Pretraining objective | Sentence-level InfoNCE | Same | **WPP multi-label BCE** |
| Projection head | 2-layer MLP + BN | Linear | **None (no projection)** |
| Collapse prevention | None | VICReg (weight=1.0) | **VICReg (weight=0.25)** |
| Encoder depth | 2 layers | 2 layers | **4 layers** |
| T5 in pretraining | Yes (frozen) | Yes (frozen) | **No T5** |
| Translation loss | CE only | CE + contrastive | **CE + WPP aux** |
| Encoder capacity | 1.7M | 1.7M | **~3.3M** |
| Gate metric | Content lift (1.15×) | Same | **Per-word AUC (0.62)** |

---

## 5. What Could Still Fail and What To Do

### If Stage A v3 also fails to reach gate (avg AUC ≤ 0.55)

**Interpretation:** The pose features genuinely do not encode consistent lexical information — the same word is signed differently enough across samples that no linear classifier on mean-pooled features can tell them apart.

**Next actions:**
1. Probe raw input discriminability (before any encoder) on L0 features using the word labels. If L0 lift is also ≈ 1.0, the problem is in the pose extraction / dataset noise, not the model.
2. Try a windowed WPP objective: instead of predicting from mean-pool, predict from each overlapping temporal window of 16 frames independently. Signs are short (~10–15 frames each); mean-pooling over 96 frames may blend the signal.
3. Increase WPP vocabulary to the top-500 words and lower the gate threshold — perhaps only common/concrete words can be detected.
4. Consider a pre-trained sign recognition backbone: if a pre-trained wrist-velocity classifier or a MediaPipe Gesture recognizer exists for ISL, use it as a feature extractor rather than training from scratch on L1 features.

### If Stage B passes but Stage C still hallucinates

**Interpretation:** The encoder has learned vocabulary but the decoder still ignores it.

**Next actions:**
1. Increase the WPP auxiliary weight (start at 1.0 instead of 0.5).
2. Add cross-attention entropy monitoring: if mean entropy of T5 cross-attention weights is > log(T) - 0.5, the decoder is attending uniformly (ignoring encoder).
3. Try attention forcing: add a loss term that maximises the variance of T5 cross-attention weights across temporal positions.
4. Try a smaller decoder: use t5-base encoder-only + a 2-layer transformer decoder trained from scratch. This removes the T5 LM prior entirely.

---

## 6. Evaluation Criteria for Success

| Metric | Minimum required | Target |
|--------|-----------------|--------|
| Stage A avg AUC (val) | 0.55 | 0.65 |
| Stage B words @ AUC≥0.62 | 25 / 150 | 50 / 150 |
| Stage C BLEU (val) | 1.0 | 5.0 |
| Stage C dominant_ratio | < 0.5 | < 0.2 |
| Stage C generic_ratio ("he said that he") | < 0.3 | < 0.05 |
| Stage C val ROUGE-L | > 8 | > 15 |
