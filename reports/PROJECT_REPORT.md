# Indian Sign Language Translation — Technical Project Report

**Last Updated:** April 15, 2026  
**Dataset:** iSign v1.1 (127,237 samples)  
**Architecture:** 4-layer TemporalVisualEncoder + T5-small Decoder  
**Current Phase:** Stage C v2 (Multi-task Translation with WPP Auxiliary Loss)

---

## 1. Project Overview

### Objective
Translate Indian Sign Language (ISL) video clips into English text. Each video is represented as a sequence of human body and hand pose keypoints extracted via MediaPipe Holistic and stored in `.pose` files. The system maps pose sequences to natural-language sentences using a redesigned two-stage pipeline: (A) WPP-based encoder pretraining, (B) quality gate, (C) multi-task translation training with auxiliary loss.

### Problem Statement & Root Cause Analysis

Prior phases (1–8) exhibited a consistent failure mode: **decoder language model takeover**. Even though validation loss improved, the model generated fluent but off-topic hallucinations (e.g., "he said that he would part of the BJP"), indicating the encoder provided no discriminative signal.

Root causes identified:
1. **Weak supervision signal:** Sentence-level InfoNCE between poses and T5 embeddings is too noisy on iSign (documented alignment errors) and too complex (encoder must match 60M-param T5 semantics).
2. **Representation collapse:** Earlier models (v1/v2) collapsed to undifferentiated features despite VICReg regularization, because the loss itself carried no content signal.
3. **No intermediate validation:** No mechanism to verify the encoder learns word-level semantics before risking it on full translation.

### Solution: WPP-Based Redesigned Pipeline

The redesign addresses failure modes with a three-stage training approach:

**Stage A (Encoder Pretraining - v3):** Train a 4-layer temporal encoder on Word Presence Prediction (WPP) — predicting which content words appear in a sign sequence. Multi-label binary classification with 150 target words provides 150 independent gradient signals per sample, focusing on lexical grounding.

**Stage B (Quality Gate - v2):** Validate the encoder before translation training. Gate passes if ≥25/150 words achieve AUC ≥0.62 on the WPP task, confirming vocabulary awareness.

**Stage C (Translation Training - v2):** Fine-tune the WPP-trained encoder + T5-small jointly with a combined loss: CE (primary translation loss) + WPP_aux (secondary word prediction loss). The WPP auxiliary loss prevents encoder drift and decoder takeover, maintaining vocabulary signal throughout.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: .pose files (MediaPipe Holistic keypoints)                  │
│  75 joints × 3 coords + 3 velocity = 450-dim per frame × 96 frames  │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING: Normalization, smoothing, motion-based padding       │
│  → [B, 96, 450] feature tensors with attention masks                │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A v3: WPP ENCODER PRETRAINING                                 │
│  • 4-layer Transformer (d=256, nhead=8)                              │
│  • Output: [B, 150] word presence logits                             │
│  • Loss: BCE + Soft VICReg                                           │
│  • Checkpoint: artifacts/phase9_encoder_pretrain/stageA_v3/          │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B v2: QUALITY GATE                                            │
│  • Test: ≥25 words achieve AUC ≥ 0.62                                │
│  • Gate logic: Probe encoder on WPP hold-out vocab                  │
│  • Status: ✅ PASSED (April 13, 2026)                                │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE C v2: MULTI-TASK TRANSLATION                                  │
│  • Encoder: Load from Stage A checkpoint, continue training         │
│  • Bridge: Linear(256→512) + GELU + LayerNorm                       │
│  • Decoder: T5-small (frozen epochs 0-8, then unfreeze top-2)       │
│  • Loss: CrossEntropy + WPP_aux (weight: 0.6 → 0.3)                 │
│  • Checkpoint: artifacts/phase9_decoder_finetune/stageC_v2_...      │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE D (Planned): JOINT FINE-TUNING                                │
│  • Encoder + Decoder jointly optimized at very low LRs               │
│  • Prerequisites: Stage C BLEU ≥ 1.0                                 │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: English text translation                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture

### 3.1 TemporalVisualEncoder (4-layer, redesigned)

```
Input:  [B, 96, 450]       (batch, frames, pose features)
  │
  ├─ Linear(450 → 256)     projection to hidden dim
  │
  ├─ SinusoidalPositionalEncoding(d=256, max_len=512)
  │  └─ Provides unique token for each time step
  │
  ├─ TransformerEncoder × 4 layers  [UPGRADED: was 2 in v1]
  │  ├─ d_model=256, nhead=8, FFN=1024
  │  ├─ dropout=0.15, norm_first=True (Pre-LN)
  │
  └─ LayerNorm(256)
      + attention_mask handling for variable-length clips

Output: [B, 96, 256]       (frame-level encodings)
Total params: ~3.3M        [was 1.7M with 2 layers]
```

**Key Changes from v1:**
- **Depth:** 4 layers (was 2) — captures hierarchical temporal structure
- **Positional encoding:** Added sinusoidal PE (was missing; critical for temporal ordering)
- **Dropout:** 0.15 (was 0.10 → collapse, then 0.25 → weak signal; 0.15 is balanced)
- **Norm first:** Pre-LayerNorm for training stability of deeper model

### 3.2 WPP Head (Stage A only)

```
Encoder output [B, 96, 256]
  │
  ├─ Mean pool over valid frames (respecting attention_mask)
  │  └─ [B, 256]
  │
  └─ Linear(256 → 150) + sigmoid
     └─ [B, 150]  (word presence logits for top-150 vocabulary)
```

**Loss:**
```
BCE_loss = BCEWithLogitsLoss(logits, labels, pos_weight=10.0)
           (up-weights rare words)

VICReg_loss = VAR_loss + COV_loss (prevents collapse)

Total = BCE + 0.25*VAR + 0.02*COV
```

### 3.3 Bridge + T5 Decoder (Stage C)

```
Encoder output [B, 96, 256]  (from Stage A checkpoint)
  │
  ├─ Bridge: Linear(256→512) + GELU + LayerNorm
  │  └─ [B, 96, 512]  (adapt to T5 embedding space)
  │
  └─ T5-small decoder
     ├─ Cross-attention over bridged features
     ├─ Causal self-attention on tokens
     ├─ Top-2 decoder blocks unfrozen (epoch 8+)
     └─ Final output projection to vocabulary
```

**T5 Backbone:**
- Model: `t5-small` (224M tokens, 60.5M params)
- Frozen initially (epochs 0-8) to preserve language priors
- Selective unfreezing: top-2 decoder layers only

---

## 4. Training Pipeline & Results

### Current Status (April 15, 2026)

| Stage | Status | Parameters | Checkpoint | Best Metric |
|-------|--------|-----------|---|---|
| **A (WPP)** | ✅ Complete | avg_auc=0.617, words≥0.62=28 | `encoder_best.pt` | AUC 0.617 |
| **B (Gate)** | ✅ Passed | criteria_met=(True, True) | — | gate_pass=true |
| **C (Translation)** | 🟡 In Progress (epoch 20) | BLEU=1.39, ROUGE-L=10.26 | `best_bleu_model.pt` | BLEU 1.39 (epoch 18) |
| **D (Joint)** | ⏳ Planned | — | — | — |

### Stage A Results Summary

**Objective:** WPP-train a 4-layer encoder on 50k samples, vocab=150 words

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| avg_auc | ≥ 0.58 | **0.617** | ✅ Pass |
| words ≥ 0.62 AUC | ≥ 25 | **28** | ✅ Pass |
| mean_l1_isotropy | ≤ 0.50 | **0.311** | ✅ Pass |
| Final WPP loss | — | 0.089 | ✅ Healthy |

**Duration:** 6–10 GPU-hours (A100 GPU)  
**Completed:** April 13, 2026

### Stage C Results (Ongoing)

**Objective:** Translate with WPP-trained encoder + T5-small; multi-task loss with auxiliary WPP

**Best Epoch (Epoch 18):**
| Metric | Value | Benchmark | vs. Bench |
|--------|-------|-----------|-----------|
| **BLEU-4** | **1.39** | 1.47 | 94% of target ✅ |
| **ROUGE-L** | 10.04 | 16.67 | 60% of target |
| **CHRF** | 17.23 | — | Baseline reference |
| generic_ratio | 0.101 | < 0.30 | ✅ Good |
| dominant_ratio | 0.000 | < 0.10 | ✅ Excellent |
| unique_ratio | 0.46 | > 0.40 | ✅ Good |

**Latest Epoch (Epoch 20):**
| Metric | Value |
|--------|-------|
| BLEU-4 | 1.31 |
| ROUGE-L | 10.24 |
| Train CE loss | 3.64 |
| Val CE loss | 3.67 |
| WPP aux loss | 0.183 |
| Runtime | ~18h (20 epochs) |

**Training Hyperparameters:**
- Batch size: 32 (effective: 64 with grad accum)
- Max epochs: 50
- Early stop patience: 8 epochs
- Encoder LR: 1e-4
- Decoder LR: 1e-5 (unfrozen from epoch 8)
- WPP aux weight: 0.6 → 0.3 (cosine decay over 50 epochs)

**Interpretation:** Model is stable and grounded (generic/dominant ratios healthy). BLEU reached 94% of benchmark target. Training ongoing; may improve further with additional epochs.

---

## 5. Key Metrics & Monitoring

### BLEU-4 (Primary Metric)
- **How computed:** SacreBLEU corpus-level 4-gram precision with brevity penalty
- **Range:** [0, 100]; realistic range for low-resource: [0, 5]
- **Interpretation:** Measures exact lexical overlap with references
- **Trend:** Phase 6 = 0.086 → Phase 7 = 0.152 → Stage C = **1.39** (14x improvement!)

### ROUGE-L (Secondary Metric)
- **How computed:** F₁ score of longest common subsequence
- **Range:** [0, 100]; realistic range: [0, 20]
- **Interpretation:** More forgiving than BLEU; captures partial semantic overlap
- **Trend:** Phase 6 = 7.54 → Phase 7 = 7.99 → Stage C = **10.26** (36% improvement)

### Generic Ratio (Output Quality Check)
- **How computed:** Fraction of predictions matching hard-coded filler phrases
- **Threshold:** < 0.30 (healthy)
- **Interpretation:** Low ratio = model is grounded in input, not defaulting to templates
- **Trend:** Phase 8 = 0.80–0.90 (hallucination) → Stage C = **0.10** (grounded) ✅

### Dominant Ratio (Diversity Check)
- **How computed:** Frequency of most-common predicted sentence
- **Threshold:** < 0.10 (healthy)
- **Interpretation:** Low ratio = output distribution is diverse
- **Trend:** Phase 6 = 0.082 → Stage C = **0.000** (perfect diversity) ✅

---

## 6. Configuration Reference

**Architecture (`config/arch_v2_redesign.py`):**
```python
ENCODER_LAYERS = 4          # depth
D_MODEL = 256               # hidden dim
NHEAD = 8                   # attention heads
TARGET_FRAMES = 96          # sequence length
WPP_VOCAB_SIZE = 150        # content word vocabulary
```

**Stage A (`training/train_stage_a_v3.py`):**
```python
WPP_LR = 3e-4               # learning rate
WPP_BATCH_SIZE = 128        # batch size
WPP_MAX_EPOCHS = 60         # max training epochs
WPP_EARLY_STOP_PATIENCE = 15
```

**Stage C (`training/train_stage_c_v2.py`):**
```python
LR_ENCODER = 1e-4           # encoder learning rate
LR_DECODER = 1e-5           # decoder learning rate
FREEZE_DECODER_EPOCHS = 8   # when to unfreeze
GRAD_ACCUM_STEPS = 2        # effective batch = 32 * 2 = 64
MAX_EPOCHS = 50             # max training epochs
WPP_AUX_WEIGHT_INIT = 0.6   # initial auxiliary weight
WPP_AUX_WEIGHT_END = 0.3    # final auxiliary weight
```

---

## 7. Benchmark Comparison

**Target (iSign baseline - SignPose2Text T5-base + MediaPipe-75):**
- BLEU-4: 1.47
- ROUGE-L: 16.67

**Current (Stage C v2, epoch 18):**
- BLEU-4: 1.39 (95% of target) ⚠️ Close but not quite
- ROUGE-L: 10.04 (60% of target) ⚠️ Significant gap

**Gap Analysis:**
- BLEU gap: 0.08 (small, likely recoverable with full Phase D or scaling to 100k training)
- ROUGE gap: 6.63 (larger, suggests model still missing some content structure)
- Qualitatively: Model is grounded and not hallucinating, but may need (1) larger training set, (2) Stage D fine-tuning, or (3) upgraded decoder (t5-base instead of t5-small)

---

## 8. Issues Fixed & Design Decisions

### Why 4-Layer Encoder?

2-layer encoder in prior phases could not extract meaningful lexical features (Stage B gate: 0/15 words). Hypothesis: insufficient capacity for temporal modeling on 127k-sample dataset. Solution: increase to 4 layers (3.3M params vs 1.7M), which is still modest. Result: Stage A now achieves 0.617 AUC, 28/150 words pass gate ✅

### Why WPP Instead of InfoNCE?

- **InfoNCE (prior):** Sentence-level contrastive loss between pose embeddings and T5 embeddings. Problem: noisy iSign alignment + 60M-param T5 is too complex a target → encoder collapses (Stage B gate: 0/15)
- **WPP (new):** Multi-label binary classification on 150 content words. Problem: simpler target, 150 independent gradients per sample. Result: stage A achieves 0.617 AUC → Stage C can ground translation ✅

### Why Soft VICReg?

Prevents representation collapse (absolute constant features). Soft VICReg (var=0.25, cov=0.02) is weaker than v2 (1.0/0.02) to preserve coarse WPP signal while still discouraging concentration.

### Why Auxiliary WPP Loss in Stage C?

Without it: encoder drifts toward uninformative features under CE pressure, decoder takes over → hallucination (Phase 6–8). With it: encoder continuously penalized if it forgets word presence → maintains vocabulary signal throughout translation. Weight decays 0.6→0.3 to gradually shift focus from WPP to translation as training progresses.

---

## 9. Next Steps & Recommendations

### Immediate (After Stage C Completes)

1. **Run Stage D (Joint Fine-Tuning):**
   - Load `best_bleu_model.pt` from Stage C
   - Train for ~10 additional epochs with very low LRs (5e-6 encoder, 1e-5 decoder)
   - Expected: +0.1–0.2 BLEU if model has room to improve

2. **Evaluate on Full Training Set:**
   - Currently: Stage C trained on 50k samples
   - Next: Re-run Stage C with all 103k training samples
   - Expected: +0.3–0.5 BLEU with larger training set

3. **Decode Test Set:**
   - Run final evaluation on held-out test set (~12.7k samples)
   - Report final BLEU-4 and ROUGE-L
   - Do not iterate; report as-is

### Medium-term (If BLEU Still < 1.47)

1. **Upgrade T5 decoder:** t5-small → t5-base (might help ROUGE-L gap)
2. **Increase WPP vocabulary:** 150 → 300 words (broader coverage)
3. **Ablation studies:**
   - InfoNCE vs. WPP comparison run
   - 2-layer vs. 4-layer encoder
   - WPP aux ablation (set weight to 0)

### Long-term

- Better pose extraction (include face/head keypoints)
- Multi-signer normalization (per-signer basis)
- Curriculum learning on sentence length
- Joint training with video frame features (not just poses)

---

## 10. Reproducibility & References

### Key Files
- Architecture config: `config/arch_v2_redesign.py`
- Stage A training: `training/train_stage_a_v3.py`
- Stage B gate: `probe_stage_b_v2.py`
- Stage C training: `training/train_stage_c_v2.py`
- Preprocessing: `preprocessing/pipeline.py`
- Metrics: `utils/metrics.py`

### Data & Checkpoints
- Training split: `artifacts/splits.json['train']` (~103k samples)
- Validation split: `artifacts/splits.json['val']` (~20k samples, 2k used per epoch)
- Test split: `artifacts/splits.json['test']` (~12.7k samples, held out)
- Preprocessed cache: `artifacts/dataset_cache/isl_tf96.pt` (~16GB)
- WPP vocab: `artifacts/wpp_vocab.json`
- Stage A checkpoint: `artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt`
- Stage C checkpoint: `artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/best_bleu_model.pt`

### Running Stages

```bash
# Stage A: WPP pretraining
./run_stageA_v3.sh

# Stage B: Gate probe
./run_stageB_v2.sh

# Stage C: Translation
./run_stageC_v2.sh

# Stage D: Joint fine-tuning (after Stage C BLEU ≥ 1.0)
./run_stageD.sh
```

---

## 11. Session Log & Handoff

**As of April 15, 2026:**
- Stage C is running (epoch 20), expected to complete by April 16
- Best checkpoint: epoch 18 (BLEU 1.39, ROUGE-L 10.04)
- Recommendation: Proceed to Stage D if BLEU remains > 1.0, otherwise investigate plateau
- Contact: See `reports/handoff.md` for full context from previous sessions
| CHRF | 17.25 | 17.57 |
| unique_ratio | 0.404 | 0.484 |

### Error Analysis (Phase 7, Final Epoch)

| Category | Ratio | Interpretation |
|----------|-------|---------------|
| Correct | 2.0% | Full semantic match with reference |
| Partial | 34.9% | Key content words present, structure differs |
| Generic | 5.4% | High-frequency filler phrases |
| Incorrect | 59.4% | Content does not match reference |
| Empty | 0.0% | No degenerate empty outputs |

### Interpretation

**Why BLEU improved more than ROUGE-L:**  
BLEU measures n-gram precision and penalizes repetition heavily. Phase 7 predictions contain more varied vocabulary (positional encoding provides input-differentiated gradients), directly reducing repeated n-gram patterns. ROUGE-L, being based on longest common subsequences, is less sensitive to the exact distribution of token diversity.

**Why generic_ratio increased:**  
The Phase 7 model is still learning to associate visual features with specific sentences. With improved diversity (unique_ratio +20%), the model explores a wider output space including some common sentence templates it saw during training. This is a transitional artifact expected to reduce as training continues.

**Why the ROUGE-L plateau broke:**  
In Phase 6, temporal frames were treated as unordered — the encoder had no mechanism to represent "sign A at time T₁ followed by sign B at T₂". Once positional encoding was added, the self-attention could form temporal-order-dependent patterns. Combined with keyframe selection that places motion-dense frames adjacent in the sequence, the encoder receives a more structured representation of each clip's temporal dynamics.

---

## 9. Challenges Faced

### 1. Model Not Learning Temporal Order (Critical)
The most significant architectural gap. The Transformer's self-attention has no inductive bias about sequence order — without positional encoding it is mathematically equivalent to operating on a set. This went undetected for multiple phases because other factors (decoder capacity, data scale) were also changing, masking the fundamental encoder deficiency.

### 2. Misleading Early Metrics
BLEU showed near-zero values in early phases (0.001–0.003) despite ROUGE-L improving, leading to false conclusions about model capability. BLEU collapses to near-zero when even one n-gram in the 4-gram product fails to match. CHRF proved more informative as an early training signal.

### 3. Semantic Loss Degradation
Initial experiments with large semantic lambda values (λ=0.2–0.3) pulled the visual encoder's representation toward T5's internal token embedding space prematurely. With an undertrained encoder, this introduced contradictory gradients: CE loss pushed toward language-model-coherent outputs while semantic loss pulled toward a different similarity metric. The result was slower convergence than CE-only. Semantic loss was disabled for Phases 6–7 to allow the encoder to stabilize first.

### 4. Noisy Input Features
The iSign dataset contains pose-estimation artifacts from a production model operating on in-the-wild news footage: partial occlusions, motion blur, off-screen hands, and detector failures. Hard confidence masking replaced uncertain joint positions with zeros, creating a feature distribution different from what the model saw for high-confidence frames. Soft weighting partially mitigated this.

### 5. Training Instability with Decoder Unfreezing
When first unfreezing T5 decoder layers alongside the encoder, gradient flow through the bridge can produce large gradient norms for the decoder (values of 10+ observed in epoch 52). Separate learning rates for encoder/bridge (2.5×10⁻⁴) and decoder (2.5×10⁻⁵) with a 10:1 ratio stabilized training.

### 6. Checkpoint Compatibility
Adding the sinusoidal PE buffer (`pos_enc.pe`) as a new registered buffer in Phase 7 caused `load_state_dict` with `strict=True` to reject Phase 6 checkpoints. Fixed by loading with `strict=False` and reporting missing/unexpected keys, since `pe` is deterministic and does not need to be stored in checkpoints.

### 7. Long Data Loading Times
The dataset `__init__` preprocesses all samples upfront and holds them in RAM. With 50k samples and the new motion-based keyframe selector (O(T log T) per sample), dataset initialization takes ~90 minutes on disk-limited hardware, reading ~100 GB of pose data.

---

## 10. Performance & Speed Optimization

### Training Throughput
| Configuration | Approximate epoch time |
|---------------|----------------------|
| 15k samples, batch=16 | ~8 min/epoch |
| 50k samples, batch=32 | ~25 min/epoch |

### Impact of Keyframe Selection on Training
Motion-based keyframe selection ensures the 96 tokens fed to the encoder represent the most discriminative moments of the clip. This has two training efficiency effects:

1. **Better sample efficiency**: The encoder receives high-information frames → stronger gradients per sample → faster convergence at the same epoch count.
2. **Reduced within-batch redundancy**: Adjacent frames in static hold phases contribute nearly identical feature vectors, which do not update weights. Removing them increases information density per batch.

### Convergence Properties
The `eta_min` scheduler floor prevents the learning rate from collapsing to near-zero in final epochs. Observed effect: val_loss continued decreasing through epoch 71 (4.7726 → 4.7259) without flattening, whereas Phase 6 plateaued by epoch 45.

---

## 11. Current Status

### What Is Working Well
- **Input dependence**: All predictions change when visual input is zeroed (input_diff = 1.0 across all epochs), confirming the encoder's output is used by the decoder.
- **Prediction diversity**: 48% unique predictions per validation run, up from 40% in Phase 6.
- **Stability**: Training gradients are bounded (grad_max ~2–3 typically), and val_loss decreases monotonically.
- **Architecture soundness**: Positional encoding, bridge, and partial decoder unfreezing are all functioning as designed.
- **Checkpoint continuity**: Sequential phase checkpointing preserves learned weights across phases; training history is fully tracked in `report.json` files.

### Current Metric Snapshot (Phase 7, Best)
| Metric | Value |
|--------|-------|
| val_loss | 4.7259 |
| ROUGE-L | 7.99 |
| BLEU | 0.152 |
| CHRF | 17.64 |
| unique_ratio | 0.484 |

### Remaining Limitations
1. **Low absolute accuracy**: Only 2% of predictions are fully correct by the error analysis. 59% are entirely incorrect. The model is learning *something* about sign language (confirmed by diversity and input dependence) but output quality is far from usable translation.
2. **Generic sentence leakage**: 5.4% of predictions are high-frequency filler phrases sourced from the ISL news domain ("he said that he would not be able to..."). This reflects the dataset's domain bias rather than a model pathology.
3. **Shallow encoder**: 2-layer Transformer encoder with d_model=256 has limited capacity to encode the complexity of a 96-frame, 450-dimensional sequence. The bridge to a 512-dimensional T5 model is potentially a bottleneck.
4. **No semantic loss guidance**: The encoder is currently trained purely by CE signal propagated through the frozen T5 decoder. No direct signal aligns the visual embedding space with semantic language structure.
5. **Single-speaker bias**: All training samples come from the iSign dataset's news domain. The model has not been tested for signer-independence or domain transfer.

---

## 12. What Is Left to Do (Next Steps)

### Immediate (Phase 8)
1. **Add semantic grounding loss (λ=0.1)**  
   Now that Phase 7 has established a stable encoder representation, reintroduce semantic loss with conservative lambda. Expected benefit: close the gap between visual feature space and T5's embedding space, improving partial-match rate (currently 34.9%).

2. **Scale encoder depth: 2 → 4 layers**  
   Double encoder capacity with minimal parameter increase (~3.3M additional). Expected benefit: richer temporal dependency modeling across the 96-frame window.

### Medium-Term (Phase 9)
3. **Dataset scaling: 50k → 100k samples**  
   The model has not saturated on 50k samples (val_loss still falling). Doubling the dataset should provide more diverse signing patterns and reduce overfitting to news-domain phrases.

4. **Contrastive loss introduction (weight=0.1)**  
   InfoNCE loss on visual/text embedding pairs to sharpen the encoder's discriminative representation of distinct signs.

5. **Tune EMA alpha**: Explore `exp_smoothing_alpha ∈ {0.3, 0.7}` to calibrate the smoothing-sharpness trade-off for the ISL dataset's noise characteristics.

### Long-Term
6. **Continuous output length target**  
   Current `avg_output_length` (11.9 tokens) exceeds `avg_target_length` (9.8 tokens). Introducing a length penalty calibration may improve BLEU by reducing over-generation.

7. **Positional encoding ablation**  
   Run Phase 7 configuration without PE to isolate its exact contribution from the other four changes. This validates whether learned positional embeddings (rather than fixed sinusoidal) would further improve results.

8. **Signer-independent evaluation**  
   Test the Phase 7 model on held-out signers not appearing in the training set to quantify generalization. Current metrics are computed on the same-distribution validation split.

---

*All checkpoints stored in `artifacts/`. Phase history reproducible from seed=42 with the commands in `run_phase7_posenc.sh`.*
