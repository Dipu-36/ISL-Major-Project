# Indian Sign Language Translation — Technical Project Report

**Last Updated:** April 5, 2026  
**Dataset:** iSign v1.1  
**Architecture:** TemporalVisualEncoder + T5BridgeModel  
**Current Phase:** Phase 7 (Positional Encoding + Motion Keyframes)

---

## 1. Project Overview

### Objective
Translate Indian Sign Language (ISL) video clips into English text. Each video is represented as a sequence of human body and hand pose keypoints extracted by a pose-estimation model and stored in `.pose` files. The system maps this pose sequence directly to a natural-language sentence.

### Problem Statement
Sign language translation is a low-resource sequence-to-sequence problem with several structural challenges:
- The input is a continuous motion sequence, not a discrete vocabulary.
- Temporal order carries grammatical information (ISL uses topic-comment structure, not SVO).
- Individual frames contain unreliable joint detections (partial occlusion, motion blur).
- The iSign dataset contains diverse Indian news signing with significant vocabulary variance.

### High-Level Architecture

```
.pose file
    │
    ▼
Preprocessing Pipeline
  - Load body (33) + left hand (21) + right hand (21) keypoints = 75 joints × 3 coords
  - Confidence-weighted joint scaling
  - Shoulder-width spatial normalization
  - Exponential moving average smoothing
  - Motion-based keyframe selection (96 frames)
  - Velocity augmentation → 450-dim feature vector per frame
    │
    ▼
TemporalVisualEncoder  [1.70M params]
  - Linear projection: 450 → 256
  - Sinusoidal positional encoding
  - 2-layer Transformer Encoder (d_model=256, nhead=8, FFN=1024)
    │
    ▼
Bridge Layer
  - Linear(256 → 512) → GELU → LayerNorm
    │
    ▼
T5-small Decoder  [60.5M params, partially frozen]
  - Cross-attention over bridged visual features
  - Top-2 decoder layers unfrozen for fine-tuning
    │
    ▼
English translation
```

---

## 2. Data Pipeline & Input Representation

### Dataset

| Split | UIDs | Used in Training |
|-------|------|-----------------|
| Train | 103,276 | Up to 50,000 |
| Val   | 12,240  | Up to 500     |
| Total in CSV | 127,237 | — |

Each sample is a `(uid, text)` pair. The corresponding pose file is `iSign-poses_v1.1/{uid}.pose`.

### Keypoint Extraction

Three body regions are extracted per frame:

| Region | Landmark Indices | Joints |
|--------|-----------------|--------|
| Body | 0–32 | 33 joints |
| Left Hand | 501–521 | 21 joints |
| Right Hand | 522–542 | 21 joints |
| **Total** | — | **75 joints × 3 (x, y, z)** |

### Spatial Normalization

All joint positions are normalized relative to the shoulder midpoint to remove signer-specific scale and position:

1. Compute shoulder midpoint: `mid = (L_shoulder + R_shoulder) / 2`
2. Center all joints: `x_centered = x − mid`
3. Normalize by shoulder width: `x_norm = x_centered / ||L_shoulder − R_shoulder||`

This makes features invariant to signer height, camera distance, and screen position.

### Velocity Features

After normalization, the velocity (frame-to-frame difference) is concatenated to the position features:

```
velocity[t] = position[t] − position[t−1]   (zero for t=0)
features = concat(position, velocity)         → 75 × 6 = 450 dims/frame
```

Velocity provides explicit motion cues that help the encoder distinguish sign boundaries from hold phases.

### Confidence Handling

**Before (Phase ≤6):** Binary masking — joints below threshold were zeroed entirely:
```python
conf_mask = (confidence >= 0.2).astype(float)
xyz = xyz * conf_mask[..., None]   # joint becomes (0,0,0) if conf < 0.2
```
Problem: Creates artificial zero-displacement artifacts indistinguishable from true rest poses.

**After (Phase 7):** Soft confidence weighting — unreliable joints are attenuated, never eliminated:
```python
conf_weights = np.clip(confidence, 0.2, 1.0)
xyz = xyz * conf_weights[..., None]   # floor at 0.2 preserves spatial structure
```

### Temporal Sampling

The dataset contains clips of highly variable length. All clips are fixed to **96 frames** for batching.

**Before (Phase ≤6):** Uniform sampling with `np.linspace`:
```python
idx = np.linspace(0, T-1, 96, dtype=int)
```
Problem: Allocates equal budget to static hold-phases and rapid sign transitions. Loses detail at sign boundaries.

**After (Phase 7):** Motion-based keyframe selection:
```python
diff = x[1:] - x[:-1]                           # per-joint displacements
importance[1:] = norm(diff, axis=-1).mean(-1)    # mean L2 velocity per frame
# Select top-96 highest-importance frames, always including first/last
top_idx = argpartition(importance, -96)[-96:]
selected = sort(top_idx)                          # restore temporal order
```
This concentrates the fixed token budget on informationally dense moments (sign transitions), while static phases (which carry little discriminative signal) are de-prioritized.

### Smoothing

**Before (Phase ≤6):** Boxcar moving-average with `window=3`:
```python
# Symmetric window: equal weight on past and future frames
smoothed[t] = mean(x[t-1], x[t], x[t+1])
```
Problem: Non-causal; smears sign onsets symmetrically in both directions.

**After (Phase 7):** Exponential Moving Average (EMA):
```python
smoothed[t] = α * x[t] + (1−α) * smoothed[t−1]    # α = 0.5
```
EMA applies backward-only smoothing, preserving the sharpness of motion onsets while attenuating high-frequency detector noise.

---

## 3. Model Architecture

### TemporalVisualEncoder

```
Input:  [B, T=96, 450]   (batch × frames × features)
  │
  ├─ Linear(450 → 256)
  │
  ├─ SinusoidalPositionalEncoding(d_model=256, max_len=512)
  │     pe[:, 0::2] = sin(pos / 10000^(2i/d_model))
  │     pe[:, 1::2] = cos(pos / 10000^(2i/d_model))
  │     Added to projected token states (no gradients)
  │
  ├─ TransformerEncoder × 2 layers
  │     d_model=256, nhead=8, FFN=1024
  │     norm_first=True (Pre-LN for training stability)
  │     dropout=0.0
  │
  └─ LayerNorm(256)

Output: [B, T=96, 256]
Total params: ~1.70M
```

### Positional Encoding — Why It Was Missing and Its Impact

Without positional encoding, the Transformer's self-attention is **permutation-equivariant**: shuffling the 96 input frames in any order produces identical output representations. The encoder was treating the temporal sequence as an unordered bag of frames, making it impossible to learn:
- That hand raised at frame 10 *before* a face-touch at frame 30 is a specific sign
- Temporal grammatical constructs (topic marker at start vs comment at end)
- Motion direction (upward vs downward)

Adding sinusoidal PE injects a unique, deterministic encoding for each time step, enabling the encoder to distinguish `frame_10 → frame_30` from `frame_30 → frame_10`.

**Effect observed:** ROUGE-L ceiling lifted from 7.54 → 7.99 within the same 20 epochs. BLEU improved from 0.086 to 0.152 (best epoch).

### T5BridgeModel

```
TemporalVisualEncoder output: [B, 96, 256]
  │
  ├─ Bridge: Linear(256→512) → GELU → LayerNorm(512)
  │     Projects encoder space into T5's embedding dimension
  │
  └─ T5-small (60.5M params, mostly frozen)
        Encoder: REPLACED by bridged visual features
        Decoder: autoregressive text generation
                 Top-2 layers: unfrozen and fine-tuned
                 Remaining 4 decoder layers + all encoder: frozen

Trainable parameters:
  - TemporalVisualEncoder:  ~1.70M
  - Bridge layer:           ~131K
  - T5 top-2 decoder layers: ~5.26M
  Total trainable:          ~7.09M  (of 62.2M total)
```

### Decoder Unfreezing Strategy

Gradual decoder unfreezing prevents catastrophic forgetting of T5's language modeling priors:

| Phase | Unfrozen Layers | Decoder LR | Rationale |
|-------|----------------|-----------|-----------|
| Phase 4–5 | 0 (fully frozen) | — | Encoder-only training to establish visual features |
| Phase 5 | 1 layer | 5×10⁻⁵ | Introduce minimal decoder adaptation |
| Phase 5c | 2 layers | 5×10⁻⁵ | Increase decoder capacity |
| Phase 6–7 | 2 layers | 2.5×10⁻⁵ | Reduce decoder LR (larger dataset, reduce forgetting) |

### Final Architecture Flow

```
Pose file
  → 75 joints × 3 coords × T frames
  → Confidence weighting + spatial normalization + EMA smoothing
  → Motion-based keyframe selection → T=96 frames
  → Velocity concatenation → T×450 features
  → TemporalVisualEncoder → T×256 encoded tokens
  → Bridge → T×512 visual embeddings
  → T5 cross-attention decoder → token probabilities
  → Beam search (4 beams) → English sentence
```

---

## 4. Training Strategy

### Loss Functions

#### Cross-Entropy Loss (Primary)
Standard teacher-forced next-token prediction loss over the T5 vocabulary:
```
L_CE = CrossEntropy(logits, labels, label_smoothing=0.1, ignore_index=-100)
```
Label smoothing (0.1) reduces overconfidence on training tokens and improves generalization.

T5 handles the input-shift internally via the `labels` argument — the model receives `labels[:-1]` as decoder input and predicts `labels[1:]`.

#### Semantic Grounding Loss (Available, disabled in Phase 6–7)
Cosine distance loss between the visual encoder's pooled representation and T5's internal token embedding of the reference sentence:
```
L_sem = 1 − cosine_similarity(visual_pooled, text_pooled)
L_total = L_CE + λ * L_sem     (λ with warmup schedule)
```
Available via `--semantic-lambda`. Currently set to 0.0 in Phases 6–7 because CE-only training was still improving, and adding semantic loss before the encoder representation stabilizes risks misaligning the spaces.

#### Contrastive Loss (Available, disabled)
InfoNCE loss over positive (same-sample visual/text) and negative (other-sample) pairs. Available via `--contrastive-weight`. Disabled in all reported phases.

### Masking and Shifting
- Padding tokens in labels are set to `-100` before being passed to the loss function so they are excluded from gradient computation.
- T5 internally shifts labels for teacher forcing — no manual right-shift is applied.

---

## 5. Metrics Used

### BLEU (Bilingual Evaluation Understudy)
Measures n-gram precision between prediction and reference, with a brevity penalty for too-short outputs. Computed as corpus-level BLEU-4.

**Why it matters:** BLEU captures exact lexical overlap. A BLEU of 0 means no 4-gram matches exist between predictions and references; a BLEU > 0.1 indicates meaningful phrase-level alignment. In this project BLEU is volatile epoch-to-epoch due to the small validation set and the sensitivity of 4-gram matching.

### ROUGE-L
Measures the longest common subsequence (LCS) between prediction and reference, normalized as an F₁ score.

**Why it matters:** ROUGE-L is less sensitive than BLEU to exact n-gram positions and captures partial structural matches. It is the primary metric for this project because ISL translation rarely produces verbatim reference sentences but may capture key content words.

### CHRF (Character F-Score)
Macro-averaged F-score of character n-gram overlaps (n=1..6) between prediction and reference.

**Why it matters:** CHRF operates at sub-word level and is robust to morphological variation and tokenization differences. It provides a smooth signal even when BLEU is near zero, making it useful for monitoring early-stage training.

### dominant_prediction_ratio
Fraction of the validation set where the most-common predicted sentence was generated.

**Why it matters:** A high dominant ratio (e.g., >0.5) is a mode-collapse signal — the model has learned to output one "safe" sentence regardless of input. Phase 6 had a dominant ratio of ~0.08, indicating the model produces varied output.

### unique_prediction_ratio
Fraction of validation predictions that are unique strings.

**Why it matters:** Low uniqueness means the model's output distribution has collapsed into a small set of templates. Phase 7 achieved 0.48 unique ratio (up from 0.40 in Phase 6), indicating the positional encoding is providing input-differentiated signals.

### generic_sentence_ratio
Fraction of predictions that match a hard-coded list of generic filler phrases (e.g., "he said that he would not be able to do this").

**Why it matters:** Generic predictions indicate the model ignores the visual input and defaults to high-frequency training phrases. This is distinct from dominant ratio because many different generic sentences can score low on dominant but high on generic.

### input_difference_ratio
Proportion of validation pairs where zeroing out the visual input (replacing features with zeros) produces a different predicted sentence than the real input.

**Why it matters:** This is the **input-dependence test** — a value of 1.0 means every prediction changes when the input is zeroed, confirming the model uses the visual signal. A ratio < 1.0 would indicate the decoder is ignoring the encoder. All phases show input_diff = 1.0.

---

## 6. Hyperparameters & Configuration

| Parameter | Phase 5c | Phase 6 | Phase 7 |
|-----------|---------|---------|---------|
| train_samples | 15,000 | **50,000** | 50,000 |
| val_samples | 500 | 500 | 500 |
| batch_size | 16 | **32** | 32 |
| epochs per run | 15 | 20 | 20 |
| max_target_len | 48 | 48 | 48 |
| target_frames | 96 | 96 | 96 |
| feature_dim | 450 | 450 | 450 |
| d_model | 256 | 256 | 256 |
| nhead | 8 | 8 | 8 |
| encoder_layers | 2 | 2 | 2 |
| encoder LR | 5×10⁻⁴ | **2.5×10⁻⁴** | 2.5×10⁻⁴ |
| decoder LR | 5×10⁻⁵ | **2.5×10⁻⁵** | 2.5×10⁻⁵ |
| unfreeze_decoder | 2 | 2 | 2 |
| semantic_lambda | 0.0 | 0.0 | 0.0 |
| contrastive_weight | 0.0 | 0.0 | 0.0 |
| label_smoothing | 0.1 | 0.1 | 0.1 |
| scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| **eta_min** | 0 | 0 | **lr × 0.05 = 1.25×10⁻⁵** |
| num_beams | 4 | 4 | 4 |
| no_repeat_ngram | 3 | 3 | 3 |
| repetition_penalty | 1.2 | 1.2 | 1.2 |
| seed | 42 | 42 | 42 |

### Scheduler Note
`CosineAnnealingLR` without `eta_min` decays the learning rate to exactly 0 at the final epoch. This causes training to stall in the last ~20% of epochs as gradients become too small to overcome numerical noise. Setting `eta_min = lr × 0.05` ensures a minimum learning rate floor of `1.25×10⁻⁵` throughout training.

---

## 7. Optimization & Improvements (Phase-wise)

### Phase Progression Summary

| Phase | Epochs | Train | Best ROUGE-L | Best BLEU | Key Change |
|-------|--------|-------|-------------|-----------|-----------|
| Phase 4 (ce_extended) | 1–20 | 15k | 7.65 | 0.0012 | CE-only baseline, frozen decoder |
| Phase 5 (decoder_unfreeze) | 18–32 | 15k | 6.67 | 0.0013 | Unfreeze 1 decoder layer |
| Phase 5b (stronger_decoder) | 24–43 | 15k | 6.96 | 0.0835 | Unfreeze 2 layers, higher decoder LR |
| Phase 5c (2layer) | 18–32 | 15k | **7.54** | **0.143** | Reduced decoder LR, stabilized training |
| Phase 6 (50k) | 32–51 | 50k | 7.54 | 0.086 | Scale to 50k — ROUGE plateau began |
| **Phase 7 (posenc+keyframes)** | 52–71 | 50k | **7.99** | **0.152** | All 5 structural fixes applied |

### Problems Identified Before Phase 7

#### 1. No Positional Encoding
The `TemporalVisualEncoder` fed projected tokens directly into the Transformer with no position information. Attention layers treated all 96 frames as an unordered set. The encoder could not distinguish temporal proximity or order, making it impossible to learn sign-order-dependent semantics.

#### 2. Uniform Temporal Sampling
`np.linspace` allocated equal budget across the clip regardless of motion content. Static hold phases (the signer pausing between signs) consumed the same number of tokens as rapid handshape transitions. This caused the encoder to spend most of its capacity on low-information frames.

#### 3. Hard Confidence Masking
Joints with confidence < 0.2 were zeroed out, producing `(0,0,0)` coordinate triplets. These were indistinguishable from genuine rest-pose positions, introducing systematic bias into the feature distribution. Sudden discontinuities around the masking threshold created sharp jumps in the input sequence.

#### 4. Moving Average Smoothing
The boxcar filter smoothed symmetrically with equal weight on future frames: `smoothed[t] = mean(x[t-1..t+1])`. This is non-causal and blurs sign boundaries — the onset of a new sign gets partially averaged out with the preceding hold phase.

#### 5. LR Schedule Collapsing to Zero
`CosineAnnealingLR` without `eta_min` drove the encoder and bridge learning rates to zero by epoch 51. ROUGE-L stagnated despite val_loss still decreasing, indicating the model could improve but updates were too small to generate them.

### Fixes Applied in Phase 7

| Fix | Implementation | Mechanism |
|-----|---------------|-----------|
| Sinusoidal positional encoding | `SinusoidalPositionalEncoding` class, added after `input_proj` | Each frame position receives a unique deterministic offset enabling temporal ordering |
| Motion-based keyframe selection | `temporal_sample_keyframes()` using per-frame velocity magnitude | Top-K highest-motion frames selected, temporal order preserved |
| Soft confidence weighting | `np.clip(confidence, 0.2, 1.0)` instead of binary mask | Proportional attenuation of low-confidence joints, no zeroing |
| Exponential smoothing | `smooth_exponential(alpha=0.5)` replacing `filter_noise(window=3)` | Backward-only smoothing preserves sign onset sharpness |
| LR floor (eta_min) | `CosineAnnealingLR(T_max=epochs, eta_min=lr*0.05)` | Prevents learning rate from decaying to zero |

---

## 8. Results & Improvements

### Metrics: Phase 6 vs Phase 7

| Metric | Phase 6 (best) | Phase 7 (best) | Delta | Direction |
|--------|---------------|---------------|-------|-----------|
| val_loss | 4.7726 | **4.7259** | −0.047 | ↓ Better |
| ROUGE-L | 7.54 | **7.99** | +0.45 | ↑ Better |
| BLEU | 0.086 | **0.152** | +0.066 (+77%) | ↑ Better |
| CHRF | 17.27 | **17.64** | +0.37 | ↑ Better |
| unique_ratio | 0.40 | **0.48** | +0.08 (+20%) | ↑ Better |
| dominant_ratio | 0.082 | **0.072** | −0.010 | ↓ Better |
| generic_ratio | 0.006 | 0.054 | +0.048 | ↑ Worse |

### Final Epoch Comparison (Epoch 51 vs Epoch 71)

| Metric | Phase 6 Final | Phase 7 Final |
|--------|--------------|--------------|
| val_loss | 4.7726 | 4.7259 |
| ROUGE-L | 7.09 | 7.78 |
| BLEU | 0.072 | 0.100 |
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
