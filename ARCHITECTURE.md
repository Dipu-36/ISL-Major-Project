# ISL Translation Project — Complete Architecture Documentation

**Last Updated:** April 10, 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Preprocessing Pipeline](#2-data-preprocessing-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Key Technical Concepts Explained](#5-key-technical-concepts-explained)
6. [Configuration Reference](#6-configuration-reference)
7. [Failure Analysis & Redesign History](#7-failure-analysis--redesign-history)

---

## 1. System Overview

This project implements an **Indian Sign Language (ISL) to English text translation** system using a two-stage neural architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video/Pose Data                           │
│                    (MediaPipe Holistic keypoints)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 0: PREPROCESSING                                │
│  • Extract body + hand keypoints from .pose files                       │
│  • Normalize spatial coordinates (shoulder-centered)                    │
│  • Temporal sampling (motion-based keyframe selection)                  │
│  • Add velocity features                                                │
│  • Output: [96 frames × 450 dimensions] per sample                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE A: ENCODER PRETRAINING (v3)                     │
│  • Task: Word Presence Prediction (WPP)                                 │
│  • Model: TemporalVisualEncoder (4-layer Transformer)                   │
│  • Loss: BCE + Soft VICReg                                              │
│  • Output: Encoder checkpoint with lexical awareness                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE B: QUALITY GATE (v2)                            │
│  • Validate: Per-word AUC ≥ 0.62 for ≥25 words                          │
│  • Validate: Isotropy ≤ 0.50 (geometric diversity)                      │
│  • Decision: Proceed to Stage C only if gate passes                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE C: TRANSLATION TRAINING (v2)                    │
│  • Model: WPP-trained encoder + T5-small decoder                        │
│  • Loss: CrossEntropy + WPP auxiliary loss                              │
│  • Strategy: Gradual decoder unfreezing                                 │
│  • Output: Full translation model                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE D: JOINT FINE-TUNING                            │
│  • Fine-tune both encoder and decoder jointly                           │
│  • Conservative learning rates to preserve learned features             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: English Text                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

The iSign dataset presents unique challenges:
- **Noisy alignments**: Sentence-level pose-text pairs have significant annotation noise
- **High linguistic diversity**: ~127K samples with rich vocabulary
- **Motion variability**: Same sign varies significantly across signers/contexts

Our redesign (April 2026) addresses these by:
1. **Replacing sentence-level contrastive learning** with word-level supervision (WPP)
2. **Removing T5 from pretraining** — encoder learns independently
3. **Using multi-task learning** during translation to preserve lexical signal
4. **Adding deeper encoder** (4 layers vs 2) for richer temporal modeling

---

## 2. Data Preprocessing Pipeline

File: [`preprocessing/pipeline.py`](preprocessing/pipeline.py)

### 2.1 Input Data Format

Source: [iSign v1.1 dataset](https://www.sign-lang.uni-hamburg.de/isl/index.html)

| Component | Details |
|-----------|---------|
| **Pose files** | `.pose` format (MediaPipe Holistic) |
| **Body joints** | 33 landmarks (indices 0-32) |
| **Left hand** | 21 landmarks (indices 501-521) |
| **Right hand** | 21 landmarks (indices 522-542) |
| **Features per joint** | (x, y, z) coordinates + confidence score |
| **Raw input** | Variable-length sequences (~50-200 frames) |

### 2.2 Preprocessing Steps

#### Step 1: Spatial Normalization
```python
def normalize_spatial(x: np.ndarray) -> np.ndarray:
    """Center on shoulder midpoint, scale by shoulder width."""
    shoulders_mid = (x[:, L_SHOULDER_IDX, :] + x[:, R_SHOULDER_IDX, :]) / 2.0
    centered = x - shoulders_mid[:, None, :]

    shoulder_vec = centered[:, L_SHOULDER_IDX, :] - centered[:, R_SHOULDER_IDX, :]
    scale = np.linalg.norm(shoulder_vec, axis=-1)
    normalized = centered / safe_scale[:, None, None]
```

**Why:** Makes poses invariant to camera distance and body position in frame. Using shoulder width as scale factor accounts for different body sizes.

#### Step 2: Confidence-Based Weighting
```python
conf_weights = np.clip(confidence, CONF_THRESHOLD, 1.0)
xyz = xyz * conf_weights[..., None]
```

**Why:** MediaPipe provides confidence scores per joint. Low-confidence detections (often occluded hands) are down-weighted rather than discarded.

#### Step 3: Temporal Smoothing (EMA)
```python
def smooth_exponential(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    smoothed[t] = alpha * x[t] + (1 - alpha) * smoothed[t-1]
```

**Why:** Reduces high-frequency noise from pose estimation while preserving motion onsets better than simple averaging.

#### Step 4: Motion-Based Keyframe Sampling
```python
def temporal_sample_keyframes(x: np.ndarray, target_frames: int = 96):
    # Compute frame importance = velocity magnitude
    importance[1:] = np.linalg.norm(x[1:] - x[:-1], axis=-1).mean(axis=-1)

    # Always keep first and last frames
    importance[0] = importance.max() + 1.0
    importance[-1] = importance.max() + 1.0

    # Select top-K most dynamic frames
    top_idx = np.argpartition(importance, -target_frames)[-target_frames:]
    return x[np.sort(top_idx)]  # Restore temporal order
```

**Why:** Long sequences are compressed to fixed length (96 frames) while preserving the most informative motion segments.

#### Step 5: Velocity Feature Augmentation
```python
velocity[1:] = x[1:] - x[:-1]
merged = np.concatenate([x, velocity], axis=-1)
```

**Why:** Static pose (position) + dynamic motion (velocity) provides richer signal for distinguishing signs. Many ISL signs differ primarily in motion trajectory, not handshape.

### 2.3 Output Tensor Shape

```
Final feature tensor: [96 frames, 450 dimensions]

Breakdown:
- 75 joints total (33 body + 21 left hand + 21 right hand)
- 3 coordinates per joint (x, y, z)
- 2 features per coordinate (position + velocity)
- 75 × 3 × 2 = 450 dimensions
```

---

## 3. Model Architecture

### 3.1 TemporalVisualEncoder

File: [`model/temporal_encoder.py`](model/temporal_encoder.py)

```python
class TemporalVisualEncoder(nn.Module):
    """4-layer Transformer encoder for pose sequences."""

    def __init__(
        self,
        src_dim: int = 450,        # Input: pose features
        d_model: int = 256,        # Hidden dimension
        nhead: int = 8,            # Attention heads
        num_layers: int = 4,       # Encoder depth (was 2 in v1)
        dropout: float = 0.15,
    ):
        self.input_proj = nn.Linear(src_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,  # 1024
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            ),
            num_layers=num_layers,
        )
```

**Key Design Choices:**

| Choice | Rationale |
|--------|-----------|
| **4 layers (was 2)** | Deeper model captures hierarchical temporal patterns; 1.7M → 3.3M params but still tractable on 127K samples |
| **d_model = 256** | Balanced capacity vs. overfitting risk |
| **Sinusoidal position encoding** | Fixed (non-learned) encoding generalizes to arbitrary sequence lengths |
| **Pre-LayerNorm** | `norm_first=True` stabilizes training of deeper transformers |
| **Batch-first** | Matches PyTorch conventions; `[B, T, D]` tensors |

### 3.2 WPPEncoder (Stage A v3)

File: [`training/train_stage_a_v3.py`](training/train_stage_a_v3.py)

```python
class WPPEncoder(nn.Module):
    """Encoder + Word Presence Prediction head."""

    def __init__(self, ..., vocab_size: int = 150):
        self.encoder = TemporalVisualEncoder(...)
        self.head = nn.Linear(d_model, vocab_size)  # Binary classifier per word

    def forward(self, src, mask):
        seq_out, out_mask = self.encoder(src, mask)

        # Mean-pool over valid frames
        valid = out_mask.float().unsqueeze(-1)
        pooled = (seq_out * valid).sum(1) / valid.sum(1).clamp(min=1.0)

        logits = self.head(pooled)  # [B, 150]
        return logits, pooled
```

**Output Interpretation:**
- `logits[b, i]` > 0 → model predicts word `i` is present in the sentence
- `logits[b, i]` < 0 → model predicts word `i` is absent
- Applied to mean-pooled encoder output (aggregating evidence across all frames)

### 3.3 WPPTranslationModel (Stage C v2)

File: [`training/train_stage_c_v2.py`](training/train_stage_c_v2.py)

```python
class WPPTranslationModel(nn.Module):
    """Full translation model: encoder + bridge + T5 decoder."""

    def __init__(self, ...):
        # From Stage A checkpoint
        self.temporal_encoder = TemporalVisualEncoder(...)

        # T5 decoder (pretrained, partially frozen)
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        # Bridge: adapt encoder output to T5 embedding space
        self.bridge = nn.Sequential(
            nn.Linear(d_model, T5_D_MODEL),  # 256 → 512
            nn.GELU(),
            nn.LayerNorm(T5_D_MODEL),
        )

        # Retained for auxiliary loss
        self.wpp_head = nn.Linear(d_model, vocab_size)
```

**Bridge Purpose:**
- Encoder outputs 256-dim representations
- T5 expects 512-dim embeddings
- Bridge provides nonlinear adaptation with learned normalization

---

## 4. Training Pipeline

### 4.1 Stage A v3: WPP Pretraining

File: [`training/train_stage_a_v3.py`](training/train_stage_a_v3.py)

**Objective:** Teach the encoder to recognize which words appear in a sign sequence.

```python
# Primary loss: Multi-label binary classification
wpp_loss = F.binary_cross_entropy_with_logits(
    logits, labels,
    pos_weight=torch.full((150,), 10.0),  # Up-weight rare words
)

# Auxiliary loss: Prevent representation collapse
var_loss, cov_loss = soft_vicreg_loss(pooled)

# Combined
total_loss = wpp_loss + var_loss + cov_loss
```

**Training Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `WPP_LR` | 3e-4 | Aggressive LR for fast convergence on classification |
| `WPP_MAX_EPOCHS` | 60 | Cap to prevent overfitting |
| `WPP_EARLY_STOP_PATIENCE` | 15 | Stop if validation WPP loss plateaus |
| `WPP_BATCH_SIZE` | 128 | Large batches for stable gradient estimates |
| `FRAME_DROPOUT_RATE` | 0.15 | Augmentation: randomly drop frames |
| `JOINT_NOISE_STD` | 0.01 | Augmentation: add Gaussian noise |

**Learning Rate Schedule:**
```
Epoch 0-2:  Linear warmup (0 → 3e-4)
Epoch 3-60: Cosine decay (3e-4 → 1e-6)
```

### 4.2 Stage B v2: Quality Gate

File: [`probe_stage_b_v2.py`](probe_stage_b_v2.py)

**Purpose:** Validate that Stage A produced a usable encoder before investing compute in Stage C.

**Pass Criteria:**
```python
GATE_PASS = (
    words_above_auc_0_60 >= 25 and      # At least 25 words predictable
    isotropy_l1 < 0.50                   # Not collapsed to narrow cone
)
```

Why 0.62 AUC? Random guessing gives 0.50; 0.62 represents meaningful discrimination without being too stringent for noisy sign data.

### 4.3 Stage C v2: Translation Training

File: [`training/train_stage_c_v2.py`](training/train_stage_c_v2.py)

**Strategy:** Multi-task learning with gradually unfreezing the decoder.

```python
# Epoch 0-7:  T5 completely frozen
# Epoch 8+:   Unfreeze top-1 decoder block + final_layer_norm

# Loss with decaying auxiliary weight
loss = ce_loss + aux_weight * wpp_aux_loss

# aux_weight schedule: 0.8 → 0.2 over 40 epochs (cosine decay)
```

**Why Multi-Task?**
- Without WPP auxiliary: encoder drifts away from vocabulary awareness
- With WPP auxiliary: encoder is penalized if it stops encoding word presence
- This prevents the "hallucination attractor" where decoder relies solely on its language model prior

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LR_ENCODER` | 1e-4 | Warm encoder continues learning |
| `LR_DECODER` | 5e-6 (0.5× base) | Conservative to preserve T5's knowledge |
| `FREEZE_DECODER_EPOCHS` | 8 | Force encoder-bridge alignment before decoder influences gradients |
| `LABEL_SMOOTHING` | 0.1 | Reduces overconfidence on noisy alignments |

### 4.4 Stage D: Joint Fine-tuning

Conservative joint optimization of both encoder and decoder with:
- `LR_ENCODER_JOINT = 5e-6`
- `LR_DECODER_JOINT = 1e-5`

---

## 5. Key Technical Concepts Explained

### 5.1 WPP (Word Presence Prediction)

**What is it?**
A multi-label binary classification task where the model predicts which content words appear in the reference sentence, given only the pose sequence.

**Mathematical Formulation:**
```
Given:
  - Pose sequence X ∈ ℝ^(B×T×D)
  - Vocabulary V = {w₁, w₂, ..., wₙ} (top-150 content words)

Target:
  - Label vector y ∈ {0,1}^V where yᵢ = 1 iff word wᵢ ∈ sentence

Prediction:
  - z = MeanPool(Encoder(X)) ∈ ℝ^D
  - ŷ = σ(W · z) ∈ (0,1)^V  where σ is sigmoid
```

**Why It Works for ISL:**
1. **Granular supervision**: Each word provides a training signal (vs. one per sentence in contrastive learning)
2. **Alignment-robust**: Doesn't require precise temporal alignment of each sign to its word
3. **Grounded in linguistics**: Signs and words have approximate correspondence

**Comparison with Alternatives:**

| Approach | Pros | Cons |
|----------|------|------|
| **Sentence-level InfoNCE** | Natural for retrieval | Near-zero gradient on noisy data |
| **Frame-level alignment** | Precise | Requires expensive temporal annotations |
| **WPP** | Balance of granularity and practicality | Misses word order information |

### 5.2 BCE with Logits Loss

**Standard BCE Loss:**
```python
loss = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
```

**Numerical Instability Problem:**
- `σ(x)` can saturate at 0 or 1
- `log(0)` → -∞ → NaN gradients

**Solution: Log-Sum-Exp Trick**
```python
# BCEWithLogitsLoss combines sigmoid + BCE in numerically stable way
loss = max(x, 0) - x·y + log(1 + exp(-|x|))
```

**Positive Weighting:**
```python
pos_weight = 10.0  # Typical word appears in ~5% of sentences

# Effect: False negatives penalized 10× more than false positives
# This compensates for severe class imbalance (rare words)
```

### 5.3 VICReg Regularization

**Purpose:** Prevent representation collapse without requiring negative samples.

**The Collapse Problem:**
- Without regularization, the encoder can output identical vectors for all inputs
- Gradient descent finds this "attractor" because it minimizes loss trivially

**VICReg Components:**

#### Variance Term
```python
# For each dimension d, penalize if std < γ
centered = z - z.mean(dim=0)
std = sqrt(centered.var(dim=0) + 1e-4)
var_loss = mean(ReLU(γ - std))
```

**Intuition:** Forces each dimension to vary across the batch. If all samples have the same value in dimension d, variance = 0 → loss increases.

#### Covariance Term
```python
# Penalize correlation between dimensions
cov = (centered.T @ centered) / (B - 1)  # [D, D]
cov_loss = sum(cov[i,j]² for i ≠ j) / D
```

**Intuition:** Prevents dimensions from becoming correlated. Without this, all dimensions might learn to encode the same information redundantly.

**Soft VICReg (Our Variant):**
```python
VICREG_COEFF_VAR = 0.25   # Was 1.0 in v2
VICREG_COEFF_COV = 0.02
```

We use **weaker** regularization because:
1. Full-strength VICReg destroyed coarse lexical similarity in v2
2. WPP loss already provides strong content signal
3. We want geometric diversity, not orthogonality

### 5.4 AUC (Area Under ROC Curve)

**Definition:**
AUC measures a classifier's ability to distinguish between classes across all thresholds.

```
ROC Curve:
  - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
  - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)

AUC = Area under this curve ∈ [0, 1]
```

**Interpretation:**
| AUC | Meaning |
|-----|---------|
| 0.50 | Random guessing |
| 0.60 | Weak discrimination |
| 0.70 | Moderate discrimination |
| 0.80 | Strong discrimination |
| 1.00 | Perfect discrimination |

**Per-Word AUC in Stage B:**
```python
for word in vocab:
    y_true = [word in sentence for sentence in batch]
    y_score = model.predict_proba(batch)[:, word_idx]
    auc = roc_auc_score(y_true, y_score)
```

**Why AUC for the Gate?**
1. **Threshold-independent**: Doesn't require choosing a classification threshold
2. **Class-imbalance robust**: Works even when word appears rarely
3. **Probability-ranked**: Measures whether model ranks positives higher than negatives

### 5.5 Isotropy (Geometric Diversity)

**Definition:**
Measures how uniformly distributed embeddings are in the vector space.

**Calculation:**
```python
def compute_isotropy(embeddings):  # [N, D]
    # Normalize to unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms

    # Pairwise cosine similarities
    cos_matrix = normalized @ normalized.T  # [N, N]

    # Mean of off-diagonal (exclude self-similarity = 1.0)
    off_diag = cos_matrix[np.triu_indices_from(cos_matrix, k=1)]
    isotropy = np.mean(np.abs(off_diag))
    return isotropy
```

**Interpretation:**

| Isotropy | Geometry | Meaning |
|----------|----------|---------|
| ~1.0 | Collapsed | All embeddings nearly identical (bad) |
| 0.5-0.7 | Narrow cone | Similar direction, varying magnitude (problematic) |
| 0.1-0.3 | Spread | Diverse directions (good) |
| ~0.0 | Isotropic | Uniform distribution on sphere (ideal) |

**Why It Matters:**
- **Collapse (high isotropy):** Model outputs same vector regardless of input → no information
- **Narrow cone:** Model distinguishes samples by magnitude, not direction → limited expressiveness
- **Low isotropy:** Rich geometric structure enables downstream tasks to separate classes

**Gate Threshold (0.50):**
- Allows moderate clustering (similar signs nearby)
- Prevents severe collapse where all signs map to same region
- Balanced for 256-dimensional space with 127K samples

---

## 6. Configuration Reference

### v1 (Legacy — Frozen)
File: [`config/arch_v1_locked.py`](config/arch_v1_locked.py)

Used for InfoNCE-based training (Phases 1-8). Locked to maintain reproducibility of legacy experiments.

### v2 (Active)
File: [`config/arch_v2_redesign.py`](config/arch_v2_redesign.py)

| Constant | Value | Description |
|----------|-------|-------------|
| **ENCODER_LAYERS** | 4 | Transformer depth (was 2) |
| **ENCODER_DROPOUT** | 0.15 | Moderate regularization |
| **WPP_VOCAB_SIZE** | 150 | Content words to predict |
| **WPP_LR** | 3e-4 | Encoder pretraining LR |
| **VICREG_COEFF_VAR** | 0.25 | Soft variance penalty |
| **VICREG_COEFF_COV** | 0.02 | Covariance penalty |
| **GATE_MIN_AUC** | 0.62 | Minimum per-word AUC to pass |
| **GATE_MIN_WORDS_ABOVE_AUC** | 25 | Minimum passing words |
| **GATE_MAX_L1_ISOTROPY** | 0.50 | Maximum allowed collapse |
| **WPP_AUX_WEIGHT_INIT** | 0.5 | Starting auxiliary weight |
| **WPP_AUX_WEIGHT_END** | 0.05 | Final auxiliary weight |

---

## 7. Failure Analysis & Redesign History

### Phase 1-6: InfoNCE with MLP Projection (Failed)

**Architecture:**
- 2-layer encoder + MLP projection head with BatchNorm
- Sentence-level InfoNCE against T5 embeddings

**Failure:** 0/15 content words with lift > 1.15

**Root Cause:**
1. BatchNorm in projection head enabled collapse
2. Sentence-level InfoNCE unreliable on noisy alignments
3. T5 embeddings inappropriate as alignment target

### Phase 7-8: InfoNCE + VICReg (Failed)

**Change:** Removed BatchNorm, added VICReg regularization

**Result:**
- Collapse fixed (isotropy: 0.75 → 0.09)
- Content lift: 0.994 → 0.957 (still random)

**Insight:** Fixing geometry doesn't create semantic signal. The objective itself was wrong.

### Phase 9+ (Current): WPP + Soft VICReg

**Changes:**
1. Replaced InfoNCE with WPP multi-label classification
2. Removed T5 from pretraining
3. Deeper encoder (4 layers)
4. Softer VICReg (weight 0.25)
5. Multi-task translation training

**Expected Outcome:**
- Direct word-level supervision provides tractable learning signal
- No misaligned cross-modal alignment required
- Auxiliary loss preserves vocabulary awareness during translation

---

## References

1. Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
2. Raffel, C., et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." JMLR 2020.
3. Bardes, A., Ponce, J., & LeCun, Y. "VICReg: Variance-invariance-covariance regularization for self-supervised learning." ICLR 2022.
4. iSign Dataset Paper: "Real-Time Continuous Indian Sign Language Translation Using a Gloss-Free Spatial-Temporal Transformer"

---

## Appendix: Quick Commands

```bash
# Stage A: Encoder pretraining
python training/train_stage_a_v3.py --run-name stageA_v3_run1

# Stage B: Quality gate
python probe_stage_b_v2.py --checkpoint artifacts/phase9_encoder_pretrain/stageA_v3_run1/encoder_best.pt

# Stage C: Translation training
python training/train_stage_c_v2.py \
    --encoder-checkpoint artifacts/phase9_encoder_pretrain/stageA_v3_run1/encoder_best.pt \
    --run-name stageC_v2_run1

# Generate WPP vocabulary
python training/word_inventory.py
```
