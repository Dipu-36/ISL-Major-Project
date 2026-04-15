"""Architecture constants for the redesigned ISL pipeline — v2.

PURPOSE
-------
This config governs the WPP-based (Word Presence Prediction) pretraining
pipeline (Stage A v3) and multi-task translation pipeline (Stage C v2).

Design rationale is documented in reports/ROOT_CAUSE_ANALYSIS.md and
reports/PIPELINE_REDESIGN.md.  Do NOT mix v1 and v2 constants mid-run.

KEY DIFFERENCES FROM arch_v1_locked.py
---------------------------------------
- Training objective: WPP multi-label classification (not InfoNCE)
- Encoder depth: 4 layers (was 2) — intermediate capacity increase
- Pre-training uses NO T5 at all — encoder is self-contained
- Translation training (Stage C) adds WPP as auxiliary loss
- Stage B gate is AUC-based (not content-lift based)
- WPP vocabulary: top-150 content words (not 30/15 probes)
"""
from __future__ import annotations

# ── Visual encoder ─────────────────────────────────────────────────────────────
SRC_DIM: int = 450          # pose keypoint dimensionality (75 joints × 6: xyz + velocity)
TARGET_FRAMES: int = 96     # temporal resolution (unchanged from v1)
ENCODER_LAYERS: int = 4     # deeper encoder (was 2 — insufficient for lexical features)
D_MODEL: int = 256          # hidden dim (unchanged; wider would risk overfit on 127k samples)
NHEAD: int = 8              # attention heads
DIM_FEEDFORWARD: int = 1024 # FFN = D_MODEL × 4
ENCODER_DROPOUT: float = 0.15  # moderate dropout (0.10 in v1, 0.25 in v2 was too strong)

# ── T5 backbone (Stage C only — NOT used in Stage A) ──────────────────────────
T5_NAME: str = "t5-small"
T5_D_MODEL: int = 512
BRIDGE_IN: int = D_MODEL
BRIDGE_OUT: int = T5_D_MODEL

# ── WPP (Word Presence Prediction) pretraining — Stage A v3 ──────────────────
# Vocabulary
WPP_VOCAB_SIZE: int = 150           # number of content words to predict
WPP_MIN_WORD_FREQ: int = 50         # min occurrences in train set to include a word
WPP_VOCAB_CACHE: str = "artifacts/wpp_vocab.json"  # cached vocabulary file

# Training
WPP_LR: float = 3e-4                # encoder + classification head LR
WPP_LR_MIN: float = 1e-6            # cosine decay floor
WPP_WARMUP_EPOCHS: int = 3          # linear LR warmup
WPP_MAX_EPOCHS: int = 60            # maximum Stage A epochs
WPP_EARLY_STOP_PATIENCE: int = 15   # patience on val WPP loss
WPP_BATCH_SIZE: int = 128
WPP_WEIGHT_DECAY: float = 1e-2

# Regularisation to prevent collapse during WPP training
# Using SOFT VICReg (weaker than v2 to preserve coarse structure)
VICREG_COEFF_VAR: float = 0.25      # variance regulariser (was 1.0 in v2 — too aggressive)
VICREG_COEFF_COV: float = 0.02      # covariance regulariser
VICREG_GAMMA: float = 1.0           # target std per dimension

# ── Stage B gate v2 — WPP AUC based ──────────────────────────────────────────
# Gate passes if ≥ GATE_MIN_WORDS_ABOVE_THRESHOLD words achieve AUC ≥ GATE_MIN_AUC
GATE_MIN_AUC: float = 0.62          # above 0.50 (random) with meaningful margin
GATE_MIN_WORDS_ABOVE_AUC: int = 25  # out of top WPP_VOCAB_SIZE words

# Secondary gate: isotropy check (enforce some geometric diversity)
GATE_MAX_L1_ISOTROPY: float = 0.50  # mean pairwise cosine similarity at encoder output

# ── Stage C v2 — multi-task translation + WPP auxiliary ───────────────────────
LR_ENCODER: float = 1e-4            # encoder during Stage C (warm from Stage A)
LR_DECODER: float = 1e-5            # T5 decoder (conservative)
DECODER_LAYERS_UNFREEZE: int = 2    # top-2 T5 decoder blocks

# WPP auxiliary loss weight schedule during translation training
WPP_AUX_WEIGHT_INIT: float = 0.5   # initial WPP weight relative to CE loss
WPP_AUX_WEIGHT_END: float = 0.05   # final WPP weight (cosine decay over training)
WPP_AUX_EPOCHS: int = 30           # epochs over which WPP weight anneals

# ── Stage D — joint fine-tuning (same as v1) ──────────────────────────────────
LR_ENCODER_JOINT: float = 5e-6
LR_DECODER_JOINT: float = 1e-5

# ── Augmentation ──────────────────────────────────────────────────────────────
# Frame dropout: randomly zero a fraction of valid frames during pretraining
FRAME_DROPOUT_RATE: float = 0.15    # 15% of frames zeroed per sample

# Joint noise: small Gaussian noise added to keypoints during training
JOINT_NOISE_STD: float = 0.01       # std of Gaussian noise in normalised coordinates
