"""Locked architecture constants for ISL translation — v1 lineage.

These values are frozen for the entire training lineage starting at Stage A.
No code may override them mid-lineage.  Any future lineage must use a new
versioned file (arch_v2_locked.py, etc.).

Architecture spec (from human specification):
  src_dim          = 450       (pose keypoint dimensionality)
  target_frames    = 96        (temporal resolution after preprocessing)
  encoder_layers   = 2         (TemporalVisualEncoder depth)
  d_model          = 256       (TemporalVisualEncoder hidden dimension)
  nhead            = 8         (multi-head attention heads)
  dim_feedforward  = 1024      (FFN width = d_model * 4)
  encoder_dropout  = 0.1       (transformer dropout rate)
  bridge           = Linear(256 → 512) + GELU + LayerNorm(512)
  t5_name          = "t5-small"  (d_model=512 matches bridge output)
  freeze_t5        = True      (T5 fully frozen during encoder pretraining)

Stage A projection head:
  proj_dim         = 256       (contrastive projection dimension)
  temperature_init = 0.10      (InfoNCE temperature at epoch 1)
  temperature_end  = 0.07      (InfoNCE temperature after warmup_temp_epochs)
  warmup_temp_epochs = 15      (linear temperature annealing epochs)

Stage C learning rates:
  lr_decoder       = 1e-5      (top-2 T5 decoder layers only)
  decoder_layers_unfreeze = 2  (number of top decoder layers to unfreeze)
"""
from __future__ import annotations

# ── Visual encoder ────────────────────────────────────────────────────────────
SRC_DIM: int = 450
TARGET_FRAMES: int = 96
ENCODER_LAYERS: int = 2
D_MODEL: int = 256
NHEAD: int = 8
DIM_FEEDFORWARD: int = 1024   # must equal D_MODEL * 4
ENCODER_DROPOUT: float = 0.1

# ── T5 backbone ───────────────────────────────────────────────────────────────
T5_NAME: str = "t5-small"
T5_D_MODEL: int = 512         # t5-small's internal d_model; bridge maps D_MODEL → T5_D_MODEL

# ── Bridge (D_MODEL → T5_D_MODEL) ────────────────────────────────────────────
# Realised as: nn.Linear(D_MODEL, T5_D_MODEL) → nn.GELU() → nn.LayerNorm(T5_D_MODEL)
BRIDGE_IN: int = D_MODEL
BRIDGE_OUT: int = T5_D_MODEL

# ── Stage A: contrastive pretraining ─────────────────────────────────────────
PROJ_DIM: int = 256
TEMPERATURE_INIT: float = 0.10
TEMPERATURE_END: float = 0.07
WARMUP_TEMP_EPOCHS: int = 15

# ── Stage C: decoder fine-tuning ─────────────────────────────────────────────
LR_DECODER: float = 1e-5
DECODER_LAYERS_UNFREEZE: int = 2

# ── Stage D: joint fine-tuning ────────────────────────────────────────────────
LR_ENCODER_JOINT: float = 5e-6
LR_DECODER_JOINT: float = 1e-5

# ── Probe gate (Stage B) ──────────────────────────────────────────────────────
PROBE_GATE_MIN_WORDS_ABOVE_LIFT: int = 10   # ≥10 of top-30 content words must exceed:
PROBE_GATE_MIN_LIFT: float = 1.15           # this lift threshold to proceed to Stage C
