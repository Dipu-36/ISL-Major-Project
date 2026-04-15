# Benchmark Evaluation Protocol

This document defines the locked evaluation protocol for fair comparison against the iSign paper baseline (BLEU-4 1.47, ROUGE-L 16.67).

## Goal

Achieve BLEU-4 > 1.47 on the iSign SignPose-to-English translation task using our improved architecture.

## Dataset

- **Source**: iSign dataset v1.1 (118,228 video-sentence pairs)
- **Splits**: Defined in `artifacts/splits.json`
  - Train: ~80% of data
  - Validation: ~10% of data
  - Test: ~10% of data (held out, not used during development)

## Split Definitions

Splits are loaded from `artifacts/splits.json` with the following keys:
- `train_uids` or `train`: Training set UIDs
- `val_uids` or `val`: Validation set UIDs
- `test_uids` or `test`: Test set UIDs (for final evaluation only)

**Protocol**: Use ALL available samples in each split unless explicitly subsampling for debugging.

## Preprocessing Protocol

### Input Features
- **Body keypoints**: Indices 0-33 (34 joints)
- **Left hand**: Indices 501-522 (21 joints)
- **Right hand**: Indices 522-543 (21 joints)
- **Total joints**: 75 joints (450 dims with velocity)
- **Face/head**: EXCLUDED (known limitation, potential improvement area)

### Normalization
- Shoulder-centered: Subtract midpoint of left/right shoulders
- Scale by shoulder width (L2 distance between shoulders)
- Fallback to median scale if shoulders coincide

### Temporal Processing
- Target: 96 frames
- **Short clips** (< 96 frames): Right-pad with last frame
- **Long clips** (> 96 frames): Motion-based top-K selection
- **Attention mask**: Binary mask (1=valid, 0=padding) based on original clip length

### Text Preprocessing
- Lowercase conversion
- Keep: a-z, 0-9, apostrophe, period, comma, question mark, exclamation, hyphen, space
- Collapse multiple spaces to single space
- Trim to max 256 characters

## Training Protocol

### Stage A (Encoder Pretraining)
- Objective: Word Presence Prediction (WPP)
- Vocabulary: Top 150 content words (freq >= 50)
- Encoder: 4-layer Transformer, d=256, 8 heads
- Batch size: 128
- Learning rate: 3e-4 with cosine decay to 1e-6
- Max epochs: 60
- Early stopping: 15 epochs patience on validation WPP loss

### Stage B (Gate)
- Requirement: >= 25 words achieve AUC >= 0.62
- Secondary: Mean L1 isotropy <= 0.50

### Stage C (Translation Training)
- Loss: CE_loss + aux_weight * WPP_loss
- Aux weight: Cosine decay from 0.5 to 0.05 over 30 epochs
- Encoder LR: 1e-4
- Decoder LR: 1e-5
- Decoder freeze: First 8 epochs frozen, then unfreeze top-1 block
- Batch size: 32
- Max epochs: 50
- Early stopping: 10 epochs patience on validation BLEU

### Architecture Constants (arch_v2_redesign.py)
```python
T5_NAME = "t5-small"  # TODO: Consider upgrade to t5-base
ENCODER_LAYERS = 4
D_MODEL = 256
TARGET_FRAMES = 96
```

## Evaluation Protocol

### Metrics
- **BLEU**: BLEU-1, BLEU-2, BLEU-3, BLEU-4 (primary metric)
- **ROUGE**: ROUGE-L (secondary metric)
- Computation: `evaluate` library (HuggingFace)

### Validation Evaluation
- Decode ALL validation samples (no subsampling)
- Compute metrics on full validation set
- Select best checkpoint by validation BLEU-4

### Test Evaluation (Final Only)
- Run once on test set after final model selection
- Report BLEU-4 and ROUGE-L
- Do not tune hyperparameters on test set

## Known Issues Fixed

1. **Masking** (FIXED): Attention mask now correctly indicates valid vs padded frames
2. **Subset evaluation** (FIXED): Defaults changed to use full datasets
3. **Limited BLEU computation** (FIXED): Decode all validation samples

## Experiment Tracking

- Save checkpoints to: `artifacts/phase9_decoder_finetune/<run_name>/`
- Save metrics to: `artifacts/phase9_decoder_finetune/<run_name>/history.json`
- Log sample predictions for inspection

## Reproducibility

- Seed: 42 (default)
- Deterministic operations where possible
- Record exact commit hash for each run
- Document any deviations from this protocol

## Target

Beat iSign paper baseline:
- **Target BLEU-4**: > 1.47
- **Target ROUGE-L**: > 16.67
