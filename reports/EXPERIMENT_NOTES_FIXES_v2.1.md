# Stage C v2.1 — Implementation Fixes

**Date**: 2026-04-13  
**Status**: Implemented (pending evaluation)  
**Parent Document**: PIPELINE_REDESIGN.md, ROOT_CAUSE_ANALYSIS.md  

## Summary of Fixes Applied

This document records the 6 targeted fixes applied to address stability and decoder utilization issues in the Stage C v2 pipeline.

---

## Fix 1: Increase Decoder Unfreezing from 1 Layer to 2 Layers

**Location**: `training/train_stage_c_v2.py` (line 104)  
**Change**:
```python
# WAS:
DECODER_LAYERS_TO_UNFREEZE: int = 1

# NOW:
DECODER_LAYERS_TO_UNFREEZE: int = 2  # INCREASED for better decoder capacity
```

**Rationale**: The original 1-layer unfreezing proved too conservative. With only 1 decoder layer trainable, the decoder lacks sufficient capacity to adapt to the encoder's visual features. Increasing to 2 layers provides more expressive power while still keeping most of T5 frozen to prevent LM prior domination.

**Expected Impact**: Better BLEU scores, reduced generic hallucinations, more diverse output.

---

## Fix 2: Extend WPP Auxiliary Weight Floor

**Location**: `training/train_stage_c_v2.py` (lines 106-108)  
**Change**:
```python
# WAS: Decay from 0.8 -> 0.2 over 40 epochs
# Problem: End floor too low, encoder drifts away from lexical signal

# NOW:
STAGE_C_WPP_AUX_WEIGHT_INIT: float = 0.6
STAGE_C_WPP_AUX_WEIGHT_END: float = 0.3
STAGE_C_WPP_AUX_EPOCHS: int = 50
```

**Rationale**: The original 0.2 end-floor allowed the encoder to drift too early. By maintaining a higher floor (0.3) and extending the decay period (50 epochs), the encoder retains stronger vocabulary awareness throughout training, preventing the "hallucination attractor" where T5's LM prior dominates.

**Expected Impact**: Reduced generic_ratio, better preservation of Stage A lexical features.

---

## Fix 3: Add Gradient Accumulation

**Location**: `training/train_stage_c_v2.py` (new constant + `_run_train_epoch`)  
**Change**:
```python
# NEW CONSTANT:
GRAD_ACCUM_STEPS: int = 2  # Effective batch size = 32 * 2 = 64

# MODIFIED _run_train_epoch to accumulate gradients:
loss = loss / grad_accum_steps  # Scale for accumulation
loss.backward()

# Step optimizer only after accumulating:
if accum_count >= grad_accum_steps or step == total_steps:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Rationale**: Larger effective batch sizes improve gradient estimate stability. Without gradient accumulation, batch size 32 is the GPU memory limit. With accumulation, we achieve effective batch size 64 without memory issues.

**Expected Impact**: Smoother training curves, better convergence, reduced noise in updates.

---

## Fix 4: Add Plateau-Based Learning Rate Reduction

**Location**: `training/train_stage_c_v2.py` (training loop)  
**Change**:
```python
# NEW CONSTANTS:
PLATEAU_PATIENCE: int = 5
PLATEAU_FACTOR: float = 0.5

# ADDED STATE TRACKING:
epochs_since_improvement = 0
current_plateau_factor = 1.0

# ADDED LOGIC ON NON-IMPROVEMENT:
if bleu > best_bleu:
    epochs_since_improvement = 0  # Reset on improvement
else:
    epochs_since_improvement += 1

# TRIGGER REDUCTION:
if epochs_since_improvement >= PLATEAU_PATIENCE:
    current_plateau_factor *= PLATEAU_FACTOR
    print(f"LR plateau detected — reducing by factor {PLATEAU_FACTOR}")
```

**Rationale**: Cosine annealing alone may not react quickly enough to validation plateaus. Plateau-based reduction acts as an adaptive safety valve, reducing LR when metrics stagnate regardless of the cosine schedule.

**Expected Impact**: Better escape from local minima, continued improvement when cosine schedule would stall.

---

## Fix 5: Make Bridge Mask-Aware

**Location**: `training/train_stage_c_v2.py` (`WPPTranslationModel`)  
**Change**:
```python
# WAS: Sequential bridge (mask-unaware)
self.bridge = nn.Sequential(
    nn.Linear(d_model, self.t5.config.d_model),
    nn.GELU(),
    nn.LayerNorm(self.t5.config.d_model),
)

# NOW: Split components with explicit masking
self.bridge_projection = nn.Linear(d_model, self.t5.config.d_model)
self.bridge_activation = nn.GELU()
self.bridge_norm = nn.LayerNorm(self.t5.config.d_model)

# In encode_visual():
bridged = self.bridge_projection(seq_out)
bridged = self.bridge_activation(bridged)
bridged = bridged * out_mask.unsqueeze(-1).float()  # Mask before norm
bridged = self.bridge_norm(bridged)
bridged = bridged * out_mask.unsqueeze(-1).float()  # Mask after norm
```

**Rationale**: The original Sequential bridge had no masking. Padded positions could leak information through LayerNorm statistics (mean/variance computed over all positions) and affect the decoder cross-attention. Explicit masking ensures zero contribution from padded frames.

**Expected Impact**: Cleaner encoder-decoder interface, reduced noise from padding artifacts.

---

## Fix 6: Preserve Old Checkpoints and Logs

**Implementation**: New runs use distinct `run_name` parameters  
**Old runs preserved at**:
- `artifacts/phase9_decoder_finetune/stageC_v2/` (original v2 run)

**New runs saved to**:
- `artifacts/phase9_decoder_finetune/stageC_v2_fixes/` (this fix run)

**Rationale**: By changing the `--run-name` argument when launching training, we ensure old checkpoints remain intact for comparison. The training script never overwrites existing checkpoints.

---

## Hyperparameter Summary Table

| Parameter | v2 (Old) | v2.1 (New) | Change |
|-----------|----------|------------|--------|
| DECODER_LAYERS_TO_UNFREEZE | 1 | 2 | +1 layer |
| WPP_AUX_WEIGHT_INIT | 0.8 | 0.6 | -0.2 |
| WPP_AUX_WEIGHT_END | 0.2 | 0.3 | +0.1 |
| WPP_AUX_EPOCHS | 40 | 50 | +10 epochs |
| GRAD_ACCUM_STEPS | 1 | 2 | New |
| PLATEAU_PATIENCE | N/A | 5 | New |
| PLATEAU_FACTOR | N/A | 0.5 | New |
| Bridge masking | No | Yes | New |

---

## Expected Outcomes

Based on the systematic nature of these fixes, we expect:

1. **BLEU-4**: Modest increase (baseline ~1.0-1.5 → target > 1.47)
2. **ROUGE-L**: Modest increase (baseline ~12-15 → target > 16.67)
3. **dominant_ratio**: Decrease (less collapse)
4. **generic_ratio**: Decrease (fewer "he said that he" hallucinations)

---

## Evaluation Protocol

Run evaluation using the benchmark script:

```bash
# Evaluate on validation set
python evaluate_stage_c_benchmark.py \
    --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_fixes/best_bleu_model.pt \
    --split val

# Final test evaluation (only after validation looks good)
python evaluate_stage_c_benchmark.py \
    --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_fixes/best_bleu_model.pt \
    --split test
```

Metrics will be compared against iSign paper baselines:
- **BLEU-4**: > 1.47 to beat paper
- **ROUGE-L**: > 16.67 to beat paper

---

## Rollback Plan

If results degrade:
1. Revert individual fixes by restoring from git
2. Rerun ablation studies with subsets of fixes
3. Compare checkpoints side-by-side using `compare_stage_c_runs.py`

---

## Notes

- All changes maintain WPP-based architecture (not copying iSign paper)
- Pipeline structure unchanged (Stage A → B → C flow preserved)
- Changes are additive/minimal, not architectural rewrites
- Experiment notes updated after each change as required
