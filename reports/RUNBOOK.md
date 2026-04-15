# ISL Project — Runbook

**Date:** April 8, 2026  
**Environment:** `isl_transformer` conda environment on GPU server  
**Root:** `/data/aicoe_gpu/debdutta_pal_n2a/ISL_Project`

---

## Prerequisites

```bash
# Activate environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate isl_transformer

# All commands are run from project root
cd /data/aicoe_gpu/debdutta_pal_n2a/ISL_Project
```

---

## Step 0 — Build Cache (if not already done)

```bash
# Check if cache exists
ls artifacts/dataset_cache/isl_tf96.pt

# If missing, build it (30–60 min first time)
./run_build_cache.sh
# OR directly:
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  build_cache.py --target-frames 96
```

---

## Step 1 — Build WPP Vocabulary

```bash
# Build top-150 content word vocabulary from iSign training split
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  training/word_inventory.py \
  --vocab-size 150 \
  --min-freq 50

# Verify output
python3 -c "import json; d=json.load(open('artifacts/wpp_vocab.json')); print(f\"Vocab size: {d['vocab_size']}\"); print('Top-20:', d['vocab'][:20])"
```

**Expected output:** vocab_size=100–150, top words: "people", "india", "year", "government", etc.

---

## Step 2 — Stage A v3: WPP Encoder Pretraining

```bash
# Run Stage A v3 (WPP-based pretraining, ~6–10 GPU-hours on A100)
./run_stageA_v3.sh

# OR directly (for custom options):
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  training/train_stage_a_v3.py \
  --run-name stageA_v3 \
  --train-samples 50000 \
  --val-samples   2000 \
  --batch-size    128 \
  --max-epochs    60 \
  --probe-every   5 \
  --seed          42 \
  --num-workers   4
```

**Monitor progress:**
```bash
tail -f logs/stageA_stageA_v3_*.log | grep "probe\|Epoch\|TRAINING"
```

**Expected output at epoch 20 (healthy sign):**
```
  Epoch  20 | train_wpp=0.082 val_wpp=0.089 | 
    probe: avg_auc=0.617 words≥0.60=28 iso=0.311 | lr=2.3e-04 | 210s
```

**Troubleshooting:**
- If avg_auc plateaus < 0.55 after 30 epochs: try `--batch-size 256` (more negatives per batch)
- If isotropy stays > 0.60: softVICReg may be too weak; increase `VICREG_COEFF_VAR` in `config/arch_v2_redesign.py`
- If val WPP loss increases (overfit): reduce ENCODER_LAYERS to 2 or increase ENCODER_DROPOUT to 0.20

---

## Step 3 — Stage B v2: Probe Gate

```bash
# Run the WPP-based gate probe
./run_stageB_v2.sh

# OR directly:
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  probe_stage_b_v2.py \
  --checkpoint artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt \
  --val-samples 5000
```

**Check results:**
```bash
python3 -c "
import json
d = json.load(open('artifacts/phase9_encoder_pretrain/stageA_v3/stage_b_v2_probe.json'))
print('Gate passed:', d['gate_passed'])
print('Words above AUC 0.62:', d['words_above_auc_threshold'])
print('Avg AUC:', d['avg_auc'])
print('Isotropy:', d['isotropy']['mean_pairwise_cos'])
print('Top 10:', list(d['top_20_words_by_auc'].items())[:10])
"
```

**If gate fails:** Do NOT run Stage C. Re-examine Stage A training curve in `history.json`.

---

## Step 4 — Stage C v2: Multi-task Translation Training

```bash
# Only after Stage B v2 gate passes (exit 0)
./run_stageC_v2.sh

# OR directly:
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  training/train_stage_c_v2.py \
  --encoder-checkpoint artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt \
  --run-name     stageC_v2 \
  --train-samples 50000 \
  --val-samples   1000 \
  --batch-size    32 \
  --seed          42
```

**Monitor for hallucination:**
```bash
# Look for "sample_preds" in history.json — they should vary across inputs
python3 -c "
import json
hist = json.load(open('artifacts/phase9_decoder_finetune/stageC_v2/history.json'))
for ep in hist[-3:]:
    print(f\"Epoch {ep['epoch']}: BLEU={ep.get('bleu',0):.2f} generic={ep.get('generic_ratio',0):.2f}\")
    for p in ep.get('sample_preds', [])[:2]:
        print(f'  pred: {p[:80]}')
"
```

**Success criteria (should appear within 10–15 epochs):**
- `generic_ratio` drops below 0.3 (was 0.80+ in prior phases)
- `bleu` shows any positive trend (even 0.5 → 1.0 is progress)
- Sample predictions show vocabulary variation (not all the same template)

---

## Step 5 — Stage D: Joint Fine-tuning (if Stage C succeeds)

```bash
# Update the checkpoint path in run_stageD.sh to stageC_v2
# Then run:
./run_stageD.sh
```

**Note:** `run_stageD.sh` (existing) reads from a JSON gate and checks BLEU > 1.0. Update the checkpoint lookup path at the top of that script.

---

## Step 6 — Comparison Against Prior Runs

```bash
# Compare Stage C v2 vs prior phases
python3 -c "
import json, os, glob

def summarize(path):
    try:
        hist = json.load(open(path))
        best = max(hist, key=lambda x: x.get('bleu', 0))
        return {'path': path, 'best_bleu': best.get('bleu', 0), 'epoch': best.get('epoch')}
    except Exception as e:
        return {'path': path, 'error': str(e)}

paths = glob.glob('artifacts/*/stageC*/history.json') + \
        glob.glob('artifacts/phase9_decoder_finetune/*/history.json')
for p in sorted(paths):
    print(summarize(p))
"
```

---

## Emergency: Reset to Fresh Stage A

If Stage C is stuck in a hallucination attractor and the encoder has drifted:

```bash
# DO NOT re-run Stage C from a bad checkpoint
# Re-run Stage A fresh with a new run name
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  training/train_stage_a_v3.py \
  --run-name stageA_v3b \
  --seed 1234

# Then probe again
/data/aicoe_gpu/debdutta_pal_n2a/miniconda3/envs/isl_transformer/bin/python \
  probe_stage_b_v2.py \
  --checkpoint artifacts/phase9_encoder_pretrain/stageA_v3b/encoder_best.pt
```

---

## File Reference

| File | Role |
|------|------|
| `config/arch_v2_redesign.py` | All architecture and training constants for v3 pipeline |
| `training/word_inventory.py` | Build/load WPP vocabulary |
| `training/train_stage_a_v3.py` | Stage A v3 training script |
| `probe_stage_b_v2.py` | Stage B v2 gate probe |
| `training/train_stage_c_v2.py` | Stage C v2 multi-task translation script |
| `run_stageA_v3.sh` | Stage A launcher |
| `run_stageB_v2.sh` | Stage B launcher |
| `run_stageC_v2.sh` | Stage C launcher |
| `artifacts/wpp_vocab.json` | WPP vocabulary (auto-built) |
| `artifacts/phase9_encoder_pretrain/stageA_v3/` | Stage A v3 outputs |
| `artifacts/phase9_decoder_finetune/stageC_v2/` | Stage C v2 outputs |
