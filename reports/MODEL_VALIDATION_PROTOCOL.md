# Model Validation & Testing Protocol

**Last Updated:** April 15, 2026  
**Dataset:** iSign v1.1  
**Evaluation Framework:** SacreBLEU + ROUGE (with HuggingFace evaluate library)

---

## 1. Validation Metrics

### Primary Metrics

#### 1.1 BLEU-4 (Bilingual Evaluation Understudy)
**Measure:** Corpus-level 4-gram precision with brevity penalty  
**Computation:** `evaluate.load('bleu').compute(predictions, references)` (SacreBLEU)  
**Range:** [0, 100]  
**Interpretation:**
- **0.0–0.5:** Minimal phrase-level overlap; model generating mostly off-topic content
- **0.5–1.5:** Low-frequency 4-grams matching; weak lexical grounding
- **1.5–5.0:** Meaningful phrase-level alignment; model grounding input-specific content
- **> 5.0:** Strong lexical alignment; model translating reliably structured content

**Why it matters:** BLEU directly measures the benchmark target. Inaccessibility of BLEU > 1.5 frequently indicates the encoder is not providing discriminative signal.

**Benchmark Target:** BLEU-4 > **1.47** (iSign baseline: SignPose2Text T5-base+MediaPipe-75)

#### 1.2 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
**Measure:** F₁ score of longest common subsequence (LCS) between prediction and reference  
**Computation:** `evaluate.load('rouge').compute(predictions, references, rouge_types=['rougeL'])` with `use_stemmer=True`  
**Range:** [0, 100]  
**Interpretation:**
- **0.0–5.0:** Very poor content overlap; hallucination mode
- **5.0–10.0:** Partial structural match; some content words appear out of order
- **10.0–15.0:** Good content preservation; model recovering key entities
- **15.0–20.0:** Strong structural alignment; approaching baseline

**Why it matters:** ROUGE-L is more forgiving than BLEU to paraphrasing and word order variation, capturing broader semantic alignment. Primary metric for this project.

**Benchmark Target:** ROUGE-L > **16.67** (iSign baseline)

#### 1.3 CHRF (Character F-score)
**Measure:** Macro-averaged F-score of character n-grams (n=1..6)  
**Computation:** `evaluate.load('chrf').compute(predictions, references)`  
**Range:** [0, 100]  
**Interpretation:**
- **0.0–10.0:** Character-level mismatch; mostly off-topic
- **10.0–20.0:** Weak morphological overlap
- **20.0–30.0:** Good character-level agreement; similar word content

**Why it matters:** CHRF operates sub-word and is robust to tokenization differences, providing a smooth signal even when BLEU stalls.

---

### Secondary Metrics (Output Distribution Health)

#### 2.1 Dominant Prediction Ratio
**Measure:** Fraction of validation set where the single most-common predicted string appears  
**Computation:** `mode_count / len(predictions)`  
**Range:** [0, 1]  
**Threshold:** < 0.10 (healthy)  
**Interpretation:**
- **> 0.5:** Mode collapse; model generates one "safe" sentence regardless of input
- **0.1–0.5:** High concentration; model has learned only a few common templates
- **< 0.1:** Healthy diversity; model generates varied outputs per input

**Monitoring:** Check every epoch. Spike above 0.3 during training indicates mode collapse starting.

#### 2.2 Unique Prediction Ratio
**Measure:** Fraction of validation predictions that are unique strings  
**Computation:** `len(set(predictions)) / len(predictions)`  
**Range:** [0, 1]  
**Threshold:** > 0.40 (healthy)  
**Interpretation:**
- **< 0.2:** Severe mode collapse; < 20% of predictions are distinct
- **0.2–0.4:** Moderate collapse; model uses repeated templates
- **> 0.4:** Healthy diversity; sufficient variation in outputs

**Trend:** Phase 6 achieved 0.40; Phase 7 improved to 0.48.

#### 2.3 Generic Sentence Ratio
**Measure:** Fraction of predictions matching hard-coded list of generic/fluent filler phrases  
**Generic templates included:**
```
"he said that he would not be able to do this"
"... part of the bjp"
"... proud of his achievements"
"the government of india"
"he also said that ..."
```
(Full list in `utils/metrics.py`)

**Computation:** Check each prediction against generic phrase list (substring match)  
**Range:** [0, 1]  
**Threshold:** < 0.30 (healthy)  
**Interpretation:**
- **> 0.8:** Severe hallucination; model defaulting to high-frequency training phrases
- **0.3–0.8:** Partial hallucination; mixture of grounded and generic outputs
- **< 0.3:** Healthy signal; most outputs are input-differentiated

**Why it matters:** Generic outputs are semantically unrelated to the input sign sequence, indicating decoder language model takeover.

#### 2.4 Input Difference Ratio
**Measure:** Fraction of validation samples where zeroing the visual input produces a different predicted sentence  
**Procedure:**
1. Decode with real visual features → pred_real
2. Replace visual encoder input with zeros → pred_null
3. Check if pred_real ≠ pred_null
4. Ratio = (# different) / (# total)

**Range:** [0, 1]  
**Threshold:** > 0.95 (healthy)  
**Interpretation:**
- **< 0.5:** Model ignoring encoder; pure language model generation
- **0.5–0.95:** Partial input dependence; some samples affected by visual signal
- **> 0.95:** Model is input-dependent; visual encoder carries meaningful signal

**Critical:** This is the **input-grounding test**. Any ratio < 1.0 is a red flag for encoder collapse.

**Historical:** All phases have achieved input_diff > 0.99, indicating the encoder is never completely ignored.

---

## 2. Model Checkpointing Strategy

### Checkpoint Types

| Type | Selection Criterion | Use Case |
|------|---|---|
| **best_bleu_model.pt** | Highest validation BLEU-4 | Final submission; directly targets benchmark |
| **best_loss_model.pt** | Lowest validation CE loss | Fallback if BLEU is erratic; often oversmoothed |
| **latest.pt** | Final epoch | Debugging; recovery point if training interrupted |

### Checkpoint Decision Logic

```python
for epoch in range(max_epochs):
    metrics = validate(model, val_loader)
    
    if metrics['bleu'] > best_bleu:
        best_bleu = metrics['bleu']
        save("best_bleu_model.pt")
        bleu_plateau_counter = 0
    else:
        bleu_plateau_counter += 1
    
    # Early stopping by BLEU
    if bleu_plateau_counter >= PATIENCE:
        break
    
    # Checkpoint by loss (secondary)
    if metrics['val_loss'] < best_loss:
        best_loss = metrics['val_loss']
        save("best_loss_model.pt")
    
    save("latest.pt")
```

### When to Promote to Next Stage

**Stage A → Stage B:**
- Checkpoint: `artifacts/phase9_encoder_pretrain/stageA_v3/encoder_best.pt`
- Criterion: avg_auc ≥ 0.58, words_above_threshold ≥ 25
- Gate script: `python probe_stage_b_v2.py --checkpoint <encoder_best.pt>`

**Stage B → Stage C:**
- Criterion: Gate return code = 0 (passed)
- Loads encoder from Stage A best checkpoint

**Stage C → Stage D (planned):**
- Criterion: BLEU ≥ 1.0 on 50k training data
- Loads `best_bleu_model.pt` from Stage C
- Candidate from current run: epoch 18 (BLEU 1.39)

---

## 3. Validation Protocol

### Validation Frequency & Coverage

| Stage | Validation Every | Decode | Metrics |
|-------|---|---|---|
| **Stage A** | Every epoch | — (not translation) | WPP loss, probe (AUC, isotropy) |
| **Stage B** | Once (after Stage A) | No | AUC per word, isotropy check |
| **Stage C** | Every epoch | Yes, all samples | BLEU, ROUGE-L, CHRF, generic ratio, dominant ratio |

### Stage C Validation Pipeline

```python
# 1. Load model (best checkpoint, or current epoch)
model = load_checkpoint("best_bleu_model.pt")
model.eval()

# 2. Decode all validation samples (2,000 samples)
predictions = []
for batch in val_loader:
    src, mask = batch['src'], batch['mask']
    with torch.no_grad():
        outputs = model.generate(
            src, attention_mask=mask,
            num_beams=4, max_length=128, min_length=5,
            length_penalty=0.6, no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    predictions.extend(decode_tokens(outputs))

# 3. Compute corpus-level metrics
bleu = evaluate.load('bleu').compute(
    predictions=predictions,
    references=references,
    smooth_method='exp'
)
rouge = evaluate.load('rouge').compute(
    predictions=predictions,
    references=references,
    rouge_types=['rougeL'],
    use_stemmer=True
)

# 4. Compute secondary metrics
generic_ratio = compute_generic_ratio(predictions)
dominant_ratio = compute_dominant_ratio(predictions)
unique_ratio = compute_unique_ratio(predictions)

# 5. Save per-sample analysis
history.append({
    'epoch': epoch,
    'bleu': bleu['bleu'],
    'rouge_l': rouge['rougeL'],
    'generic_ratio': generic_ratio,
    'dominant_ratio': dominant_ratio,
    'sample_preds': predictions[:20],  # Qualitative inspection
    'timestamp': now()
})
```

### Validation Data Split

- **Source:** `artifacts/splits.json['val']`
- **Selected:** 2,000 random samples from val SUIDs (10% of ~20k val samples)
- **Consistency:** Fixed seed (42) ensures same validation set across runs
- **Full validation:** Stage D may use full val set (~20k) for final assessment

---

## 4. Inference & Decoding Parameters

### Beam Search Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **num_beams** | 4 | Balance speed (A100 ~10s per 2k samples) vs. quality |
| **max_length** | 128 | ISL sentences are typically < 100 tokens; 128 is safe cap |
| **min_length** | 5 | Prevent trivial short outputs; most ISL require at least 5 tokens |
| **length_penalty** | 0.6 | Slightly favor longer outputs (T5 default is 1.0, too long) |
| **no_repeat_ngram_size** | 3 | Prevent same 3-gram appearing twice |
| **repetition_penalty** | 1.2 | Multiplicative penalty on already-selected tokens |
| **early_stopping** | True | Stop beam search when all beams reach EOS |

**Ablation:** Experimented with `num_beams=2` (faster) and `num_beams=8` (higher quality); found 4 to be practical sweet spot.

---

## 5. Quality Gates & Thresholds

### Per-Epoch Quality Checks

**During training, monitor these per epoch to catch failures early:**

```python
if epoch % 5 == 0:
    # Red flag checks
    assert generic_ratio < 0.50, f"Generic ratio too high: {generic_ratio}"
    assert dominant_ratio < 0.50, f"Dominant ratio too high: {dominant_ratio}"
    assert bleu > -0.1, f"BLEU degrading: {bleu}"
    
    # Yellow flag checks
    if generic_ratio > 0.30:
        print(f"WARNING: Generic ratio elevated at epoch {epoch}")
    if dominant_ratio > 0.10:
        print(f"WARNING: Dominant ratio elevated at epoch {epoch}")
```

### Stage C → Stage D Promotion Gate

**Run Stage D only if all criteria met:**

```
✅ BLEU ≥ 1.0 on 50k training data
✅ ROUGE-L ≥ 10.0
✅ Generic ratio < 0.30
✅ Dominant ratio < 0.10
✅ Unique ratio > 0.40
✅ Sample predictions show grounded (non-generic) output
```

**Current (April 15, 2026 - Stage C epoch 20):**
- BLEU best: 1.39 ✅
- ROUGE-L best: 10.26 ✅
- Generic ratio: 0.1 ✅
- Dominant ratio: 0.0 ✅
- *Status: READY for potential promotion*

---

## 6. Test Set Protocol (Final Evaluation)

### Test Set (Held Out)

- **Source:** `artifacts/splits.json['test']`
- **Size:** ~10% of iSign v1.1 (~12.7k samples)
- **Usage:** DO NOT use during development. Run once after final model selection.

### Final Test Evaluation Procedure

```bash
# After Stage C (or Stage D if run) completes:
python evaluate_test_set.py \
    --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart/best_bleu_model.pt \
    --split test \
    --num-samples all \
    --output-dir artifacts/final_test_results/
```

**Output:** Metrics file with BLEU-4, ROUGE-L, per-sample predictions, and qualitative analysis.

**Reporting:** Final metrics reported here; do not iterate on test set.

---

## 7. Qualitative Analysis

### Sample Selection for Manual Review

Each epoch's history.json stores sample predictions:

```json
{
    "epoch": 20,
    "sample_preds_detailed": [
        {
            "uid": "_-M-OpYD5dk--0",
            "prediction": "he is a political leader and he is...",
            "reference": "A person talking about government policies",
            "generic_match": false,
            "bleu_contribution": 0.05
        },
        ...
    ]
}
```

### Manual Grounding Assessment

For best-checkpoint predictions, manually inspect ~50 samples:
1. **Input grounding:** Does prediction relate to the sign input?
2. **Grammar:** Is output well-formed English?
3. **Hallucination:** Is prediction a known high-frequency template?
4. **Paraphrasing:** If not exact match, does it preserve meaning?

### Qualitative Grading Rubric

| Grade | Criteria | Example |
|-------|----------|---------|
| ✅ **Excellent** | Prediction grounded in input, grammatical, correct or near-correct | ref: "government policies" → pred: "talking about politics" |
| ⚠️ **Partial** | Some content grounded, possible paraphrase, minor grammar issues | ref: "news report" → pred: "he is talking about the report" |
| ❌ **Poor** | No apparent grounding; generic; incoherent | ref: "sports discussion" → pred: "he said that he would not be able to this" |

---

## 8. Comparison Protocol: Between Runs

### Comparing Two Stage C Runs

```bash
python training/compare_stage_c_runs.py \
    --run1 artifacts/phase9_decoder_finetune/stageC_v2_bs8_baseline \
    --run2 artifacts/phase9_decoder_finetune/stageC_v2_bs8_safe_restart \
    --metric bleu
```

**Output:** Side-by-side epoch curves, best metrics, runtime analysis.

### Statistical Significance (Planned)

For future ablations, use bootstrap resampling to test if BLEU/ROUGE differences are significant:

```python
# Bootstrap N=1000 iterations
for i in range(1000):
    sample_idx = np.random.choice(len(predictions), len(predictions), replace=True)
    bleu_run1 = evaluate_bleu(predictions_run1[sample_idx],
                              references[sample_idx])
    bleu_run2 = evaluate_bleu(predictions_run2[sample_idx],
                              references[sample_idx])
    bootstrap_diffs.append(bleu_run1 - bleu_run2)

p_value = (bootstrap_diffs > 0).mean()
# p_value < 0.05 indicates significant difference
```

---

## 9. Failure Mode Detection

### Common Failure Patterns & How to Detect

| Failure Mode | Signal | Detection |
|---|---|---|
| **Encoder collapse** | BLEU→0, generic_ratio→0.9 | Monitor epoch 2-5 |
| **Decoder takeover** | generic_ratio stays > 0.5 despite low loss | Check sample_preds |
| **Mode collapse** | dominant_ratio → 0.3+ | Hourly check during first 10 epochs |
| **Overfitting** | val_loss increases, BLEU plateaus | Compare val metrics to train loss |
| **Gradient underflow** | Loss flat for 5+ epochs | Check learning rates, batch size |

**Action:** If any failure detected at epoch < 10, stop and diagnose hyperparameters before restarting.

---

## 10. Reproducibility Checklist

**Before running any stage:**

- [ ] Seed set globally: `set_global_seed(42)`
- [ ] Splits file exists: `artifacts/splits.json`
- [ ] Preprocessed cache exists: `artifacts/dataset_cache/isl_tf96.pt`
- [ ] WPP vocab exists (Stage C only): `artifacts/wpp_vocab.json`
- [ ] Previous checkpoint available: (Stage B/C/D only)

**During training:**

- [ ] Logs directory writable: `logs/`
- [ ] Artifacts directory writable: `artifacts/`
- [ ] GPU memory available: `nvidia-smi dmon`
- [ ] No other training jobs running on same device

**After training:**

- [ ] history.json saved: `artifacts/phase9_.../history.json`
- [ ] Checkpoints saved: `best_bleu_model.pt`, `best_loss_model.pt`
- [ ] Log file complete: `logs/stageC_*.log`

---

## References

- Metrics implementation: `utils/metrics.py`
- Validation loop: `training/train_stage_c_v2.py` (lines ~600–700)
- Splits definition: `artifacts/splits.json`
- Benchmark protocol: [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md)
