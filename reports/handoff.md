# ISL Project Handoff

## Project Overview

This project is an Indian Sign Language to English translation system built from pose sequences rather than raw video. The current production training path uses a pose preprocessing pipeline, a temporal transformer encoder, a bridge into T5 embedding space, and a partially frozen T5 decoder.

High-level model path:

```text
.pose files
→ pose preprocessing
→ 456-dim frame features
→ temporal encoder
→ bridge projection to T5 space
→ T5 decoder with cross-attention
→ English text
```

Core dataset and assets:

- `iSign_v1.1.csv`: 127,237 text pairs
- `iSign-poses_v1.1/`: pose files
- `artifacts/dataset_cache/isl_tf128.pt`: cached preprocessed features

Key implementation files:

- `training/controlled_generalization_test.py`: main training loop and evaluation
- `model/t5_bridge.py`: bridge model wrapping temporal encoder + T5
- `model/temporal_encoder.py`: temporal transformer encoder
- `preprocessing/pipeline.py`: pose feature extraction and normalization
- `build_cache.py`: cache builder for preprocessed dataset

## Model Architecture

Current model:

- Input pose feature dim: 456
  - 228 pose coordinates
  - 228 velocity features
- Temporal encoder:
  - transformer encoder
  - hidden dim 256
  - originally 2 layers, later runs used 4 layers
- Bridge:
  - `Linear(256 → 512) + GELU + LayerNorm`
- Decoder:
  - T5-small
  - mostly frozen, top decoder layers selectively unfrozen in later phases

Important point: the decoder does have cross-attention to encoder outputs, so the failure is not “missing cross-attention.” The failure is that the encoder-side representation is not carrying usable lexical content.

## What Triggered This Investigation

The immediate question was why BLEU was still extremely low around Phase 8 epoch 75 even though validation loss kept improving.

The Phase 8 log showed:

- steadily improving `val_loss`
- repetitive hallucinated generations
- outputs with very weak relation to the input content
- repeated Indian-news-style template text such as:
  - `he said that he ...`
  - `... part of the bjp`
  - `... proud of his achievements`

This suggested the decoder was generating fluent prior-driven text instead of grounded translations.

## What Was Done

### 1. Phase 8 analysis

We inspected the active Phase 8 log and confirmed:

- qualitative outputs were off-topic
- BLEU was low because decoded text did not overlap with references in content
- loss improvement did not correspond to grounded generation

### 2. Historical tracing across prior phases

We checked earlier logs and traced the hallucination backward:

- Phase 8: hallucination persists
- Phase 7: already present
- Phase 6: already present by epoch 32
- earliest `ce_extended` run: already present from epoch 1

This is the critical conclusion:

**The project does not have a Phase 8-specific bug. The system never learned source-grounded translation in any phase.**

### 3. Training was stopped

The running Phase 8 process was terminated because it was continuing to improve loss without changing the qualitative failure mode.

### 4. Data-domain sanity check

We checked whether the training data itself was strongly biased toward Indian political/news language.

Result:

- only a small fraction of the corpus is political/news style
- the dominant hallucinated decoder output does **not** reflect the real data distribution

This means the bad outputs are coming from decoder prior behavior under weak conditioning, not from the dataset being mostly BJP/news text.

## Diagnostic Artifacts Created

These files were created during the investigation:

- `DIAGNOSTIC_REPORT.md`
  - long-form diagnosis of the hallucination history and failure mode
- `probe_bridge_variance.py`
  - measures variance and similarity of encoder bridge outputs
- `probe_content_signal.py`
  - checks whether encoder features contain usable text/content signal
- `PROBE_RESULTS.md`
  - results from the encoder variance and content probes

## Confirmed Findings

### Finding 1: Hallucination predates Phase 8

The same family of decoder outputs appears throughout the training history. Phase 8 is only a continuation of a much older failure.

### Finding 2: The encoder is not fully collapsed geometrically

Bridge variance probe results on Phase 8 best checkpoint:

- mean pairwise cosine similarity: `0.6316`
- cosine to global mean: `0.7910`
- 90% of variance captured by 19 principal directions

Interpretation:

- encoder outputs are **not** almost identical vectors
- there is real geometric variation
- but that variation may still be useless for translation

### Finding 3: Encoder features carry almost no lexical content

This was the decisive test.

Content probe results:

- sentence length prediction: strong signal
- weak signal for coarse stylistic/person categories
- no useful signal for question-vs-statement
- **no measurable lift for content-word prediction**

Interpretation:

- the encoder is learning surface-level statistics such as duration/energy/length
- it is **not** learning semantic or lexical content needed for grounded translation

### Finding 4: Decoder failure is downstream of weak encoder supervision

The decoder likely learned early that encoder cross-attention was not informative enough, then defaulted to its pretrained language model prior. Once this equilibrium formed, the encoder no longer received strong learning pressure to become content-discriminative.

This is the current best explanation:

1. encoder starts weak/random
2. decoder falls back to fluent prior text
3. decoder sends poor grounding gradients backward
4. encoder learns only cheap signals like sequence length
5. the whole system stabilizes in a hallucination attractor

## Current State of the Project

### Status

- active Phase 8 training has been stopped
- no further training should be resumed from the current setup without changing strategy

### Best current interpretation

This is primarily an **encoder-content-learning failure**, not just a decoder scheduling issue.

More precisely:

- not a Phase 8-only problem
- not mainly caused by newly added encoder layers
- not caused by missing cross-attention wiring
- not explained by dataset topical bias
- not explained by complete geometric collapse of encoder outputs

Instead:

- encoder features vary, but mostly encode non-semantic information
- decoder never got a sufficiently content-discriminative source signal

## Recommended Next Direction

### Main path: fix encoder learning before decoder training

The project should follow the “Path 9C” conclusion from the probe results.

Recommended sequence:

1. Pretrain the temporal encoder + bridge without the decoder.
2. Use an auxiliary objective that forces pose-text alignment.
3. Re-run the content probe after pretraining.
4. Only when content signal is measurable should the decoder be trained again.

### Best immediate strategy

Recommended encoder pretraining task:

- pose-text contrastive alignment

Concrete idea:

- pose side: temporal encoder + bridge → pooled embedding
- text side: frozen T5 encoder → pooled embedding
- train with InfoNCE / contrastive alignment so matching pose-text pairs are close and mismatches are far

Why this is the best next move:

- it directly teaches the encoder to represent text-level content
- it avoids relying on a hallucinating decoder during encoder learning
- it reuses the paired dataset already available

### Only after that

After the encoder passes the content probe:

1. Freeze encoder temporarily.
2. Train decoder on top of the pretrained encoder.
3. Check qualitative outputs early.
4. Then consider joint fine-tuning.

## What Another AI Agent Should Read First

Recommended reading order for a new agent:

1. `handoff.md`
2. `DIAGNOSTIC_REPORT.md`
3. `PROBE_RESULTS.md`
4. `training/controlled_generalization_test.py`
5. `model/t5_bridge.py`
6. `model/temporal_encoder.py`
7. `preprocessing/pipeline.py`

## What Another AI Agent Should Do Next

The next agent should not resume Phase 8 training.

Instead, it should:

1. design an encoder-only pretraining script for pose-text contrastive alignment
2. reuse the existing dataset cache and splits
3. save encoder checkpoints separately from seq2seq checkpoints
4. rerun `probe_content_signal.py` after pretraining
5. only proceed to decoder training if lexical/content lift becomes meaningfully above baseline

## Key Takeaways in One Paragraph

This ISL translation project uses pose features, a temporal encoder, a bridge into T5 space, and a mostly frozen T5 decoder. Investigation started from low BLEU in Phase 8, but historical log tracing proved the failure existed from the earliest training phase. The decoder hallucination pattern is old and persistent. The decisive probes show the encoder output is not fully collapsed geometrically, but it carries almost no lexical content signal. It mostly represents superficial properties like sentence length and motion statistics. The next step is not more Phase 8 training or decoder tweaking first; it is encoder pretraining with a direct pose-text alignment objective, followed by re-probing for content signal before attempting seq2seq fine-tuning again.