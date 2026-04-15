# ISL Project — Redesign Summary: Why the New Pipeline Should Outperform

**Date:** April 8, 2026

---

## The Core Argument

Every previous approach in this project failed at the same point: the encoder produces features that are geometrically diverse (after v2 fixing collapse) but semantically random. The new pipeline changes the fundamental training objective in a way that directly addresses this failure mode.

---

## Why InfoNCE Failed (The v1/v2 Problem)

Sentence-level InfoNCE between pose embeddings and T5 sentence embeddings is fundamentally mismatched with this data:

1. **Label noise**: iSign has documented sentence-level alignment noise. NCE treats noisy pairs as true positives and clean near-synonyms as hard negatives — the opposite of an informative gradient.

2. **Target complexity mismatch**: T5 sentence embeddings encode abstract, compositional English semantics at a scale (60M parameters, 24 transformer blocks) that a 1.7M-parameter visual encoder cannot plausibly align to in a single shot.

3. **No intermediate signal**: The loss provides one gradient direction per sample pair, from the full sentence down to all 96 frames simultaneously. There is no mechanism to individually reinforce the frames where specific signs occur.

---

## Why WPP Should Work

Word Presence Prediction directly addresses each failure point:

1. **Tractable target**: Binary classification "does word W appear in this clip?" is learnable by a shallow classifier on mean-pooled features, per the PROBE_RESULTS.md data showing that sequence length (a much weaker signal) is already detectable. Word presence is a harder but reachable target.

2. **Multi-label structure**: With 150 binary labels per sample, the loss provides 150 independent gradient signals per training step — each reinforcing sign-specific patterns in the pose features.

3. **Alignment noise tolerance**: Even if the sentence-pose alignment is noisy for some clips, word presence labels are more robust — a word is present or absent, and partial noise affects only some labels, not the entire training signal.

4. **Direct correspondence to ISL structure**: In sign languages, individual signs correspond approximately to words or morphemes. WPP supervision directly rewards the encoder for detecting sign-level content.

---

## Why the Multi-task Stage C Should Prevent Hallucination

Prior Phases 6–8 suffered "decoder language model attractor": the encoder provided no discriminative signal, so T5's pre-trained prior dominated.

Stage C v2 disrupts this by:
1. Starting from a WPP-trained encoder that already carries vocabulary signal.
2. Adding WPP as a continuously applied auxiliary loss during translation training (aux_weight: 0.5 → 0.05).
3. The auxiliary loss penalises any epoch where the encoder loses its vocabulary prediction ability, preventing the encoder from drifting to uninformative representations under cross-entropy pressure.

---

## Quantitative Expectations

| Stage | Prior result | New target | Basis for expectation |
|-------|-------------|------------|----------------------|
| Stage A avg AUC | 0.50 (random) | ≥ 0.58 | WPP is a tractable binary classification task; sequence length was detectable at AUC ≈ 0.72 with a much weaker signal |
| Stage B gate | 0/15 FAIL | ≥25/150 PASS | Gate is set conservatively; WPP provides direct word-level supervision |
| Stage C BLEU | 0.00–0.80 (artefact) | ≥ 1.0 genuine | Only requires the encoder to provide enough signal to distinguish some inputs |
| Stage C generic_ratio | 0.80–0.90 | < 0.30 | WPP auxiliary prevents full decoder takeover |

---

## Comparison to Prior Approach

| Factor | v1/v2 (InfoNCE) | v3 (WPP) |
|--------|----------------|---------|
| Supervision signal per sample | 1 scalar (InfoNCE loss) | 150 binary labels |
| Gradient quality | Unreliable on noisy pairs | Robust to partial noise |
| Target complexity | T5 sentence semantics (~60M params) | Binary word presence |
| Architectural alignment | Encoder must map to deep NLU space | Encoder must map to shallow binary labels |
| Collapse prevention | VICReg (needed to prevent all gradient collapse) | VICReg (soft, preserves coarse structure) + label diversity |

---

## What This Pipeline Does NOT Claim

1. It does not claim to solve the full ISL translation problem — that requires much larger data, better pose quality, and likely pre-trained sign features.
2. It does not claim to beat the iSign benchmark BLEU in one training run — the benchmark likely requires the full 103k training set run through multiple stages.
3. It does not claim the 4-layer encoder is optimal — it is an informed incremental improvement over the 2-layer architecture that empirically failed to extract lexical features.

---

## What Success Looks Like

The pipeline can be considered to have meaningfully improved over prior attempts if:

1. Stage B v2 gate **passes** (never done before in any prior version).
2. Stage C sample predictions show vocabulary variation — different inputs lead to different content words in outputs, even if the outputs are not yet fluent or accurate.
3. The `generic_ratio` (predicting "he said that he...") drops below 0.30, indicating the decoder is no longer in a fixed attractor.

Together, these would mean the encoder is genuinely meaning-aware for the first time in this project, establishing a foundation for further improvement through larger data, better preprocessing, and architecture scaling.
