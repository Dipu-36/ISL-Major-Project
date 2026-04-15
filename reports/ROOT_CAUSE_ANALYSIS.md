# ISL Project — Root Cause Analysis Report

**Date:** April 8, 2026  
**Context:** Both Stage A v1 (InfoNCE + MLP projection) and Stage A v2 (InfoNCE + VICReg) failed semantically. This report explains why and what the evidence proves.

---

## 1. Confirmed Failure Mode

The Stage B content-word gate requires ≥10 of 30 content words to have cosine lift >1.15×. Both v1 and v2 scored **0/15** (out of a reduced 15-word probe). Average content lift: 0.994 (v1) and 0.961 (v2). At random chance the lift would be exactly 1.000, so the encoder has *less than random* content discrimination at the bridge (L2) level.

This failure is not a training instability issue. Both runs completed their full schedule, converged to reasonable InfoNCE loss values (3.17 and 3.63), and showed no numerical anomalies. The failure is structural.

---

## 2. Root Cause 1 — Sentence-Level InfoNCE Alignment Is Too Hard for This Data

### The problem

InfoNCE requires that each (pose_i, text_i) pair be a *positive pair*, and all (pose_i, text_j≠i) be *negative pairs*. The gradient signal is correct only if two conditions hold:

1. **Positives are genuinely more similar than negatives** at the feature level.
2. **Negatives are genuinely dissimilar** in content.

Neither condition holds reliably in iSign:

- **Alignment noise**: The iSign paper explicitly warns that sentence-level alignment has significant noise due to co-reference, role-shift, and non-manual variation. A pose clip may partially represent a different sentence than its annotated pair.
- **Paraphrase positives**: "He went to the hospital" and "He visited the doctor" have near-identical signing but different T5 sentence embeddings. The model is penalised for placing these near each other.
- **Surface-similar negatives**: "India won the match" and "India lost the match" share 3 of 4 content words and similar T5 embeddings but may have visually different signing (directional verbs). These are treated as hard negatives when they need to be near each other.

### The result

When the similarity matrix contains ~50% unreliable (pose, text) gradient directions, the encoder learns the only structure it *can* reliably distinguish: **sequence length and motion energy**. This explains why the PROBE_RESULTS.md shows:

- Sentence length lift: 1.59× ✓ (reliable)
- Person 1st/3rd lift: 1.11× ⚠ (marginal)
- Question vs statement: 1.00× ✗ (none)
- Content word lift: 1.00× uniformly ✗

**The encoder is correctly identifying what the InfoNCE loss rewards it for — surface temporal statistics — not content.**

---

## 3. Root Cause 2 — T5 Sentence Embeddings Are an Inappropriate Alignment Target

The text-side of the InfoNCE objective uses mean-pooled frozen T5 encoder embeddings.

### Why this is problematic

T5 sentence embeddings:
- Are trained on C4 (English web text), not on ISL-annotated data.
- Encode abstract semantic similarity at the paragraph level.
- Two semantically distant sentences can have similar T5 embeddings if they share clause structure ("he said that he...").
- The 512-dim T5 embedding space is a high-dimensional space calibrated for English NLU, not for aligning to pose sequences.

A 2-layer, d=256 visual encoder is thus asked to map noisy 96-frame pose sequences into a 512-dimensional space calibrated for English text. This is like asking a small student network to replicate the features of a teacher 20× larger, on out-of-distribution data, with noisy labels.

---

## 4. Root Cause 3 — Projection Head Behaviour

### v1 (MLP + BatchNorm)

The 2-layer MLP with BatchNorm in the projection head was absorbing all the alignment capacity. The ablation study confirmed:
- BatchNorm normalises away batch-level variance, creating a uniform distribution regardless of input.
- The BatchNorm allows the encoder to collapse (identical outputs) without incurring any loss penalty, because the MLP can trivially map the collapsed space to any output distribution.
- Evidence: L1 isotropy = 0.64, L2 isotropy = 0.75, content lift = 0.998 ≈ random.

### v2 (Linear head + VICReg)

V2 correctly fixed the collapse:
- L2 isotropy: 0.751 → 0.108 (collapse eliminated at bridge)
- L1 cosine mean: 0.410 → 0.203 (encoder spread out)

But VICReg applied at the bridge was **too aggressive**:
- L2 cosine mean dropped from 0.563 to 0.011 (near-orthogonal).
- In v1, cos=0.56 at the bridge encoded coarse lexical similarity (sentences are more similar to each other than random). V2 destroyed even this coarse structure.
- Content lift at L2 *degraded* from 0.993 to **0.957** (below random). VICReg pushed the bridge to produce maximally diverse but semantically *random* representations.

This confirms: forcing geometric diversity cannot manufacture semantic signal that was never learned. **Spreading the geometry is not the same as learning semantic content.**

---

## 5. Root Cause 4 — No Sub-Sentence Supervision Signal

The entire contrastive approach provides only one supervision signal per sample pair: whether two samples are the same sentence or not. For a sign language dataset:

- Each clip contains 5–25 individual signs.
- Each sign corresponds (roughly) to a word or morpheme.
- The sentence-level objective provides no gradient toward recognising any individual sign.
- Even if the encoder partially recognises "hospital" in a clip, that correct recognition produces zero additional loss reduction if the full sentence similarity to T5 is not improved.

The iSign benchmark explicitly includes **Word Presence Prediction (WPP)** as a representation eval task precisely because sentence-level translation supervision alone is insufficient.

---

## 5. Root Cause 5 — Encoder Capacity vs. Task Complexity

A 2-layer TransformerEncoder with d=256 has approximately:
- Input projection: 450 × 256 = 115K parameters
- 2 encoder layers: ~1.57M parameters
- **Total encoder: ~1.7M parameters**

This encoder must:
1. Denoise 96 frames × 450 dims of noisy pose data.
2. Model temporal dependencies between signs.
3. Produce a fixed-size representation that aligns to T5's 512-dim semantic space.

The iSign dataset has 127K samples but each is aligned at sentence level with significant noise. The effective learning signal per epoch is much lower than the raw sample count suggests.

A 2-layer encoder with 1.7M parameters is likely sufficient for WPP (word presence classification), as that is a simpler, *locally-grounded* task. But it may be insufficient for the sentence-level alignment task.

---

## 6. Evidence Summary Table

| Failure hypothesis | Evidence | Status |
|--------------------|----------|--------|
| Representation collapse | v1: L1 iso=0.64, L2=0.75; v2: VICReg fixed to 0.45/0.09 | **CONFIRMED (v1), FIXED (v2)** |
| No content signal despite geometric spread | v2: content lift at L1=0.995, L2=0.957 after collapse fix | **CONFIRMED** |
| Sentence-level InfoNCE unreliable on noisy data | Content lift ≈ random even after full training | **CONFIRMED** |
| T5 text embedding as alignment target too hard | no cross-modal retrieval signal even at best val loss | **CONFIRMED** |
| Projection head absorbs alignment capacity (MLP+BN) | Ablation H1 fixed collapse; H4 (VICReg) also fixed | **CONFIRMED** |
| No sub-sentence supervision | No per-sign or per-word intermediary | **CONFIRMED — design gap** |
| Encoder capacity insufficient for sentence NCE | 1.7M params, 2 layers; all 15 content words at lift ≈ 1.0 | **LIKELY CONTRIBUTING** |

---

## 7. Key Finding

**Both Stage A versions attacked a symptom (collapse) rather than the root cause (wrong objective).**

The VICReg + linear head in v2 successfully eliminated collapse, confirming the ablation study. But it exposed the actual root problem: the InfoNCE objective at sentence level provides essentially zero gradient toward lexical/semantic content learning on this dataset.

The fix must change the **learning objective**, not the training dynamics.

---

## 8. Data Ceiling Question (Open)

The STAGE_A_VERIFICATION.md correctly identifies a critical next step:

> "Probe L1 raw visual discriminability: Are sentences with the same keyword actually visually similar in raw pose space?"

This would establish the data ceiling — the maximum content signal achievable from these pose features alone. If sentences containing "hospital" look visually similar in raw pose space, the encoder *can* in principle learn WPP. If they do not (because ISL uses role-shift, non-manual markers, or regional signing variation to encode the same word differently), the problem is more fundamental.

The new WPP-based pretraining (Stage A v3) will simultaneously test this ceiling: if WPP training also fails, it strongly suggests the pose features do not encode consistent lexical information.
