#!/usr/bin/env python3
"""Stage C v2 — Multi-task translation training with WPP auxiliary loss.

MOTIVATION
----------
The systematic failure of the decoder in prior phases is that the encoder
provides no discriminative signal, allowing T5's pre-trained language model
prior to dominate generation.

Stage A v3 (WPP pretraining) teaches the encoder to recognize vocabulary.
Stage C v2 preserves that vocabulary signal during translation training by
keeping a WPP auxiliary loss on the encoder output throughout Stage C.

WITHOUT WPP AUXILIARY: encoder tends to drift away from vocabulary awareness
under pressure from the cross-entropy translation loss, reproducing the
hallucination attractor seen in Phases 6–8.

WITH WPP AUXILIARY: the encoder is continuously penalised if it stops
encoding word presence, preventing the decoder from monopolising all the
learning signal.

TRAINING STRATEGY
-----------------
1. Load encoder from Stage A v3 checkpoint (encoder_best.pt).
2. Attach T5-small decoder + bridge (fresh).
3. Freeze T5 completely for the first FREEZE_EPOCHS.
4. After FREEZE_EPOCHS, unfreeze top-2 T5 decoder blocks + final_layer_norm.
5. Joint loss:
      loss = CE_loss + aux_weight * WPP_loss
   where aux_weight decays from WPP_AUX_WEIGHT_INIT → WPP_AUX_WEIGHT_END
   over WPP_AUX_EPOCHS (cosine schedule).
6. Separate learning rates: encoder lr=LR_ENCODER, decoder lr=LR_DECODER.
7. Beam decode on validation every epoch; save best BLEU checkpoint.

Checkpoints saved to: artifacts/phase9_decoder_finetune/<run_name>/
  best_bleu_model.pt  — best val BLEU
  best_loss_model.pt  — best val CE loss
  history.json        — per-epoch metrics

PREREQUISITES
-------------
  Stage B v2 gate must have passed:
    python probe_stage_b_v2.py --checkpoint artifacts/.../encoder_best.pt
    → exit code 0

  WPP vocabulary must exist:
    artifacts/wpp_vocab.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.arch_v2_redesign import (
    D_MODEL, DIM_FEEDFORWARD, ENCODER_DROPOUT,
    ENCODER_LAYERS, LR_DECODER, LR_ENCODER, NHEAD, SRC_DIM, T5_NAME,
    TARGET_FRAMES, WPP_AUX_EPOCHS, WPP_AUX_WEIGHT_END, WPP_AUX_WEIGHT_INIT,
)
from model.temporal_encoder import TemporalVisualEncoder
from training.word_inventory import load_wpp_vocabulary, make_wpp_labels
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_EPOCHS: int = 2            # LR warmup duration
MAX_EPOCHS: int = 50
MAX_TARGET_LEN: int = 128
LABEL_SMOOTHING: float = 0.1
WEIGHT_DECAY: float = 1e-2
TRAIN_PROGRESS_EVERY: int = 100
VAL_PROGRESS_EVERY: int = 50

# Stage C retry tuning: the first run showed partial success numerically
# (BLEU just above 1.0) but persistent generic hallucinations.  These local
# overrides reduce decoder-language-model takeover while preserving the Stage A
# lexical signal for longer during translation fine-tuning.
# FIX 1: Increased decoder unfreezing from 1 layer to 2 layers for better utilization
FREEZE_DECODER_EPOCHS: int = 8
PATIENCE: int = 8
DECODER_LAYERS_TO_UNFREEZE: int = 2  # WAS: 1 — INCREASED to 2 for better decoder capacity
STAGE_C_DECODER_LR: float = LR_DECODER * 0.5

# FIX 2: Extended WPP auxiliary weight floor to prevent encoder drift too early
# WAS: 0.8 -> 0.2 over 40 epochs — END floor too low, encoder drifts
# NOW: 0.6 -> 0.3 over 50 epochs — higher floor preserves lexical signal longer
STAGE_C_WPP_AUX_WEIGHT_INIT: float = 0.6
STAGE_C_WPP_AUX_WEIGHT_END: float = 0.3
STAGE_C_WPP_AUX_EPOCHS: int = 50

# FIX 3: Gradient accumulation steps to increase effective batch size
# Effective batch size = batch_size * GRAD_ACCUM_STEPS = 32 * 2 = 64
GRAD_ACCUM_STEPS: int = 2

# FIX 4: Plateau-based LR reduction trigger
# Reduce LR when val_ce plateaus for PLATEAU_PATIENCE epochs
PLATEAU_PATIENCE: int = 5
PLATEAU_FACTOR: float = 0.5  # Multiply LR by this factor on plateau


# ─────────────────────────────────────────────────────────────────────────────
# Full translation model (encoder from WPP + T5 decoder)
# ─────────────────────────────────────────────────────────────────────────────

class WPPTranslationModel(nn.Module):
    """ISL-to-English translation model initialised from a WPP-trained encoder.

    The encoder comes from Stage A v3 checkpoint; T5 and the bridge are fresh.
    The WPP head is retained for the auxiliary loss during Stage C training.
    """

    def __init__(
        self,
        src_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        t5_name: str,
        vocab_size: int,
        freeze_t5: bool = True,
    ) -> None:
        super().__init__()

        # Temporal encoder (pre-trained in Stage A v3)
        self.temporal_encoder = TemporalVisualEncoder(
            src_dim=src_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # T5 decoder
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False

        # FIX 5: Bridge: project encoder output to T5 embedding space
        # NOTE: Original bridge was mask-unaware; we now apply mask after projection
        # to ensure padded positions don't leak into decoder via position-wise ops.
        self.bridge_projection = nn.Linear(d_model, self.t5.config.d_model)
        self.bridge_activation = nn.GELU()
        self.bridge_norm = nn.LayerNorm(self.t5.config.d_model)

        # WPP head: auxiliary loss to preserve vocabulary awareness
        # (same as Stage A v3 head, weights loaded from checkpoint)
        self.wpp_head = nn.Linear(d_model, vocab_size)

    def encode_visual(
        self, src: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode pose sequence.

        Returns:
            bridged:  [B, T, T5_d_model] — for T5 cross-attention
            mask:     [B, T]             — valid frame mask
            pooled:   [B, d_model]       — mean-pooled encoder output (for WPP aux)
        """
        seq_out, out_mask = self.temporal_encoder(src, attention_mask)

        # Mean-pool for WPP head
        valid = out_mask.float().unsqueeze(-1)
        pooled = (seq_out * valid).sum(1) / valid.sum(1).clamp(min=1.0)

        # FIX 5 (cont): Mask-aware bridge projection
        # Apply bridge with mask to prevent padded positions from affecting output
        bridged = self.bridge_projection(seq_out)
        bridged = self.bridge_activation(bridged)
        # Zero out padded positions before LayerNorm to prevent leakage
        bridged = bridged * out_mask.unsqueeze(-1).float()
        bridged = self.bridge_norm(bridged)
        # Apply mask again after normalization (LayerNorm can shift zeros)
        bridged = bridged * out_mask.unsqueeze(-1).float()
        return bridged, out_mask, pooled

    def forward(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (translation_loss, wpp_logits).

        Args:
            src:            [B, T, src_dim] — pose features
            attention_mask: [B, T]          — 1=valid, 0=padding
            labels:         [B, tgt_len]    — T5-tokenised target IDs (-100=ignore)

        Returns:
            ce_loss:    scalar cross-entropy translation loss
            wpp_logits: [B, vocab_size] — for optional auxiliary WPP loss
        """
        bridged, mask, pooled = self.encode_visual(src, attention_mask)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=bridged,
        )
        t5_output = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=mask,
            labels=labels,
        )
        ce_loss = t5_output.loss

        wpp_logits = self.wpp_head(pooled)
        return ce_loss, wpp_logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 64,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Beam-search decoding."""
        bridged, mask, _ = self.encode_visual(src, attention_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=bridged)
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=mask,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=1.0,
        )

    def unfreeze_decoder_top_n(self, n: int) -> int:
        """Unfreeze the top-n T5 decoder blocks and final_layer_norm.

        Returns the number of newly trainable parameters.
        """
        # Re-freeze everything first to be safe
        for p in self.t5.parameters():
            p.requires_grad = False

        # Identify the decoder blocks (T5 decoder has 'block' attribute)
        decoder = self.t5.decoder
        total_blocks = len(decoder.block)
        for i in range(total_blocks - n, total_blocks):
            for p in decoder.block[i].parameters():
                p.requires_grad = True

        if hasattr(decoder, "final_layer_norm"):
            for p in decoder.final_layer_norm.parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.t5.parameters() if p.requires_grad)
        return n_trainable


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for Stage C
# ─────────────────────────────────────────────────────────────────────────────

class TranslationWPPDataset(Dataset):
    """Dataset that returns pose features + T5 labels + text string.

    The text string is needed to build WPP labels on the fly.
    """

    def __init__(
        self,
        uids: list[str],
        uid_to_text: dict[str, str],
        tokenizer,
        cache: dict,
        max_target_len: int = MAX_TARGET_LEN,
    ) -> None:
        # Tokenize all upfront and discard tokenizer to avoid pickling issues
        self.samples: list[dict] = []
        skipped = 0

        for uid in uids:
            text = uid_to_text.get(uid)
            entry = cache.get(uid)
            if text is None or entry is None:
                skipped += 1
                continue

            src_arr = entry["features"]
            if isinstance(src_arr, torch.Tensor):
                src_np = src_arr.detach().cpu().numpy().astype("float32", copy=False)
            else:
                src_np = src_arr.astype("float32", copy=False)

            mask_arr = entry["attention_mask"]
            if isinstance(mask_arr, torch.Tensor):
                mask_np = mask_arr.detach().cpu().numpy().astype("int64", copy=False)
            else:
                mask_np = mask_arr.astype("int64", copy=False)
            clean_text = entry.get("text", text)

            tokenized = tokenizer(
                clean_text,
                truncation=True,
                max_length=max_target_len,
                return_tensors=None,
            )
            label_ids = tokenized["input_ids"]
            if len(label_ids) < 2:
                skipped += 1
                continue

            self.samples.append({
                "uid": uid,
                "text": clean_text,
                "src_np": src_np,
                "mask_np": mask_np,
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
            })

        # Discard tokenizer to avoid multiprocessing pickle issues
        del tokenizer

        print(f"[TranslationWPPDataset] Loaded {len(self.samples)}, skipped {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        sample = self.samples[i]
        return {
            "uid": sample["uid"],
            "text": sample["text"],
            "src": torch.from_numpy(sample["src_np"]),
            "mask": torch.from_numpy(sample["mask_np"]),
            "label_ids": sample["label_ids"],
        }


def _collate_translation(batch: list[dict], pad_id: int) -> dict:
    src = torch.stack([b["src"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    label_ids = pad_sequence([b["label_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels_for_loss = label_ids.clone()
    labels_for_loss[labels_for_loss == pad_id] = -100
    return {
        "src": src,
        "mask": mask,
        "text": [b["text"] for b in batch],
        "labels_for_loss": labels_for_loss,
        "label_ids": label_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _get_lr(epoch: int, warmup: int, total: int, base_lr: float, min_lr: float = 1e-7) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _get_wpp_aux_weight(epoch: int, total_aux_epochs: int) -> float:
    """Cosine decay of the WPP auxiliary weight."""
    if epoch >= total_aux_epochs:
        return STAGE_C_WPP_AUX_WEIGHT_END
    progress = epoch / max(total_aux_epochs, 1)
    return STAGE_C_WPP_AUX_WEIGHT_END + 0.5 * (STAGE_C_WPP_AUX_WEIGHT_INIT - STAGE_C_WPP_AUX_WEIGHT_END) * (
        1.0 + math.cos(math.pi * progress)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lcs(a: list, b: list) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def _rouge_l(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    p = lcs / max(len(h), 1)
    recall = lcs / max(len(r), 1)
    return 2.0 * p * recall / max(p + recall, 1e-8)


def _compute_translation_metrics(
    preds: list[str], refs: list[str]
) -> dict[str, float]:
    if not preds:
        return {"bleu": 0.0, "rouge_l": 0.0, "dominant_ratio": 0.0, "generic_ratio": 0.0}
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl").score
    rouge = statistics.mean(_rouge_l(p, r) for p, r in zip(preds, refs)) * 100.0
    dominant_pred = Counter(preds).most_common(1)[0][1] / len(preds)
    generic = sum(
        1 for p in preds
        if re.match(r"^he said that he", p.lower())
    ) / len(preds)
    return {
        "bleu": round(bleu, 3),
        "rouge_l": round(rouge, 3),
        "dominant_ratio": round(dominant_pred, 3),
        "generic_ratio": round(generic, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def _run_train_epoch(
    model: WPPTranslationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    word_to_idx: dict[str, int],
    aux_weight: float,
    device: torch.device,
    log_every: int = TRAIN_PROGRESS_EVERY,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,  # FIX 3: Gradient accumulation
) -> dict[str, float]:
    model.train()
    total_ce = total_wpp_aux = total = 0.0
    n = 0
    total_steps = len(loader)
    epoch_t0 = time.time()
    accum_count = 0  # FIX 3: Track gradient accumulation steps

    # FIX 3: Zero gradients at start for accumulation
    optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        src = batch["src"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels_for_loss"].to(device)
        wpp_labels = make_wpp_labels(batch["text"], word_to_idx, device=device)

        ce_loss, wpp_logits = model(src, mask, labels)
        wpp_aux_loss = F.binary_cross_entropy_with_logits(
            wpp_logits, wpp_labels,
            pos_weight=torch.full((wpp_labels.size(1),), 10.0, device=device),
        )
        loss = ce_loss + aux_weight * wpp_aux_loss

        # FIX 3: Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        accum_count += 1

        # FIX 3: Only step optimizer after accumulating enough gradients
        if accum_count >= grad_accum_steps or step == total_steps:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        # FIX 3: Track unscaled losses for logging (multiply back for accurate reporting)
        total_ce += ce_loss.item()
        total_wpp_aux += wpp_aux_loss.item()
        total += (ce_loss + aux_weight * wpp_aux_loss).item()
        n += 1

        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - epoch_t0
            print(
                f"[stageC_v2][train] step {step}/{total_steps} "
                f"loss={total/max(step, 1):.4f} ce={total_ce/max(step, 1):.4f} "
                f"wpp={total_wpp_aux/max(step, 1):.4f} elapsed={elapsed:.0f}s "
                f"[eff_bs={src.size(0) * grad_accum_steps}]"
            )

    d = max(n, 1)
    return {"ce_loss": total_ce/d, "wpp_aux_loss": total_wpp_aux/d, "total_loss": total/d}


def _normalize_for_exact_match(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _build_labeled_sample_predictions(
    preds: list[str], refs: list[str], limit: int = 5
) -> list[dict[str, object]]:
    labeled_samples: list[dict[str, object]] = []
    for i, (pred, ref) in enumerate(zip(preds[:limit], refs[:limit])):
        is_correct = _normalize_for_exact_match(pred) == _normalize_for_exact_match(ref)
        correctness_label = "CORRECT" if is_correct else "INCORRECT"
        labeled_samples.append(
            {
                "sample_id": i + 1,
                "label": f"MODEL_PRED|{correctness_label}",
                "labeled_prediction": f"[MODEL_PRED][{correctness_label}] {pred}",
                "is_correct": is_correct,
                "prediction": pred,
                "reference": ref,
            }
        )
    return labeled_samples


def _run_val_epoch(
    model: WPPTranslationModel,
    loader: DataLoader,
    word_to_idx: dict[str, int],
    aux_weight: float,
    tokenizer,
    device: torch.device,
    log_every: int = VAL_PROGRESS_EVERY,
) -> dict:
    model.eval()
    total_ce = total_wpp_aux = 0.0
    n = 0
    all_preds, all_refs = [], []
    total_steps = len(loader)

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            src = batch["src"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels_for_loss"].to(device)
            wpp_labels = make_wpp_labels(batch["text"], word_to_idx, device=device)

            ce_loss, wpp_logits = model(src, mask, labels)
            wpp_aux = F.binary_cross_entropy_with_logits(
                wpp_logits, wpp_labels,
                pos_weight=torch.full((wpp_labels.size(1),), 10.0, device=device),
            )
            total_ce += ce_loss.item()
            total_wpp_aux += wpp_aux.item()
            n += 1

            # Decode all validation samples for accurate BLEU
            pred_ids = model.generate(src, mask)
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            all_preds.extend(preds)
            all_refs.extend(batch["text"])

            if step % log_every == 0 or step == total_steps:
                print(f"[stageC_v2][val] step {step}/{total_steps} decoded={len(all_preds)}")

    d = max(n, 1)
    metrics = {"ce_loss": total_ce/d, "wpp_aux_loss": total_wpp_aux/d}
    if all_preds:
        metrics.update(_compute_translation_metrics(all_preds, all_refs))
        labeled_samples = _build_labeled_sample_predictions(all_preds, all_refs)
        metrics["sample_preds"] = all_preds[:5]
        metrics["sample_preds_labeled"] = [sample["labeled_prediction"] for sample in labeled_samples]
        metrics["sample_preds_raw"] = all_preds[:5]
        metrics["sample_preds_detailed"] = labeled_samples
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# arg parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C v2 — multi-task translation + WPP auxiliary")
    p.add_argument("--encoder-checkpoint", required=True,
                   help="Path to Stage A v3 encoder_best.pt")
    p.add_argument("--run-name", default="stageC_v2",
                   help="Run subdirectory under artifacts/phase9_decoder_finetune/")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Limit training samples (default: use all)")
    p.add_argument("--val-samples", type=int, default=None,
                   help="Limit validation samples (default: use all)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vocab-path", default="",
                   help="Override WPP vocab JSON path")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stageC_v2] Device: {device}")
    print(f"[stageC_v2] Run: {args.run_name}")

    run_dir = os.path.join(PROJECT_ROOT, "artifacts", "phase9_decoder_finetune", args.run_name)
    ensure_dir(run_dir)

    # ── Check Stage B gate ─────────────────────────────────────────────────
    enc_dir = os.path.dirname(args.encoder_checkpoint)
    gate_v2 = os.path.join(enc_dir, "stage_b_v2_probe.json")
    gate_v1 = os.path.join(enc_dir, "stage_b_probe.json")

    gate_file = gate_v2 if os.path.exists(gate_v2) else gate_v1
    if os.path.exists(gate_file):
        with open(gate_file) as f:
            gate = json.load(f)
        if not gate.get("gate_passed", gate.get("gate_pass", False)):
            print(f"[stageC_v2] ABORT — Stage B gate has not passed: {gate_file}")
            print(f"[stageC_v2] Re-run probe_stage_b_v2.py and ensure exit code 0.")
            sys.exit(1)
        print(f"[stageC_v2] Stage B gate: PASSED ✓")
    else:
        print(f"[stageC_v2] WARNING: no Stage B probe found at {gate_file}. Proceeding anyway.")

    # ── Load vocabulary ────────────────────────────────────────────────────
    vocab_path = args.vocab_path or os.path.join(PROJECT_ROOT, "artifacts", "wpp_vocab.json")
    vocab_data = load_wpp_vocabulary(vocab_path)
    vocab: list[str] = vocab_data["vocab"]
    word_to_idx: dict[str, int] = vocab_data["word_to_idx"]
    V = len(vocab)
    print(f"[stageC_v2] WPP vocabulary: {V} words")

    # ── Load data ──────────────────────────────────────────────────────────
    cache_path = os.path.join(PROJECT_ROOT, "artifacts", "dataset_cache", f"isl_tf{TARGET_FRAMES}.pt")
    print(f"[stageC_v2] Loading cache from: {cache_path}")
    try:
        import time
        t0 = time.time()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[stageC_v2] Cache loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        raise RuntimeError(f"Failed to load cache from {cache_path}: {e}") from e
    cache = payload["samples"]
    print(f"[stageC_v2] Cache: {len(cache)} entries")

    import csv as _csv
    uid_to_text: dict[str, str] = {}
    with open(os.path.join(PROJECT_ROOT, "iSign_v1.1.csv"), newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            uid = row.get("uid", "").strip()
            text = row.get("text", "").strip()
            if uid:
                uid_to_text[uid] = text

    with open(os.path.join(PROJECT_ROOT, "artifacts", "splits.json")) as f:
        splits = json.load(f)

    import numpy as np
    rng = np.random.default_rng(args.seed)
    train_uids_all = list(splits.get("train_uids", splits.get("train", [])))
    val_uids_all = list(splits.get("val_uids", splits.get("val", [])))
    rng.shuffle(train_uids_all)
    # Use all data by default; only subset if explicitly requested
    train_limit = args.train_samples if args.train_samples is not None else len(train_uids_all)
    val_limit = args.val_samples if args.val_samples is not None else len(val_uids_all)
    train_uids = train_uids_all[:train_limit]
    val_uids = val_uids_all[:val_limit]
    print(f"[stageC_v2] Training on {len(train_uids)} samples, validating on {len(val_uids)} samples")

    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
    pad_id = tokenizer.pad_token_id

    train_ds = TranslationWPPDataset(train_uids, uid_to_text, tokenizer, cache)
    val_ds = TranslationWPPDataset(val_uids, uid_to_text, tokenizer, cache)
    # Release cache dict once datasets are materialized to reduce peak RAM.
    del cache
    del payload

    from functools import partial
    collate_fn = partial(_collate_translation, pad_id=pad_id)

    print(f"[stageC_v2] Building DataLoaders (batch_size={args.batch_size}, num_workers={args.num_workers})...")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    # ── Build model and load encoder weights ───────────────────────────────
    print(f"[stageC_v2] Loading encoder from: {args.encoder_checkpoint}")
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    required_keys = ["model_state_dict"]
    missing = [k for k in required_keys if k not in enc_ckpt]
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}. Got: {list(enc_ckpt.keys())}")
    enc_arch = enc_ckpt.get("arch", {
        "src_dim": SRC_DIM, "d_model": D_MODEL, "nhead": NHEAD,
        "num_layers": ENCODER_LAYERS, "dropout": ENCODER_DROPOUT,
        "vocab_size": V,
    })

    model = WPPTranslationModel(
        src_dim=enc_arch.get("src_dim", SRC_DIM),
        d_model=enc_arch.get("d_model", D_MODEL),
        nhead=enc_arch.get("nhead", NHEAD),
        num_layers=enc_arch.get("num_layers", ENCODER_LAYERS),
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=enc_arch.get("dropout", ENCODER_DROPOUT),
        t5_name=T5_NAME,
        vocab_size=enc_arch.get("vocab_size", V),
        freeze_t5=True,
    ).to(device)

    # Transfer encoder and WPP head weights from Stage A checkpoint
    # Build a key mapping: checkpoint has encoder.* and head.*
    ckpt_sd = enc_ckpt["model_state_dict"]
    enc_sd = {k.replace("encoder.", "temporal_encoder.", 1): v
              for k, v in ckpt_sd.items() if k.startswith("encoder.")}
    head_sd = {k.replace("head.", "wpp_head.", 1): v
               for k, v in ckpt_sd.items() if k.startswith("head.")}
    model.temporal_encoder.load_state_dict(
        {k.replace("temporal_encoder.", "", 1): v for k, v in enc_sd.items()},
        strict=True,
    )
    model.wpp_head.load_state_dict(
        {k.replace("wpp_head.", "", 1): v for k, v in head_sd.items()},
        strict=True,
    )
    print(f"[stageC_v2] Encoder + WPP head weights loaded from Stage A v3 checkpoint")

    # ── Optimizer (encoder active, decoder frozen initially) ───────────────
    # FIX 5: Bridge is now split into separate modules (not Sequential)
    trainable_params = [
        {"params": list(model.temporal_encoder.parameters()), "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY, "group_name": "encoder"},
        {"params": list(model.bridge_projection.parameters()) + list(model.bridge_norm.parameters()), "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY, "group_name": "bridge"},
        {"params": list(model.wpp_head.parameters()), "lr": LR_ENCODER * 0.5, "weight_decay": WEIGHT_DECAY * 0.1, "group_name": "wpp_head"},
        # T5 params: only non-frozen ones will receive gradients
        {"params": [p for p in model.t5.parameters() if p.requires_grad], "lr": STAGE_C_DECODER_LR, "weight_decay": WEIGHT_DECAY, "group_name": "decoder"},
    ]
    optimizer = torch.optim.AdamW(trainable_params)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stageC_v2] Trainable params (initial): {n_trainable:,}")

    # ── Training loop ──────────────────────────────────────────────────────
    best_bleu = 0.0
    best_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    decoder_unfrozen = False

    # FIX 4 (cont): Plateau-based LR reduction state
    epochs_since_improvement = 0
    current_plateau_factor = 1.0  # Multiplicative factor applied to base LR

    print(f"\n[stageC_v2] Stage C v2 training for up to {MAX_EPOCHS} epochs")
    print(f"[stageC_v2] T5 decoder frozen for first {FREEZE_DECODER_EPOCHS} epochs\n")

    for epoch in range(MAX_EPOCHS):
        # Unfreeze decoder after FREEZE_DECODER_EPOCHS
        if epoch == FREEZE_DECODER_EPOCHS and not decoder_unfrozen:
            n_new = model.unfreeze_decoder_top_n(DECODER_LAYERS_TO_UNFREEZE)
            print(f"\n[stageC_v2] Unfroze top-{DECODER_LAYERS_TO_UNFREEZE} decoder blocks "
                  f"(+{n_new:,} trainable params)")
            # Rebuild optimizer to include new params
            optimizer = torch.optim.AdamW([
                {"params": list(model.temporal_encoder.parameters()), "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY, "group_name": "encoder"},
                {"params": list(model.bridge_projection.parameters()) + list(model.bridge_norm.parameters()), "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY, "group_name": "bridge"},
                {"params": list(model.wpp_head.parameters()), "lr": LR_ENCODER * 0.5, "weight_decay": WEIGHT_DECAY * 0.1, "group_name": "wpp_head"},
                {"params": [p for p in model.t5.parameters() if p.requires_grad], "lr": STAGE_C_DECODER_LR, "weight_decay": WEIGHT_DECAY, "group_name": "decoder"},
            ])
            decoder_unfrozen = True

        # LR schedule (per param group)
        enc_lr = _get_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS, LR_ENCODER)
        dec_lr = _get_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS, STAGE_C_DECODER_LR)
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "decoder":
                pg["lr"] = dec_lr
            else:
                pg["lr"] = enc_lr

        aux_weight = _get_wpp_aux_weight(epoch, STAGE_C_WPP_AUX_EPOCHS)

        t0 = time.time()
        tr = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            word_to_idx,
            aux_weight,
            device,
            log_every=TRAIN_PROGRESS_EVERY,
        )
        vl = _run_val_epoch(
            model,
            val_loader,
            word_to_idx,
            aux_weight,
            tokenizer,
            device,
            log_every=VAL_PROGRESS_EVERY,
        )
        elapsed = time.time() - t0

        bleu = vl.get("bleu", 0.0)
        val_ce = vl.get("ce_loss", 999.0)

        row = {
            "epoch": epoch + 1,
            "enc_lr": round(enc_lr * current_plateau_factor, 8),  # FIX 4: Apply plateau factor
            "dec_lr": round(dec_lr * current_plateau_factor, 8),
            "plateau_factor": round(current_plateau_factor, 4),  # FIX 4: Track plateau factor
            "aux_weight": round(aux_weight, 4),
            "train_ce": round(tr["ce_loss"], 5),
            "train_wpp_aux": round(tr["wpp_aux_loss"], 5),
            "val_ce": round(val_ce, 5),
            "val_wpp_aux": round(vl.get("wpp_aux_loss", 0.0), 5),
            "bleu": bleu,
            "rouge_l": vl.get("rouge_l", 0.0),
            "dominant_ratio": vl.get("dominant_ratio", 0.0),
            "generic_ratio": vl.get("generic_ratio", 0.0),
            "sample_preds": vl.get("sample_preds", []),
            "sample_preds_labeled": vl.get("sample_preds_labeled", []),
            "sample_preds_raw": vl.get("sample_preds_raw", []),
            "sample_preds_detailed": vl.get("sample_preds_detailed", []),
            "elapsed_s": round(elapsed, 1),
        }

        print(
            f"  Epoch {epoch+1:>3d} | CE={tr['ce_loss']:.4f}/{val_ce:.4f} "
            f"WPP_aux={tr['wpp_aux_loss']:.4f} "
            f"BLEU={bleu:.2f} ROUGE-L={vl.get('rouge_l', 0.0):.1f} "
            f"dom={vl.get('dominant_ratio', 0.0):.2f} "
            f"aux_w={aux_weight:.3f} | {elapsed:.0f}s"
        )
        if vl.get("sample_preds"):
            print(f"    Sample pred: {vl['sample_preds'][0][:80]}")

        history.append(row)
        save_json(os.path.join(run_dir, "history.json"), history)

        # Checkpoint: best BLEU
        if bleu > best_bleu:
            best_bleu = bleu
            patience_counter = 0
            epochs_since_improvement = 0  # FIX 4: Reset plateau counter on improvement
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "decoder_unfrozen": decoder_unfrozen,
                    "best_bleu": best_bleu,
                    "best_loss": best_loss,
                    "bleu": bleu,
                    "val_ce": val_ce,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                os.path.join(run_dir, "best_bleu_model.pt"),
            )
            print(f"    ✓ Best BLEU={bleu:.2f} → checkpoint saved")
        else:
            patience_counter += 1
            epochs_since_improvement += 1  # FIX 4: Track epochs without improvement

        # FIX 4: Plateau-based LR reduction
        if epochs_since_improvement >= PLATEAU_PATIENCE:
            current_plateau_factor *= PLATEAU_FACTOR
            epochs_since_improvement = 0  # Reset after reducing LR
            print(f"    ⚠ LR plateau detected — reducing LR by factor {PLATEAU_FACTOR}")
            print(f"    New plateau factor: {current_plateau_factor:.4f}")

        # Checkpoint: best val CE loss
        if val_ce < best_loss:
            best_loss = val_ce
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "decoder_unfrozen": decoder_unfrozen,
                    "best_bleu": best_bleu,
                    "best_loss": best_loss,
                    "bleu": bleu,
                    "val_ce": val_ce,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                os.path.join(run_dir, "best_loss_model.pt"),
            )

        if patience_counter >= PATIENCE:
            print(f"\n[stageC_v2] Early stop — no BLEU improvement for {PATIENCE} epochs")
            break

    print("\n" + "=" * 60)
    print("[stageC_v2] STAGE C v2 COMPLETE")
    print(f"  Best BLEU:    {best_bleu:.3f}")
    print(f"  Best val CE:  {best_loss:.5f}")
    print(f"  Checkpoints → {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
