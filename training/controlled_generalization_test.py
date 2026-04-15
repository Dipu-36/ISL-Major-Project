#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import sacrebleu
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers.modeling_outputs import BaseModelOutput

from model.t5_bridge import T5BridgeModel
from preprocessing.pipeline import PreprocessingConfig, preprocess_single_sample
from training.semantic_loss import ContrastivePairLoss, SemanticGroundingLoss
from utils.io import ensure_dir, load_json, load_uid_to_text, save_json
from utils.seed import set_global_seed


@dataclass(frozen=True)
class DecodeConfig:
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    train_loss: float
    val_loss: float
    bleu: float
    rouge_l: float
    chrf: float
    input_difference_ratio: float
    unique_prediction_ratio: float
    dominant_prediction_ratio: float
    repetitive_output_ratio: float
    generic_sentence_ratio: float
    avg_output_length: float
    avg_target_length: float
    output_to_target_length_ratio: float
    avg_grad_norm: float
    max_grad_norm: float
    ce_loss: float = 0.0
    semantic_loss: float = 0.0
    semantic_lambda: float = 0.0
    contrastive_loss: float = 0.0
    contrastive_weight: float = 0.0
    effective_lambda: float = 0.0


class ControlledT5Dataset(Dataset):
    def __init__(
        self,
        uids: list[str],
        uid_to_text: dict[str, str],
        pose_dir: str,
        tokenizer: Any,
        preprocessing_cfg: PreprocessingConfig,
        max_target_len: int,
        cache: dict[str, Any] | None = None,
    ):
        self.samples: list[dict[str, Any]] = []

        for uid in uids:
            text = uid_to_text.get(uid)
            if text is None:
                continue

            if cache is not None:
                # Fast path: look up pre-processed features from cache
                entry = cache.get(uid)
                if entry is None:
                    continue
                clean = entry["text"]
                # Cache stores float16 to save space; cast back to float32 for computation
                src_np = entry["features"].astype("float32")
                mask_np = entry["attention_mask"].astype("int64")
            else:
                # Slow path: read and preprocess .pose file on-the-fly
                pose_path = os.path.join(pose_dir, uid + ".pose")
                if not os.path.exists(pose_path):
                    continue
                try:
                    processed = preprocess_single_sample(
                        uid=uid,
                        raw_text=text,
                        pose_path=pose_path,
                        cfg=preprocessing_cfg,
                    )
                except Exception:
                    continue
                clean = processed.text
                src_np = processed.features
                mask_np = processed.attention_mask

            tokenized = tokenizer(
                clean,
                truncation=True,
                max_length=max_target_len,
                return_tensors=None,
            )
            label_ids = tokenized["input_ids"]
            if len(label_ids) < 2:
                continue

            self.samples.append(
                {
                    "uid": uid,
                    "text": clean,
                    "src": torch.tensor(src_np, dtype=torch.float32),
                    "attention_mask": torch.tensor(mask_np, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def collate_t5(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    src = torch.stack([item["src"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=pad_id,
    )

    labels_for_loss = labels.clone()
    labels_for_loss[labels_for_loss == pad_id] = -100

    return {
        "uid": [item["uid"] for item in batch],
        "text": [item["text"] for item in batch],
        "src": src,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_for_loss": labels_for_loss,
    }


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(hyp: str, ref: str) -> float:
    h = hyp.split()
    r = ref.split()
    if not h or not r:
        return 0.0
    lcs = lcs_length(h, r)
    precision = lcs / max(len(h), 1)
    recall = lcs / max(len(r), 1)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _extract_char_ngrams(text: str, n: int) -> Counter[str]:
    padded = " " + text + " "
    grams = [padded[i : i + n] for i in range(max(0, len(padded) - n + 1))]
    return Counter(grams)


def chr_fscore(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    hyp = hyp.strip()
    ref = ref.strip()
    if not hyp or not ref:
        return 0.0

    scores = []
    for n in range(1, n_max + 1):
        h_counts = _extract_char_ngrams(hyp, n)
        r_counts = _extract_char_ngrams(ref, n)
        overlap = sum(min(h_counts[g], r_counts[g]) for g in h_counts)

        h_total = sum(h_counts.values())
        r_total = sum(r_counts.values())
        if h_total == 0 or r_total == 0:
            scores.append(0.0)
            continue

        precision = overlap / h_total
        recall = overlap / r_total
        if precision + recall == 0:
            scores.append(0.0)
            continue

        f = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        scores.append(f)

    return 100.0 * float(sum(scores) / len(scores))


def sentence_bleu_approx(hyp: str, ref: str, max_n: int = 4) -> float:
    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        if not hyp_ngrams:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        precisions.append(max(overlap / sum(hyp_ngrams.values()), 1e-9))

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - (len(ref_tokens) / max(len(hyp_tokens), 1)))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * bp * geo_mean


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"bleu": 0.0, "rouge_l": 0.0, "chrf": 0.0}

    # Corpus-level SacreBLEU — directly comparable to published papers
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score

    # ROUGE-L: sentence-level F1 averaged (verified identical to google rouge-score)
    rouge_scores = [100.0 * rouge_l_f1(h, r) for h, r in zip(predictions, references)]

    return {
        "bleu": float(bleu),
        "rouge_l": float(statistics.mean(rouge_scores)),
        "chrf": float(chrf),
    }


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - (uniq / len(grams))


def is_generic_sentence(text: str) -> bool:
    cleaned = text.strip().lower()
    tokens = cleaned.split()

    if len(tokens) <= 2:
        return True

    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    unique_alpha = len(set(alpha_tokens))

    return unique_alpha <= 2


def detect_failure_patterns(predictions: list[str]) -> dict[str, float]:
    if not predictions:
        return {
            "unique_prediction_ratio": 0.0,
            "dominant_prediction_ratio": 1.0,
            "repetitive_output_ratio": 1.0,
            "generic_sentence_ratio": 1.0,
        }

    norm = [p.strip().lower() for p in predictions]
    counts = Counter(norm)

    dominant = counts.most_common(1)[0][1] / len(norm)
    unique_ratio = len(counts) / len(norm)

    repetitive_flags = [repeated_ngram_ratio(p) > 0.35 for p in norm]
    repetitive_ratio = float(sum(repetitive_flags) / len(repetitive_flags))

    generic_flags = [is_generic_sentence(p) for p in norm]
    generic_ratio = float(sum(generic_flags) / len(generic_flags))

    return {
        "unique_prediction_ratio": float(unique_ratio),
        "dominant_prediction_ratio": float(dominant),
        "repetitive_output_ratio": float(repetitive_ratio),
        "generic_sentence_ratio": float(generic_ratio),
    }


def average_token_lengths(predictions: list[str], references: list[str]) -> tuple[float, float, float]:
    if not predictions or not references:
        return 0.0, 0.0, 0.0

    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]

    avg_pred = float(statistics.mean(pred_lens))
    avg_ref = float(statistics.mean(ref_lens))
    ratio = float(avg_pred / max(avg_ref, 1e-6))

    return avg_pred, avg_ref, ratio


def get_warmup_lambda(epoch: int, target_lambda: float) -> float:
    """Return the effective semantic λ for the given epoch using a linear warmup.

    Schedule (generalised from the target value):
      epochs 1–2 : λ = 0.0            (CE only — let model stabilise first)
      epochs 3–4 : λ = 0.5 * target   (gentle introduction)
      epochs 5+  : λ = target          (full signal)
    """
    if epoch <= 2:
        return 0.0
    elif epoch <= 4:
        return target_lambda * 0.5
    else:
        return target_lambda


def categorize_predictions(
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """Classify each prediction as correct / partial / generic / incorrect / empty.

    Thresholds:
      empty     — blank prediction
      generic   — is_generic_sentence() returns True
      correct   — ROUGE-L F1 ≥ 50%
      partial   — ROUGE-L F1 ≥ 10%
      incorrect — everything else
    """
    categories: list[str] = []
    for pred, ref in zip(predictions, references):
        p = pred.strip()
        if not p:
            cat = "empty"
        elif is_generic_sentence(p):
            cat = "generic"
        else:
            rl = 100.0 * rouge_l_f1(p, ref)
            if rl >= 50.0:
                cat = "correct"
            elif rl >= 10.0:
                cat = "partial"
            else:
                cat = "incorrect"
        categories.append(cat)

    counts = Counter(categories)
    n = max(len(categories), 1)
    return {
        "correct_ratio":   float(counts.get("correct",   0) / n),
        "partial_ratio":   float(counts.get("partial",   0) / n),
        "generic_ratio":   float(counts.get("generic",   0) / n),
        "incorrect_ratio": float(counts.get("incorrect", 0) / n),
        "empty_ratio":     float(counts.get("empty",     0) / n),
        "per_sample": [
            {"prediction": p, "reference": r, "category": c}
            for p, r, c in zip(predictions[:10], references[:10], categories[:10])
        ],
    }


def compute_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    has_grad = False
    for param in model.parameters():
        if param.grad is None:
            continue
        has_grad = True
        norm = float(torch.norm(param.grad.detach()).item())
        total_sq += norm * norm

    if not has_grad:
        return 0.0
    return math.sqrt(max(total_sq, 0.0))


def run_epoch(
    model: T5BridgeModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.CrossEntropyLoss,
    semantic_loss_fn: SemanticGroundingLoss | None = None,
    contrastive_loss_fn: ContrastivePairLoss | None = None,
    effective_lambda: float = 0.0,
    contrastive_weight: float = 0.0,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    use_amp = scaler is not None and device.type == "cuda"
    use_semantic = is_train and semantic_loss_fn is not None and effective_lambda > 0.0
    use_contrastive = is_train and contrastive_loss_fn is not None and contrastive_weight > 0.0
    # Share one encode_visual call across CE + semantic + contrastive when possible
    use_shared_encoder = use_semantic or use_contrastive
    model.train(mode=is_train)

    total_loss = 0.0
    total_ce_loss = 0.0
    total_sem_loss = 0.0
    total_con_loss = 0.0
    total_batches = 0
    grad_norms: list[float] = []

    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels_for_loss = batch["labels_for_loss"].to(device, non_blocking=True)
        labels_clean = batch["labels"].to(device, non_blocking=True)

        if not torch.isfinite(src).all():
            raise RuntimeError("NaN/Inf detected in input features.")

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_shared_encoder:
                # Single visual-encoder forward pass; tensor is reused by T5 and aux losses.
                encoder_hidden, reduced_mask = model.encode_visual(src, attention_mask)
                outputs = model.t5(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
                    attention_mask=reduced_mask,
                    labels=labels_for_loss,
                )
            else:
                encoder_hidden = None
                reduced_mask = None
                # T5 performs token shifting internally when labels are provided.
                outputs = model(src=src, attention_mask=attention_mask, labels=labels_for_loss)

            logits = outputs.logits

            if logits.shape[:2] != labels_for_loss.shape:
                raise RuntimeError(
                    f"Logit/label sequence misalignment: logits={tuple(logits.shape)}, labels={tuple(labels_for_loss.shape)}"
                )

            ce_loss = criterion(logits.reshape(-1, logits.shape[-1]), labels_for_loss.reshape(-1))

            if not torch.isfinite(ce_loss):
                raise RuntimeError("NaN/Inf detected in CE loss.")

            loss = ce_loss

            if use_semantic:
                sem_loss = semantic_loss_fn(
                    logits=logits,
                    labels_for_loss=labels_for_loss,
                    labels_clean=labels_clean,
                )
                if not torch.isfinite(sem_loss):
                    raise RuntimeError("NaN/Inf detected in semantic loss.")
                loss = loss + effective_lambda * sem_loss
                total_sem_loss += float(sem_loss.item())

            if use_contrastive:
                text_mask = labels_for_loss.ne(-100)
                con_loss = contrastive_loss_fn(
                    visual_hidden=encoder_hidden,
                    visual_mask=reduced_mask,
                    labels_clean=labels_clean,
                    text_mask=text_mask,
                )
                if not torch.isfinite(con_loss):
                    raise RuntimeError("NaN/Inf detected in contrastive loss.")
                loss = loss + contrastive_weight * con_loss
                total_con_loss += float(con_loss.item())

        total_ce_loss += float(ce_loss.item())

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                if not math.isfinite(grad_norm):
                    # AMP overflow: scaler will skip this step — expected behaviour
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss += float(loss.item())
                    total_batches += 1
                    continue
                grad_norms.append(float(grad_norm))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = compute_grad_norm(model)
                if not math.isfinite(grad_norm):
                    raise RuntimeError("Non-finite gradient norm detected.")
                grad_norms.append(float(grad_norm))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if total_batches % 100 == 0:
            phase = "train" if is_train else "val"
            print(
                f"  [{phase} batch {total_batches}/{len(loader)}]"
                f" loss={total_loss/total_batches:.4f}"
                + (f" ce={total_ce_loss/total_batches:.4f}" if is_train else ""),
                flush=True,
            )

    n = max(total_batches, 1)
    avg_grad = float(statistics.mean(grad_norms)) if grad_norms else 0.0
    max_grad = float(max(grad_norms)) if grad_norms else 0.0

    return {
        "loss": total_loss / n,
        "ce_loss": total_ce_loss / n,
        "semantic_loss": total_sem_loss / n,
        "contrastive_loss": total_con_loss / n,
        "avg_grad_norm": avg_grad,
        "max_grad_norm": max_grad,
    }


@torch.no_grad()
def decode_predictions(
    model: T5BridgeModel,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    decode_cfg: DecodeConfig,
    max_batches: int | None = None,
) -> tuple[list[str], list[str]]:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src = batch["src"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            generated = model.generate(
                src=src,
                attention_mask=attention_mask,
                max_length=48,
                num_beams=decode_cfg.num_beams,
                length_penalty=decode_cfg.length_penalty,
                no_repeat_ngram_size=decode_cfg.no_repeat_ngram_size,
                repetition_penalty=decode_cfg.repetition_penalty,
            )
        pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)

        predictions.extend([p.strip() for p in pred_text])
        references.extend([t.strip() for t in batch["text"]])

        if (batch_idx + 1) % 50 == 0:
            total = len(loader) if max_batches is None else min(max_batches, len(loader))
            print(
                f"  [decode batch {batch_idx + 1}/{total}] {len(predictions)} decoded so far",
                flush=True,
            )

    return predictions, references


@torch.no_grad()
def input_dependence_test(
    model: T5BridgeModel,
    probe_batch: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    decode_cfg: DecodeConfig,
) -> dict[str, Any]:
    model.eval()

    src = probe_batch["src"].to(device)
    attention_mask = probe_batch["attention_mask"].to(device)

    real_ids = model.generate(
        src=src,
        attention_mask=attention_mask,
        max_length=48,
        num_beams=decode_cfg.num_beams,
        length_penalty=decode_cfg.length_penalty,
        no_repeat_ngram_size=decode_cfg.no_repeat_ngram_size,
        repetition_penalty=decode_cfg.repetition_penalty,
    )
    zero_ids = model.generate(
        src=torch.zeros_like(src),
        attention_mask=attention_mask,
        max_length=48,
        num_beams=decode_cfg.num_beams,
        length_penalty=decode_cfg.length_penalty,
        no_repeat_ngram_size=decode_cfg.no_repeat_ngram_size,
        repetition_penalty=decode_cfg.repetition_penalty,
    )

    real_text = [x.strip() for x in tokenizer.batch_decode(real_ids, skip_special_tokens=True)]
    zero_text = [x.strip() for x in tokenizer.batch_decode(zero_ids, skip_special_tokens=True)]

    changed = sum(1 for a, b in zip(real_text, zero_text) if a != b)
    diff_ratio = changed / max(len(real_text), 1)

    return {
        "difference_ratio": float(diff_ratio),
        "real_examples": real_text[:5],
        "zero_examples": zero_text[:5],
    }


def build_dataloaders(
    root: str,
    train_samples: int,
    val_samples: int,
    batch_size: int,
    max_target_len: int,
    seed: int,
    target_frames: int = 128,
    num_workers: int = 4,
    cache_dir: str | None = None,
) -> tuple[DataLoader, DataLoader, Any, int, int]:
    splits = load_json(os.path.join(root, "artifacts", "splits.json"))
    uid_to_text = load_uid_to_text(os.path.join(root, "iSign_v1.1.csv"))
    pose_dir = os.path.join(root, "iSign-poses_v1.1")

    train_uids = list(splits["train_uids"])[:train_samples]
    val_uids = list(splits["val_uids"])[:val_samples]

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    prep_cfg = PreprocessingConfig(target_frames=target_frames)

    # ── Optional cache ────────────────────────────────────────────────────────
    cache: dict[str, Any] | None = None
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f"isl_tf{target_frames}.pt")
        if os.path.exists(cache_path):
            import torch as _torch
            payload = _torch.load(cache_path, map_location="cpu", weights_only=False)
            cache = payload["samples"]
            meta = payload.get("meta", {})
            print(
                f"[cache] Loaded {meta.get('n_samples', len(cache))} entries "
                f"from {cache_path}  (tf={meta.get('target_frames')})"
            )
        else:
            print(
                f"[cache] Cache not found at {cache_path} — "
                f"falling back to live preprocessing.  "
                f"Run build_cache.py to generate it."
            )

    train_uids = list(splits["train_uids"])[:train_samples]
    val_uids = list(splits["val_uids"])[:val_samples]

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    prep_cfg = PreprocessingConfig(target_frames=target_frames)

    train_ds = ControlledT5Dataset(
        uids=train_uids,
        uid_to_text=uid_to_text,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        preprocessing_cfg=prep_cfg,
        max_target_len=max_target_len,
        cache=cache,
    )
    val_ds = ControlledT5Dataset(
        uids=val_uids,
        uid_to_text=uid_to_text,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        preprocessing_cfg=prep_cfg,
        max_target_len=max_target_len,
        cache=cache,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    # num_workers > 0 triggers persistent_workers for faster epoch transitions
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=True,
        generator=generator,
        collate_fn=lambda b: collate_t5(b, pad_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=True,
        collate_fn=lambda b: collate_t5(b, pad_id=tokenizer.pad_token_id),
    )

    return train_loader, val_loader, tokenizer, len(train_ds), len(val_ds)


def save_plots(history: list[EpochSummary], out_dir: str, run_name: str) -> None:
    ensure_dir(out_dir)
    epochs = [h.epoch for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h.train_loss for h in history], marker="o", label="train_loss")
    plt.plot(epochs, [h.val_loss for h in history], marker="o", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss Curves - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_loss_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h.bleu for h in history], marker="o", label="BLEU")
    plt.plot(epochs, [h.rouge_l for h in history], marker="o", label="ROUGE-L")
    plt.plot(epochs, [h.chrf for h in history], marker="o", label="CHRF")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title(f"Metric Progression - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_metric_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h.avg_output_length for h in history], marker="o", label="avg_output_len")
    plt.plot(epochs, [h.avg_target_length for h in history], marker="o", label="avg_target_len")
    plt.xlabel("epoch")
    plt.ylabel("tokens")
    plt.title(f"Output Length Tracking - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_length_curves.png"), dpi=150)
    plt.close()

    if any(h.semantic_lambda > 0 for h in history):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h.ce_loss for h in history], marker="o", label="CE loss (train)")
        plt.plot(epochs, [h.semantic_loss for h in history], marker="o", label="semantic loss (train)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"CE vs Semantic Loss - {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{run_name}_semantic_vs_ce_loss.png"), dpi=150)
        plt.close()


def run_single_experiment(
    *,
    root: str,
    output_dir: str,
    run_name: str,
    train_samples: int,
    val_samples: int,
    batch_size: int,
    epochs: int,
    lr: float,
    max_target_len: int,
    metric_eval_batches: int,
    seed: int,
    decode_cfg: DecodeConfig,
    semantic_lambda: float = 0.0,
    contrastive_weight: float = 0.0,
    use_warmup: bool = True,
    checkpoint_path: str | None = None,
    unfreeze_decoder_layers: int = 0,
    decoder_lr: float = 2e-5,
    encoder_layers: int = 2,
    encoder_dropout: float = 0.0,
    target_frames: int = 128,
    cache_dir: str | None = None,
    decode_every_n: int = 1,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create run_dir early so checkpoints can be saved during training
    run_dir = ensure_dir(os.path.join(output_dir, run_name))

    train_loader, val_loader, tokenizer, train_size, val_size = build_dataloaders(
        root=root,
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=batch_size,
        max_target_len=max_target_len,
        seed=seed,
        target_frames=target_frames,
        cache_dir=cache_dir,
    )

    model = T5BridgeModel(
        src_dim=450,
        freeze_t5=True,
        encoder_layers=encoder_layers,
        encoder_dropout=encoder_dropout,
    ).to(device)

    # ── Selective decoder unfreezing ──────────────────────────────
    if unfreeze_decoder_layers > 0:
        unfrozen = model.unfreeze_decoder_top_n(unfreeze_decoder_layers)
        print(f"[{run_name}] Unfroze {len(unfrozen)} params in top {unfreeze_decoder_layers} decoder layer(s)")

    # ── Optimizer with separate param groups ──────────────────────
    decoder_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("t5.decoder."):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict[str, Any]] = [{"params": other_params, "lr": lr}]
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr})
        print(f"[{run_name}] Optimizer: bridge/encoder lr={lr}, decoder lr={decoder_lr}")

    optimizer = torch.optim.AdamW(param_groups)
    # eta_min = 5 % of the base LR prevents the schedule from collapsing to
    # near-zero and maintains meaningful gradient signal in later epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)

    # ── Checkpoint loading ──────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    best_rouge_l = 0.0
    loaded_history: list[dict[str, Any]] = []
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # strict=False allows loading checkpoints that predate architecture additions
        # (e.g. pos_enc.pe buffer added in Phase 7). Missing keys are filled from
        # the freshly-initialised model; unexpected keys in the checkpoint are ignored.
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if incompatible.missing_keys:
            print(f"[{run_name}] Checkpoint missing keys (filled from init): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[{run_name}] Checkpoint unexpected keys (ignored): {incompatible.unexpected_keys}")
        # Skip optimizer/scheduler restore when trainable params changed (decoder unfreezing)
        if unfreeze_decoder_layers == 0:
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print(f"[{run_name}] Fresh optimizer/scheduler (decoder layers newly unfrozen)")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_rouge_l = ckpt.get("best_rouge_l", 0.0)
        loaded_history = ckpt.get("history", [])
        print(
            f"[{run_name}] Loaded checkpoint from {checkpoint_path} "
            f"(epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4f}, best_rouge_l={best_rouge_l:.2f})"
        )

    semantic_loss_fn: SemanticGroundingLoss | None = None
    if semantic_lambda > 0.0:
        semantic_loss_fn = SemanticGroundingLoss(t5_model=model.t5).to(device)

    contrastive_loss_fn: ContrastivePairLoss | None = None
    if contrastive_weight > 0.0:
        contrastive_loss_fn = ContrastivePairLoss(
            t5_model=model.t5,
            d_model=model.t5.config.d_model,
        ).to(device)

    probe_batch = next(iter(val_loader))

    # ── Mixed precision ───────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print(f"[{run_name}] AMP enabled (float16 autocast + GradScaler)")

    history: list[EpochSummary] = []
    sample_predictions: list[dict[str, Any]] = []
    last_error_analysis: dict[str, Any] = {}
    if best_val_loss == float("inf"):
        best_val_loss = float("inf")

    last_epoch = start_epoch + epochs - 1

    for epoch in range(start_epoch, start_epoch + epochs):
        # Apply warmup to semantic lambda (contrastive weight is fixed)
        effective_lambda = (
            get_warmup_lambda(epoch, semantic_lambda) if use_warmup else semantic_lambda
        )

        train_stats = run_epoch(
            model, train_loader, device, optimizer, criterion,
            semantic_loss_fn=semantic_loss_fn,
            contrastive_loss_fn=contrastive_loss_fn,
            effective_lambda=effective_lambda,
            contrastive_weight=contrastive_weight,
            scaler=scaler,
        )
        val_stats = run_epoch(model, val_loader, device, optimizer=None, criterion=criterion)

        # ── Decide whether to run expensive decode this epoch ─────
        is_last_epoch = epoch == last_epoch
        is_decode_epoch = (
            is_last_epoch
            or decode_every_n <= 1
            or (epoch - start_epoch) % decode_every_n == 0
        )

        if is_decode_epoch:
            predictions, references = decode_predictions(
                model,
                val_loader,
                tokenizer,
                device,
                decode_cfg,
                max_batches=None,  # evaluate on full val set for reliable metrics
            )

            metrics = compute_metrics(predictions, references)
            failure = detect_failure_patterns(predictions)
            dep = input_dependence_test(model, probe_batch, tokenizer, device, decode_cfg)
            avg_out_len, avg_ref_len, len_ratio = average_token_lengths(predictions, references)
        else:
            # Lightweight epoch: carry forward zeros for generation metrics
            predictions, references = [], []
            metrics = {"bleu": 0.0, "rouge_l": 0.0, "chrf": 0.0}
            failure = {
                "unique_prediction_ratio": 0.0,
                "dominant_prediction_ratio": 0.0,
                "repetitive_output_ratio": 0.0,
                "generic_sentence_ratio": 0.0,
            }
            dep = {"difference_ratio": 0.0, "real_examples": [], "zero_examples": []}
            avg_out_len, avg_ref_len, len_ratio = 0.0, 0.0, 0.0

        summary = EpochSummary(
            epoch=epoch,
            train_loss=float(train_stats["loss"]),
            val_loss=float(val_stats["loss"]),
            bleu=metrics["bleu"],
            rouge_l=metrics["rouge_l"],
            chrf=metrics["chrf"],
            input_difference_ratio=dep["difference_ratio"],
            unique_prediction_ratio=failure["unique_prediction_ratio"],
            dominant_prediction_ratio=failure["dominant_prediction_ratio"],
            repetitive_output_ratio=failure["repetitive_output_ratio"],
            generic_sentence_ratio=failure["generic_sentence_ratio"],
            avg_output_length=avg_out_len,
            avg_target_length=avg_ref_len,
            output_to_target_length_ratio=len_ratio,
            avg_grad_norm=float(train_stats["avg_grad_norm"]),
            max_grad_norm=float(train_stats["max_grad_norm"]),
            ce_loss=float(train_stats["ce_loss"]),
            semantic_loss=float(train_stats["semantic_loss"]),
            semantic_lambda=float(effective_lambda),
            contrastive_loss=float(train_stats["contrastive_loss"]),
            contrastive_weight=float(contrastive_weight),
            effective_lambda=float(effective_lambda),
        )
        history.append(summary)

        # ── Checkpoint saving ─────────────────────────────────────
        val_loss_val = float(val_stats["loss"])
        rouge_l_val = metrics["rouge_l"]
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss_val),
            "best_rouge_l": max(best_rouge_l, rouge_l_val),
            "history": [asdict(s) for s in history],
        }
        torch.save(ckpt_payload, os.path.join(run_dir, "last_model.pt"))
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            torch.save(ckpt_payload, os.path.join(run_dir, "best_model.pt"))
            print(f"  ↳ New best val_loss={best_val_loss:.4f} — saved best_model.pt")
        if is_decode_epoch and rouge_l_val > best_rouge_l:
            best_rouge_l = rouge_l_val
            torch.save(ckpt_payload, os.path.join(run_dir, "best_rouge_l_model.pt"))
            print(f"  ↳ New best ROUGE-L={best_rouge_l:.2f} — saved best_rouge_l_model.pt")

        scheduler.step()

        # Run error analysis on the final epoch only (avoids per-epoch overhead)
        if is_last_epoch and predictions:
            last_error_analysis = categorize_predictions(predictions, references)

        # ── Logging ───────────────────────────────────────────────
        sem_info = (
            f" ce={train_stats['ce_loss']:.4f} sem={train_stats['semantic_loss']:.4f}"
            f" λ_eff={effective_lambda:.3f}"
            + (f" con={train_stats['contrastive_loss']:.4f}" if contrastive_weight > 0.0 else "")
            if (semantic_lambda > 0.0 or contrastive_weight > 0.0)
            else ""
        )

        if is_decode_epoch:
            n_log = min(10, len(predictions))
            sample_predictions.append(
                {
                    "epoch": epoch,
                    "examples": [
                        {
                            "reference": references[i] if i < len(references) else "",
                            "prediction": predictions[i] if i < len(predictions) else "",
                            "is_generic": is_generic_sentence(predictions[i]) if i < len(predictions) else True,
                        }
                        for i in range(n_log)
                    ],
                    "input_dependence_examples": {
                        "real": dep["real_examples"],
                        "zero": dep["zero_examples"],
                    },
                }
            )

            print(f"  Predictions (epoch {epoch}, {n_log} samples):")
            for i in range(n_log):
                ref = references[i] if i < len(references) else ""
                pred = predictions[i] if i < len(predictions) else ""
                marker = " [G]" if is_generic_sentence(pred) else ""
                print(f"    [{i}] ref:  {ref}")
                print(f"         pred: {pred}{marker}")

            print(
                f"[{run_name}] epoch={epoch} train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f}"
                f"{sem_info} "
                f"BLEU={metrics['bleu']:.2f} ROUGE-L={metrics['rouge_l']:.2f} CHRF={metrics['chrf']:.2f} "
                f"input_diff={dep['difference_ratio']:.2f} unique={failure['unique_prediction_ratio']:.2f} "
                f"dominant={failure['dominant_prediction_ratio']:.2f} generic={failure['generic_sentence_ratio']:.2f} "
                f"len_ratio={len_ratio:.2f} grad_avg={train_stats['avg_grad_norm']:.3f} grad_max={train_stats['max_grad_norm']:.3f}"
            )
        else:
            print(
                f"[{run_name}] epoch={epoch} train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f}"
                f"{sem_info} "
                f"(decode skipped — next decode at epoch {epoch + decode_every_n - (epoch - start_epoch) % decode_every_n})"
                f" grad_avg={train_stats['avg_grad_norm']:.3f} grad_max={train_stats['max_grad_norm']:.3f}",
                flush=True,
            )

    save_plots(history, run_dir, run_name)

    report = {
        "run_name": run_name,
        "seed": seed,
        "device": str(device),
        "train_size": train_size,
        "val_size": val_size,
        "epochs": epochs,
        "criterion": {
            "type": "CrossEntropyLoss",
            "label_smoothing": 0.1,
            "ignore_index": -100,
            "shift_strategy": "T5 internal shift via labels argument",
        },
        "semantic_grounding": {
            "enabled": semantic_lambda > 0.0,
            "lambda": semantic_lambda,
            "use_warmup": use_warmup,
            "method": "attention_pool_cosine_distance" if semantic_lambda > 0.0 else "none",
        },
        "contrastive": {
            "enabled": contrastive_weight > 0.0,
            "weight": contrastive_weight,
            "method": "infonce" if contrastive_weight > 0.0 else "none",
        },
        "error_analysis": last_error_analysis,
        "decoder_unfreezing": {
            "layers_unfrozen": unfreeze_decoder_layers,
            "decoder_lr": decoder_lr if unfreeze_decoder_layers > 0 else None,
        },
        "checkpoint": {
            "loaded_from": checkpoint_path,
            "best_model_path": os.path.join(run_dir, "best_model.pt"),
            "best_rouge_l_model_path": os.path.join(run_dir, "best_rouge_l_model.pt"),
            "last_model_path": os.path.join(run_dir, "last_model.pt"),
            "best_val_loss": best_val_loss,
            "best_rouge_l": best_rouge_l,
            "start_epoch": start_epoch,
            "loaded_history_length": len(loaded_history),
        },
        "decode_config": asdict(decode_cfg),
        "history": [asdict(x) for x in history],
        "sample_predictions": sample_predictions,
    }
    save_json(os.path.join(run_dir, "report.json"), report)

    return report


def compare_runs(run_1k: dict[str, Any], run_3k: dict[str, Any]) -> dict[str, Any]:
    history_1k = run_1k.get("history", [])
    history_3k = run_3k.get("history", [])

    if not history_1k or not history_3k:
        return {
            "comparison_available": False,
            "reason": "Missing histories",
        }

    final_1k = history_1k[-1]
    final_3k = history_3k[-1]

    deltas = {
        "train_loss_delta_3k_minus_1k": float(final_3k["train_loss"] - final_1k["train_loss"]),
        "val_loss_delta_3k_minus_1k": float(final_3k["val_loss"] - final_1k["val_loss"]),
        "bleu_delta_3k_minus_1k": float(final_3k["bleu"] - final_1k["bleu"]),
        "rouge_l_delta_3k_minus_1k": float(final_3k["rouge_l"] - final_1k["rouge_l"]),
        "chrf_delta_3k_minus_1k": float(final_3k["chrf"] - final_1k["chrf"]),
        "input_diff_delta_3k_minus_1k": float(final_3k["input_difference_ratio"] - final_1k["input_difference_ratio"]),
        "dominant_ratio_delta_3k_minus_1k": float(final_3k["dominant_prediction_ratio"] - final_1k["dominant_prediction_ratio"]),
        "generic_ratio_delta_3k_minus_1k": float(final_3k["generic_sentence_ratio"] - final_1k["generic_sentence_ratio"]),
        "length_ratio_delta_3k_minus_1k": float(
            final_3k["output_to_target_length_ratio"] - final_1k["output_to_target_length_ratio"]
        ),
    }

    return {
        "comparison_available": True,
        "final_1k": final_1k,
        "final_3k": final_3k,
        "deltas": deltas,
        "trend_1k": history_1k,
        "trend_3k": history_3k,
    }


def compare_ce_vs_semantic(ce_run: dict[str, Any], sem_run: dict[str, Any]) -> dict[str, Any]:
    """Compare a CE-only run against a CE+semantic run on final-epoch metrics."""
    ce_hist = ce_run.get("history", [])
    sem_hist = sem_run.get("history", [])

    if not ce_hist or not sem_hist:
        return {"comparison_available": False, "reason": "Missing histories"}

    ce_final = ce_hist[-1]
    sem_final = sem_hist[-1]

    metric_keys = ["bleu", "rouge_l", "chrf", "generic_sentence_ratio",
                   "unique_prediction_ratio", "dominant_prediction_ratio",
                   "repetitive_output_ratio", "train_loss", "val_loss"]

    deltas = {
        f"{k}_delta_semantic_minus_ce": float(sem_final[k] - ce_final[k])
        for k in metric_keys
        if k in ce_final and k in sem_final
    }

    interpretation = {
        "bleu_improved": deltas.get("bleu_delta_semantic_minus_ce", 0.0) > 0,
        "rouge_l_improved": deltas.get("rouge_l_delta_semantic_minus_ce", 0.0) > 0,
        "chrf_improved": deltas.get("chrf_delta_semantic_minus_ce", 0.0) > 0,
        "generic_ratio_reduced": deltas.get("generic_sentence_ratio_delta_semantic_minus_ce", 0.0) < 0,
        "unique_ratio_improved": deltas.get("unique_prediction_ratio_delta_semantic_minus_ce", 0.0) > 0,
    }

    return {
        "comparison_available": True,
        "ce_final": ce_final,
        "semantic_final": sem_final,
        "deltas": deltas,
        "interpretation": interpretation,
        "ce_lambda": float(ce_run.get("semantic_grounding", {}).get("lambda", 0.0)),
        "semantic_lambda": float(sem_run.get("semantic_grounding", {}).get("lambda", 0.0)),
    }


def save_comparison_plot(
    ce_history: list[dict[str, Any]],
    sem_history: list[dict[str, Any]],
    metric: str,
    title: str,
    out_path: str,
) -> None:
    """Side-by-side epoch plot comparing CE-only vs CE+semantic for a single metric."""
    epochs_ce = [h["epoch"] for h in ce_history]
    epochs_sem = [h["epoch"] for h in sem_history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ce, [h[metric] for h in ce_history], marker="o", label="CE-only")
    plt.plot(epochs_sem, [h[metric] for h in sem_history], marker="s", label="CE+semantic")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled-scale T5 training and generalization test.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="artifacts/controlled_generalization_v2")
    parser.add_argument("--train-samples-small", type=int, default=1000,
                        help="Training samples for the small-scale run (default: 1000).")
    parser.add_argument("--train-samples-large", type=int, default=3000,
                        help="Training samples for the large-scale run (default: 3000).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per run (default: 10).")
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-target-len", type=int, default=48)
    parser.add_argument("--metric-eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-regularization-alpha", type=float, default=0.0)
    parser.add_argument(
        "--semantic-lambda",
        type=float,
        default=0.2,
        help="Weight for semantic cosine-distance loss (0.0 = CE-only, recommended range 0.05–0.3).",
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for InfoNCE contrastive loss (0.0 = disabled, recommended ≤ 0.2).",
    )
    parser.add_argument(
        "--no-warmup",
        dest="use_warmup",
        action="store_false",
        default=True,
        help="Disable λ warmup schedule (default: warmup is ON).",
    )
    parser.add_argument(
        "--lambda-sweep",
        action="store_true",
        default=False,
        help="Run a λ sweep (0, 0.1, 0.2, 0.3) at large scale instead of the default 4-run experiment.",
    )
    parser.add_argument(
        "--scale-study",
        action="store_true",
        default=False,
        help=(
            "Run a 2-experiment scale study: CE-only baseline + best semantic config "
            "(semantic_lambda, contrastive_weight) at train_samples_large. "
            "Intended for comparing performance across data scales."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to a .pt checkpoint to resume from. The model and optimizer states "
            "are loaded and training continues for --epochs additional epochs. "
            "Only used with --scale-study (applied to the semantic run) or the default mode."
        ),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        default=False,
        help=(
            "Run exactly one experiment at train_samples_large with the given "
            "semantic_lambda, contrastive_weight, and optionally a checkpoint. "
            "Saves best_model.pt and last_model.pt for downstream use."
        ),
    )
    parser.add_argument(
        "--unfreeze-decoder-layers",
        type=int,
        default=0,
        help="Number of top T5 decoder layers to unfreeze (0 = fully frozen, 1 = last layer only).",
    )
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=2e-5,
        help="Learning rate for unfrozen decoder parameters (default: 2e-5).",
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=2,
        help="Number of Transformer layers in the visual encoder (default: 2).",
    )
    parser.add_argument(
        "--encoder-dropout",
        type=float,
        default=0.0,
        help="Dropout rate in visual encoder layers (default: 0.0).",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=128,
        help="Number of frames to keep per sample after keyframe selection (default: 128).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Directory containing pre-built dataset cache files produced by build_cache.py. "
            "If set and the cache for --target-frames exists, loading skips all .pose file I/O "
            "and runs in seconds instead of ~90 minutes. "
            "Example: artifacts/dataset_cache"
        ),
    )
    parser.add_argument(
        "--decode-every-n",
        type=int,
        default=1,
        help=(
            "Run full beam-search decode every N epochs (default: 1 = every epoch). "
            "Set to 3 to skip decode on 2 out of 3 epochs for ~30-40%% speedup. "
            "The last epoch always runs full decode regardless of this setting."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root = os.path.abspath(args.root)
    args.output_dir = os.path.abspath(args.output_dir)

    if args.length_regularization_alpha < 0.0 or args.length_regularization_alpha > 0.05:
        raise ValueError("length_regularization_alpha must be in [0.0, 0.05].")
    if args.semantic_lambda < 0.0 or args.semantic_lambda > 2.0:
        raise ValueError("semantic_lambda must be in [0.0, 2.0].")
    if args.contrastive_weight < 0.0 or args.contrastive_weight > 2.0:
        raise ValueError("contrastive_weight must be in [0.0, 2.0].")

    train_small = args.train_samples_small
    train_large = args.train_samples_large
    epochs = args.epochs
    lam = args.semantic_lambda
    cw = args.contrastive_weight

    decode_cfg = DecodeConfig(
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )

    shared_kwargs: dict[str, Any] = {
        "root": args.root,
        "output_dir": args.output_dir,
        "val_samples": args.val_samples,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_target_len": args.max_target_len,
        "metric_eval_batches": args.metric_eval_batches,
        "seed": args.seed,
        "decode_cfg": decode_cfg,
        "use_warmup": args.use_warmup,
        "checkpoint_path": args.checkpoint_path,
        "unfreeze_decoder_layers": args.unfreeze_decoder_layers,
        "decoder_lr": args.decoder_lr,
        "encoder_layers": args.encoder_layers,
        "encoder_dropout": args.encoder_dropout,
        "target_frames": args.target_frames,
        "cache_dir": os.path.abspath(args.cache_dir) if args.cache_dir else None,
        "decode_every_n": args.decode_every_n,
    }

    if args.lambda_sweep:
        # ── λ sweep: CE vs λ=0.1, 0.2, 0.3 at large scale ───────────
        sweep_lambdas = [0.0, 0.1, 0.2, 0.3]
        sweep_runs: dict[str, dict[str, Any]] = {}

        for sweep_lam in sweep_lambdas:
            label = f"ce" if sweep_lam == 0.0 else f"sem{sweep_lam}"
            run_name = f"sweep_{train_large}train_e{epochs}_{label}_seed{args.seed}"
            sweep_runs[label] = run_single_experiment(
                run_name=run_name,
                train_samples=train_large,
                epochs=epochs,
                semantic_lambda=sweep_lam,
                contrastive_weight=cw if sweep_lam > 0.0 else 0.0,
                **shared_kwargs,
            )

        summary = {
            "config": {
                "root": args.root,
                "output_dir": args.output_dir,
                "train_samples_large": train_large,
                "epochs": epochs,
                "val_samples": args.val_samples,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "sweep_lambdas": sweep_lambdas,
                "contrastive_weight": cw,
                "use_warmup": args.use_warmup,
                "decode_config": asdict(decode_cfg),
            },
            "sweep_runs": {k: v.get("history", [])[-1] for k, v in sweep_runs.items()},
            "report_paths": {k: v.get("run_name", "") for k, v in sweep_runs.items()},
        }

        summary_path = os.path.join(args.output_dir, "lambda_sweep_summary.json")
        save_json(summary_path, summary)
        print("Saved λ-sweep summary to", summary_path)

        # ── Print λ-sweep table ──────────────────────────────────────
        print("\n=== λ Sweep Results (large scale) ===")
        header = f"{'λ':>6}  {'BLEU':>6}  {'ROUGE-L':>8}  {'CHRF':>6}  {'generic':>8}  {'val_loss':>8}"
        print(header)
        print("-" * len(header))
        for label, run in sweep_runs.items():
            hist = run.get("history", [])
            if hist:
                last = hist[-1]
                sweep_lam = last.get("effective_lambda", last.get("semantic_lambda", 0.0))
                print(
                    f"{sweep_lam:>6.2f}  {last['bleu']:>6.2f}  {last['rouge_l']:>8.2f}"
                    f"  {last['chrf']:>6.2f}  {last['generic_sentence_ratio']:>8.3f}"
                    f"  {last['val_loss']:>8.4f}"
                )
        return

    if args.single_run:
        # ── Single run: one experiment at large scale ─────────────────
        n = train_large
        run_label = (
            f"ce_extended_{n}train_e{epochs}"
            if lam == 0.0
            else f"sem{lam}_extended_{n}train_e{epochs}"
        )
        run = run_single_experiment(
            run_name=f"{run_label}_seed{args.seed}",
            train_samples=n,
            epochs=epochs,
            semantic_lambda=lam,
            contrastive_weight=cw,
            **shared_kwargs,
        )
        hist = run.get("history", [])
        if hist:
            last = hist[-1]
            ckpt_info = run.get("checkpoint", {})
            print(f"\n=== Single Run Final (epoch {last['epoch']}) ===")
            print(f"  val_loss  = {last['val_loss']:.4f}")
            print(f"  ROUGE-L   = {last['rouge_l']:.2f}")
            print(f"  BLEU      = {last['bleu']:.2f}")
            print(f"  CHRF      = {last['chrf']:.2f}")
            print(f"  dominant  = {last['dominant_prediction_ratio']:.3f}")
            print(f"  unique    = {last['unique_prediction_ratio']:.3f}")
            print(f"  generic   = {last['generic_sentence_ratio']:.3f}")
            print(f"  best_val_loss = {ckpt_info.get('best_val_loss', 'N/A')}")
            print(f"  best_model    = {ckpt_info.get('best_model_path', 'N/A')}")
        return

    if args.scale_study:
        # ── Scale study: CE baseline + best semantic config ───────────
        # Runs exactly 2 experiments at train_samples_large.
        # Designed to be called three times at sizes 1500, 5000, 15000
        # and then results compared via a separate analysis script.
        n = train_large

        run_ce = run_single_experiment(
            run_name=f"scale{n}_e{epochs}_ce_seed{args.seed}",
            train_samples=n,
            epochs=epochs,
            semantic_lambda=0.0,
            contrastive_weight=0.0,
            **shared_kwargs,
        )
        run_sem = run_single_experiment(
            run_name=f"scale{n}_e{epochs}_sem{lam}_cw{cw}_seed{args.seed}",
            train_samples=n,
            epochs=epochs,
            semantic_lambda=lam,
            contrastive_weight=cw,
            **shared_kwargs,
        )

        ce_vs_sem = compare_ce_vs_semantic(run_ce, run_sem)

        summary = {
            "config": {
                "root": args.root,
                "output_dir": args.output_dir,
                "train_samples": n,
                "epochs": epochs,
                "val_samples": args.val_samples,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "semantic_lambda": lam,
                "contrastive_weight": cw,
                "use_warmup": args.use_warmup,
                "decode_config": asdict(decode_cfg),
            },
            "ce_final": run_ce.get("history", [{}])[-1],
            "sem_final": run_sem.get("history", [{}])[-1],
            "ce_vs_sem": ce_vs_sem,
            "error_analysis": {
                "ce":  run_ce.get("error_analysis", {}),
                "sem": run_sem.get("error_analysis", {}),
            },
            "sample_predictions": {
                "ce":  run_ce.get("sample_predictions", [{}])[-1],
                "sem": run_sem.get("sample_predictions", [{}])[-1],
            },
        }

        summary_path = os.path.join(args.output_dir, f"scale_study_{n}train_summary.json")
        save_json(summary_path, summary)
        print(f"Saved scale-study summary to {summary_path}")

        # ── Print comparison table ───────────────────────────────────
        print(f"\n=== Scale Study Results ({n} training samples) ===")
        for label, h in [("CE-only", run_ce.get("history", [{}])[-1]),
                          (f"Semantic λ={lam}", run_sem.get("history", [{}])[-1])]:
            print(
                f"  {label:25s}  val_loss={h.get('val_loss', 0):.4f}"
                f"  ROUGE-L={h.get('rouge_l', 0):.2f}"
                f"  CHRF={h.get('chrf', 0):.2f}"
                f"  dominant={h.get('dominant_prediction_ratio', 0):.3f}"
                f"  unique={h.get('unique_prediction_ratio', 0):.3f}"
            )
        if ce_vs_sem.get("comparison_available"):
            d = ce_vs_sem["deltas"]
            print(
                f"\n  Δ (sem − CE):  ROUGE-L={d.get('rouge_l_delta_semantic_minus_ce', 0):+.2f}"
                f"  CHRF={d.get('chrf_delta_semantic_minus_ce', 0):+.2f}"
                f"  dominant={d.get('dominant_prediction_ratio_delta_semantic_minus_ce', 0):+.3f}"
            )
        return

    # ── Default: CE-only baseline  ──────────────────────────────────
    run_small_ce = run_single_experiment(
        run_name=f"run_1k_e{epochs}_ce_seed{args.seed}",
        train_samples=train_small,
        epochs=epochs,
        semantic_lambda=0.0,
        contrastive_weight=0.0,
        **shared_kwargs,
    )

    run_large_ce = run_single_experiment(
        run_name=f"run_3k_e{epochs}_ce_seed{args.seed}",
        train_samples=train_large,
        epochs=epochs,
        semantic_lambda=0.0,
        contrastive_weight=0.0,
        **shared_kwargs,
    )

    # ── CE + semantic loss ──────────────────────────────────────────
    run_small_sem = run_single_experiment(
        run_name=f"run_1k_e{epochs}_sem{lam}_seed{args.seed}",
        train_samples=train_small,
        epochs=epochs,
        semantic_lambda=lam,
        contrastive_weight=cw,
        **shared_kwargs,
    )

    run_large_sem = run_single_experiment(
        run_name=f"run_3k_e{epochs}_sem{lam}_seed{args.seed}",
        train_samples=train_large,
        epochs=epochs,
        semantic_lambda=lam,
        contrastive_weight=cw,
        **shared_kwargs,
    )

    # ── Comparisons ─────────────────────────────────────────────────
    scale_comparison = compare_runs(run_small_ce, run_large_ce)
    ce_vs_sem_small = compare_ce_vs_semantic(run_small_ce, run_small_sem)
    ce_vs_sem_large = compare_ce_vs_semantic(run_large_ce, run_large_sem)

    # ── Side-by-side comparison plots ──────────────────────────────
    plots_dir = ensure_dir(args.output_dir)
    for metric, ylabel in [
        ("train_loss", f"Train loss — {train_small} samples"),
        ("rouge_l",    f"ROUGE-L — {train_small} samples"),
        ("chrf",       f"CHRF — {train_small} samples"),
        ("generic_sentence_ratio", f"Generic ratio — {train_small} samples"),
    ]:
        save_comparison_plot(
            run_small_ce.get("history", []),
            run_small_sem.get("history", []),
            metric=metric,
            title=ylabel,
            out_path=os.path.join(plots_dir, f"comparison_{train_small}train_{epochs}ep_{metric}.png"),
        )

    for metric, ylabel in [
        ("train_loss", f"Train loss — {train_large} samples"),
        ("rouge_l",    f"ROUGE-L — {train_large} samples"),
        ("chrf",       f"CHRF — {train_large} samples"),
        ("generic_sentence_ratio", f"Generic ratio — {train_large} samples"),
    ]:
        save_comparison_plot(
            run_large_ce.get("history", []),
            run_large_sem.get("history", []),
            metric=metric,
            title=ylabel,
            out_path=os.path.join(plots_dir, f"comparison_{train_large}train_{epochs}ep_{metric}.png"),
        )

    summary = {
        "config": {
            "root": args.root,
            "output_dir": args.output_dir,
            "train_samples_small": train_small,
            "train_samples_large": train_large,
            "epochs": epochs,
            "val_samples": args.val_samples,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_target_len": args.max_target_len,
            "metric_eval_batches": args.metric_eval_batches,
            "seed": args.seed,
            "semantic_lambda": lam,
            "contrastive_weight": cw,
            "use_warmup": args.use_warmup,
            "criterion": {
                "label_smoothing": 0.1,
                "ignore_index": -100,
            },
            "decode_config": asdict(decode_cfg),
            "length_regularization_alpha": args.length_regularization_alpha,
            "deterministic_mode": True,
        },
        "scale_comparison_ce_only": scale_comparison,
        "ce_vs_semantic_1k": ce_vs_sem_small,
        "ce_vs_semantic_3k": ce_vs_sem_large,
        "report_paths": {
            "run_small_ce":       os.path.join(args.output_dir, f"run_1k_e{epochs}_ce_seed{args.seed}", "report.json"),
            "run_large_ce":       os.path.join(args.output_dir, f"run_3k_e{epochs}_ce_seed{args.seed}", "report.json"),
            "run_small_semantic": os.path.join(args.output_dir, f"run_1k_e{epochs}_sem{lam}_seed{args.seed}", "report.json"),
            "run_large_semantic": os.path.join(args.output_dir, f"run_3k_e{epochs}_sem{lam}_seed{args.seed}", "report.json"),
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary_path, summary)
    print("Saved comparison summary to", summary_path)

    # ── Print human-readable CE vs Semantic summary ────────────────
    for label, comp in [
        (f"CE-only vs CE+Semantic ({train_small} train)", ce_vs_sem_small),
        (f"CE-only vs CE+Semantic ({train_large} train)", ce_vs_sem_large),
    ]:
        print(f"\n=== {label} ===")
        if comp.get("comparison_available"):
            d = comp["deltas"]
            interp = comp["interpretation"]
            for m in ["bleu", "rouge_l", "chrf"]:
                print(f"  {m.upper():8s}: {d.get(f'{m}_delta_semantic_minus_ce', 0.0):+.3f}")
            print(f"  Generic ratio delta : {d.get('generic_sentence_ratio_delta_semantic_minus_ce', 0.0):+.3f}")
            print(f"  Unique ratio delta  : {d.get('unique_prediction_ratio_delta_semantic_minus_ce', 0.0):+.3f}")
            print(f"  Positives          : {[k for k, v in interp.items() if v]}")


if __name__ == "__main__":
    main()
