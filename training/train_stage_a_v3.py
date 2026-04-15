#!/usr/bin/env python3
"""Stage A v3 — WPP-based encoder pretraining.

ROOT CAUSE ADDRESSED
--------------------
Both Stage A v1 (InfoNCE + MLP head) and v2 (InfoNCE + VICReg) failed to
produce any content signal (Stage B gate: 0/15 words at lift ≥1.15×).

Root cause: sentence-level InfoNCE against T5 embeddings provides near-zero
gradient toward lexical/semantic content on noisy iSign alignment.  Fixing
geometric collapse (VICReg) is insufficient when the objective itself carries
no content signal.

DESIGN DECISION
---------------
Replace sentence-level InfoNCE with Word Presence Prediction (WPP):
  - Multi-label binary classification: does word W appear in the reference?
  - Uses top-150 content words as WPP vocabulary (see training/word_inventory.py)
  - Primary loss: BCEWithLogitsLoss over [B, 150] label matrix
  - Secondary loss: Soft VICReg on encoder output to prevent collapse
                    (weaker than v2: var_weight=0.25 to preserve coarse structure)
  - NO T5 used in Stage A at all — encoder is self-contained

This approach is directly motivated by the iSign benchmark paper's WPP task,
which shows that word-level prediction is a more tractable representation
learning target than sentence-level translation on this dataset.

ARCHITECTURE (v2 config — 4-layer encoder)
------------------------------------------
  src_dim=450, target_frames=96, encoder_layers=4, d_model=256,
  nhead=8, dim_feedforward=1024, encoder_dropout=0.15

  Classification head: Linear(256, WPP_VOCAB_SIZE) + sigmoid
  (applied to mean-pooled encoder output)

CHECKPOINTS
-----------
  artifacts/phase9_encoder_pretrain/stageA_v3/
    encoder_best.pt    — best val WPP loss
    encoder_last.pt    — final epoch state
    history.json       — per-epoch metrics
    final_probe.json   — content signal summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.arch_v2_redesign import (
    D_MODEL, DIM_FEEDFORWARD, ENCODER_DROPOUT, ENCODER_LAYERS, FRAME_DROPOUT_RATE,
    JOINT_NOISE_STD, NHEAD, SRC_DIM, TARGET_FRAMES,
    VICREG_COEFF_COV, VICREG_COEFF_VAR, VICREG_GAMMA,
    WPP_BATCH_SIZE, WPP_EARLY_STOP_PATIENCE, WPP_LR, WPP_LR_MIN,
    WPP_MAX_EPOCHS, WPP_VOCAB_SIZE, WPP_WARMUP_EPOCHS, WPP_WEIGHT_DECAY,
)
from model.temporal_encoder import TemporalVisualEncoder
from training.word_inventory import load_wpp_vocabulary, make_wpp_labels
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


# ─────────────────────────────────────────────────────────────────────────────
# WPP Dataset — lightweight, no tokenizer required
# ─────────────────────────────────────────────────────────────────────────────

class WPPDataset(Dataset):
    """Dataset for Word Presence Prediction pretraining.

    Returns (src_tensor, attention_mask_tensor, text_string) triples.
    Loads directly from the preprocessed feature cache.
    """

    def __init__(
        self,
        uids: list[str],
        uid_to_text: dict[str, str],
        cache: dict,
    ) -> None:
        self.samples: list[dict] = []
        skipped = 0

        for uid in uids:
            text = uid_to_text.get(uid)
            if text is None:
                skipped += 1
                continue
            entry = cache.get(uid)
            if entry is None:
                skipped += 1
                continue
            src_np = entry["features"].astype("float32")
            mask_np = entry["attention_mask"].astype("int64")
            self.samples.append(
                {
                    "uid": uid,
                    "text": entry.get("text", text),
                    "src": torch.tensor(src_np, dtype=torch.float32),
                    "mask": torch.tensor(mask_np, dtype=torch.long),
                }
            )

        print(f"[WPPDataset] Loaded {len(self.samples)} samples, skipped {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate_wpp(batch: list[dict]) -> dict:
    """Stack tensors; keep text as a list of strings."""
    return {
        "src": torch.stack([b["src"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# WPP Model
# ─────────────────────────────────────────────────────────────────────────────

class WPPEncoder(nn.Module):
    """Temporal encoder + WPP classification head.

    The encoder is an upgraded TemporalVisualEncoder (4 layers, d=256).
    The classification head predicts which content words appear in the
    reference text, given the mean-pooled encoder output.
    """

    def __init__(
        self,
        src_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.encoder = TemporalVisualEncoder(
            src_dim=src_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Classification head: mean-pool → Linear → binary logits per word
        # No sigmoid here; BCEWithLogitsLoss applies it internally (numerically stable)
        self.head = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder, return (sequence_output [B,T,D], pooled [B,D])."""
        seq_out, out_mask = self.encoder(src, mask)
        # Mean-pool over valid frames
        valid = out_mask.float().unsqueeze(-1)          # [B, T, 1]
        pooled = (seq_out * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # [B, D]
        return seq_out, pooled

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (wpp_logits [B, V], pooled [B, D])."""
        seq_out, pooled = self.encode(src, mask)
        logits = self.head(pooled)
        return logits, pooled


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def frame_dropout(src: torch.Tensor, mask: torch.Tensor, rate: float) -> torch.Tensor:
    """Randomly zero a fraction of valid frames during training.

    Args:
        src:  [B, T, D] input features
        mask: [B, T]   binary valid-frame mask (1=valid, 0=padding)
        rate: fraction of valid frames to zero

    Returns:
        Augmented src tensor (new object, no in-place modification).
    """
    if rate <= 0.0:
        return src
    drop_mask = (torch.rand_like(mask.float()) < rate) & (mask == 1)
    return src * (~drop_mask).unsqueeze(-1).float()


def joint_noise(src: torch.Tensor, std: float, training: bool) -> torch.Tensor:
    """Add Gaussian noise to keypoint coordinates during training."""
    if not training or std <= 0.0:
        return src
    return src + torch.randn_like(src) * std


# ─────────────────────────────────────────────────────────────────────────────
# Soft VICReg regulariser (on pooled encoder output)
# ─────────────────────────────────────────────────────────────────────────────

def soft_vicreg_loss(
    z: torch.Tensor,
    var_weight: float = VICREG_COEFF_VAR,
    cov_weight: float = VICREG_COEFF_COV,
    gamma: float = VICREG_GAMMA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance + covariance regulariser on a batch of embeddings.

    Applied to the mean-pooled encoder output (not a separate projection).
    Kept SOFT (var_weight=0.25) to preserve coarse content structure.
    Full VICReg (var_weight=1.0) was shown in v2 to destroy coarse similarity.

    Args:
        z: [B, D] embeddings, NOT necessarily normalised
    Returns:
        (var_loss, cov_loss) scalars
    """
    B, D = z.shape
    if B < 2:
        zero = z.sum() * 0.0
        return zero, zero

    # Variance term: penalise std < gamma along each dimension
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_centered.var(dim=0) + 1e-4)  # [D]
    var_loss = F.relu(gamma - std).mean()

    # Covariance term: penalise off-diagonal entries of the cov matrix
    cov = (z_centered.T @ z_centered) / (B - 1)     # [D, D]
    cov_off_diag = cov.pow(2)
    cov_off_diag.fill_diagonal_(0.0)
    cov_loss = cov_off_diag.sum() / D

    return var_weight * var_loss, cov_weight * cov_loss


# ─────────────────────────────────────────────────────────────────────────────
# Probe helper (runs on validation set, logs per-word signal)
# ─────────────────────────────────────────────────────────────────────────────

def _run_probe(
    model: WPPEncoder,
    loader: DataLoader,
    word_to_idx: dict[str, int],
    vocab: list[str],
    device: torch.device,
    max_samples: int = 2000,
) -> dict:
    """Evaluate WPP signal quality beyond the training loss.

    Returns a dict with per-word AUC values and summary statistics.
    Uses sklearn's roc_auc_score; approximates with accuracy when AUC
    computation fails for rare words.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_logits, all_labels, all_pooled = [], [], []

    count = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            mask = batch["mask"].to(device)
            texts = batch["text"]
            labels = make_wpp_labels(texts, word_to_idx).to(device)

            logits, pooled = model(src, mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_pooled.append(pooled.cpu().numpy())

            count += src.size(0)
            if count >= max_samples:
                break

    logits_np = np.concatenate(all_logits, axis=0)   # [N, V]
    labels_np = np.concatenate(all_labels, axis=0)    # [N, V]
    pooled_np = np.concatenate(all_pooled, axis=0)    # [N, D]

    # Per-word AUC
    per_word_auc = {}
    for i, word in enumerate(vocab):
        y = labels_np[:, i]
        if y.sum() < 5 or y.sum() > len(y) - 5:
            continue  # skip words with too few positive/negative examples
        try:
            auc = float(roc_auc_score(y, logits_np[:, i]))
        except Exception:
            auc = 0.5
        per_word_auc[word] = round(auc, 4)

    words_above_60 = sum(1 for v in per_word_auc.values() if v >= 0.60)
    words_above_65 = sum(1 for v in per_word_auc.values() if v >= 0.65)
    avg_auc = float(np.mean(list(per_word_auc.values()))) if per_word_auc else 0.5

    # Isotropy at encoder output (mean pairwise cosine similarity; lower = better)
    N = min(500, len(pooled_np))
    sub = pooled_np[:N]
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    normed = sub / norms
    cos_matrix = normed @ normed.T
    off_diag = cos_matrix[np.triu_indices_from(cos_matrix, k=1)]
    isotropy = float(np.mean(np.abs(off_diag)))

    return {
        "per_word_auc": per_word_auc,
        "words_above_auc_0.60": words_above_60,
        "words_above_auc_0.65": words_above_65,
        "avg_auc": round(avg_auc, 4),
        "n_words_evaluated": len(per_word_auc),
        "isotropy_l1": round(isotropy, 4),
        "n_samples": int(count),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    model: WPPEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    word_to_idx: dict[str, int],
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    """One epoch of WPP training or validation.

    Returns dict with keys: wpp_loss, var_loss, cov_loss, total_loss.
    """
    model.train(training)
    total_wpp = total_var = total_cov = total = 0.0
    n_batches = 0

    for batch in loader:
        src = batch["src"].to(device)
        mask = batch["mask"].to(device)
        texts = batch["text"]
        labels = make_wpp_labels(texts, word_to_idx).to(device)

        if training:
            # Apply augmentations only during training
            src = frame_dropout(src, mask, FRAME_DROPOUT_RATE)
            src = joint_noise(src, JOINT_NOISE_STD, training=True)

        logits, pooled = model(src, mask)

        wpp_loss = F.binary_cross_entropy_with_logits(
            logits, labels,
            # Positive weight to up-weight rare words:
            # avg word frequency is ~5% so positives are underrepresented
            pos_weight=torch.full((labels.size(1),), 10.0, device=device),
        )

        var_loss, cov_loss = soft_vicreg_loss(pooled)
        loss = wpp_loss + var_loss + cov_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_wpp += wpp_loss.item()
        total_var += var_loss.item()
        total_cov += cov_loss.item()
        total += loss.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "wpp_loss": total_wpp / denom,
        "var_loss": total_var / denom,
        "cov_loss": total_cov / denom,
        "total_loss": total / denom,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule with linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def _get_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
) -> float:
    """Linear warmup followed by cosine decay.  Returns the new learning rate."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage A v3 — WPP-based encoder pretraining")
    p.add_argument("--run-name", default="stageA_v3", help="Run subdirectory under phase9_encoder_pretrain/")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Limit training samples (default: use all)")
    p.add_argument("--val-samples", type=int, default=None,
                   help="Limit validation samples (default: use all)")
    p.add_argument("--batch-size", type=int, default=WPP_BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=WPP_MAX_EPOCHS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--probe-every", type=int, default=5, help="Probe interval in epochs")
    p.add_argument("--vocab-path", default="", help="Override WPP vocab path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stageA_v3] Device: {device}")
    print(f"[stageA_v3] Run: {args.run_name}")

    run_dir = os.path.join(PROJECT_ROOT, "artifacts", "phase9_encoder_pretrain", args.run_name)
    ensure_dir(run_dir)

    # ── Load WPP vocabulary ────────────────────────────────────────────────
    vocab_path = args.vocab_path or os.path.join(PROJECT_ROOT, "artifacts", "wpp_vocab.json")

    if not os.path.exists(vocab_path):
        print(f"[stageA_v3] WPP vocabulary not found at {vocab_path}.")
        print(f"[stageA_v3] Building vocabulary from iSign_v1.1.csv ...")
        from training.word_inventory import build_wpp_vocabulary
        build_wpp_vocabulary(
            csv_path=os.path.join(PROJECT_ROOT, "iSign_v1.1.csv"),
            splits_json=os.path.join(PROJECT_ROOT, "artifacts", "splits.json"),
            vocab_size=WPP_VOCAB_SIZE,
            output_path=vocab_path,
        )

    vocab_data = load_wpp_vocabulary(vocab_path)
    vocab: list[str] = vocab_data["vocab"]
    word_to_idx: dict[str, int] = vocab_data["word_to_idx"]
    V = len(vocab)
    if V == 0:
        raise ValueError(
            f"Loaded empty WPP vocabulary from {vocab_path}. "
            "Rebuild it before starting Stage A."
        )
    print(f"[stageA_v3] WPP vocabulary size: {V}  (first 10: {vocab[:10]})")

    # ── Load data from cache ───────────────────────────────────────────────
    cache_path = os.path.join(PROJECT_ROOT, "artifacts", "dataset_cache", f"isl_tf{TARGET_FRAMES}.pt")
    print(f"[stageA_v3] Loading cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = payload["samples"]
    print(f"[stageA_v3] Cache entries: {len(cache)}")

    # Load UID→text mapping
    import csv as _csv
    uid_to_text: dict[str, str] = {}
    with open(os.path.join(PROJECT_ROOT, "iSign_v1.1.csv"), newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            uid = row.get("uid", "").strip()
            text = row.get("text", "").strip()
            if uid:
                uid_to_text[uid] = text

    # Load splits
    with open(os.path.join(PROJECT_ROOT, "artifacts", "splits.json")) as f:
        splits = json.load(f)

    rng = np.random.default_rng(args.seed)
    train_uids_all = list(splits.get("train_uids") or splits.get("train") or [])
    val_uids_all = list(splits.get("val_uids") or splits.get("val") or [])

    if not train_uids_all:
        raise ValueError(
            "No training UIDs found in splits.json. Expected one of: train_uids, train"
        )
    if not val_uids_all:
        raise ValueError(
            "No validation UIDs found in splits.json. Expected one of: val_uids, val"
        )

    rng.shuffle(train_uids_all)
    # Use all data by default; only subset if explicitly requested
    train_limit = args.train_samples if args.train_samples is not None else len(train_uids_all)
    val_limit = args.val_samples if args.val_samples is not None else len(val_uids_all)
    train_uids = train_uids_all[:train_limit]
    val_uids = val_uids_all[:val_limit]
    print(f"[stageA_v3] Training on {len(train_uids)} samples, validating on {len(val_uids)} samples")

    if not train_uids:
        raise ValueError(
            f"No training samples after sampling from {len(train_uids_all)} UIDs."
        )
    if not val_uids:
        raise ValueError(
            f"No validation samples after sampling from {len(val_uids_all)} UIDs."
        )

    train_ds = WPPDataset(train_uids, uid_to_text, cache)
    val_ds = WPPDataset(val_uids, uid_to_text, cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_wpp,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_wpp,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # ── Build model ───────────────────────────────────────────────────────
    model = WPPEncoder(
        src_dim=SRC_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=ENCODER_DROPOUT,
        vocab_size=V,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stageA_v3] Model parameters: {n_params:,}")

    # ── Optimizer with per-group weight decay ─────────────────────────────
    # Encoder: higher weight decay to regularise (risk of overfit on 50k samples)
    # Head: lower weight decay (small, needs to fit quickly)
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "weight_decay": WPP_WEIGHT_DECAY},
            {"params": head_params, "weight_decay": WPP_WEIGHT_DECAY * 0.1},
        ],
        lr=WPP_LR,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\n[stageA_v3] Starting WPP pretraining for up to {args.max_epochs} epochs")
    print(f"[stageA_v3] Early stop patience: {WPP_EARLY_STOP_PATIENCE} epochs")
    print(f"[stageA_v3] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[stageA_v3] Checkpoints → {run_dir}\n")

    for epoch in range(args.max_epochs):
        lr = _get_lr_with_warmup(
            optimizer, epoch,
            warmup_epochs=WPP_WARMUP_EPOCHS,
            base_lr=WPP_LR,
            min_lr=WPP_LR_MIN,
            total_epochs=args.max_epochs,
        )

        t0 = time.time()
        train_metrics = _run_epoch(model, train_loader, optimizer, word_to_idx, device, training=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, word_to_idx, device, training=False)
        elapsed = time.time() - t0

        row: dict = {
            "epoch": epoch + 1,
            "lr": round(lr, 8),
            "train_wpp_loss": round(train_metrics["wpp_loss"], 5),
            "train_total_loss": round(train_metrics["total_loss"], 5),
            "val_wpp_loss": round(val_metrics["wpp_loss"], 5),
            "val_total_loss": round(val_metrics["total_loss"], 5),
            "elapsed_s": round(elapsed, 1),
        }

        # Periodic probe
        if (epoch + 1) % args.probe_every == 0 or epoch == 0:
            probe = _run_probe(model, val_loader, word_to_idx, vocab, device)
            row["probe"] = probe
            print(
                f"  Epoch {epoch+1:>3d} | "
                f"train_wpp={train_metrics['wpp_loss']:.4f} "
                f"val_wpp={val_metrics['wpp_loss']:.4f} | "
                f"probe: avg_auc={probe['avg_auc']:.3f} "
                f"words≥0.60={probe['words_above_auc_0.60']} "
                f"iso={probe['isotropy_l1']:.3f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )
        else:
            print(
                f"  Epoch {epoch+1:>3d} | "
                f"train_wpp={train_metrics['wpp_loss']:.4f} "
                f"val_wpp={val_metrics['wpp_loss']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

        history.append(row)
        save_json(os.path.join(run_dir, "history.json"), history)

        # Checkpoint on best val WPP loss
        val_loss = val_metrics["wpp_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_wpp_loss": val_loss,
                    "arch": {
                        "src_dim": SRC_DIM, "d_model": D_MODEL, "nhead": NHEAD,
                        "num_layers": ENCODER_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
                        "dropout": ENCODER_DROPOUT, "vocab_size": V,
                    },
                    "vocab": vocab,
                },
                os.path.join(run_dir, "encoder_best.pt"),
            )
            print(f"    ✓ New best val_wpp_loss={val_loss:.5f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= WPP_EARLY_STOP_PATIENCE:
                print(f"\n[stageA_v3] Early stop: no val improvement for {WPP_EARLY_STOP_PATIENCE} epochs")
                break

    # Save final checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_wpp_loss": val_loss,
            "arch": {
                "src_dim": SRC_DIM, "d_model": D_MODEL, "nhead": NHEAD,
                "num_layers": ENCODER_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": ENCODER_DROPOUT, "vocab_size": V,
            },
            "vocab": vocab,
        },
        os.path.join(run_dir, "encoder_last.pt"),
    )

    # Final probe on best checkpoint
    ckpt = torch.load(os.path.join(run_dir, "encoder_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_probe = _run_probe(model, val_loader, word_to_idx, vocab, device, max_samples=5000)
    save_json(os.path.join(run_dir, "final_probe.json"), final_probe)

    print("\n" + "=" * 60)
    print("[stageA_v3] TRAINING COMPLETE")
    print(f"  Best val WPP loss:  {best_val_loss:.5f}")
    print(f"  Final probe avg AUC: {final_probe['avg_auc']:.3f}")
    print(f"  Words with AUC ≥ 0.60: {final_probe['words_above_auc_0.60']} / {final_probe['n_words_evaluated']}")
    print(f"  Encoder isotropy (L1): {final_probe['isotropy_l1']:.3f}")
    gate_threshold = 25
    gate_pass = final_probe["words_above_auc_0.60"] >= gate_threshold and final_probe["isotropy_l1"] < 0.50
    print(f"  Stage B gate (≥{gate_threshold} words @ AUC≥0.60, iso<0.50): {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print("=" * 60)

    save_json(
        os.path.join(run_dir, "final_probe.json"),
        {**final_probe, "gate_pass": gate_pass, "best_val_wpp_loss": best_val_loss},
    )

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
