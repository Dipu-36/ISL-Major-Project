#!/usr/bin/env python3
"""Benchmark evaluation for Stage C v2 models.

Evaluates checkpoints using the protocol defined in BENCHMARK_PROTOCOL.md
for fair comparison against the iSign paper baseline (BLEU-4 1.47, ROUGE-L 16.67).

USAGE
-----
    # Evaluate best BLEU checkpoint on validation set (default)
    python evaluate_stage_c_benchmark.py \
        --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_run1/best_bleu_model.pt \
        --split val

    # Evaluate on test set (final evaluation only)
    python evaluate_stage_c_benchmark.py \
        --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_run1/best_bleu_model.pt \
        --split test

    # Specify custom batch size for decoding
    python evaluate_stage_c_benchmark.py \
        --checkpoint artifacts/phase9_decoder_finetune/stageC_v2_run1/best_bleu_model.pt \
        --split val --batch-size 64

REPORTED METRICS
----------------
- BLEU-4 (primary metric for comparison with iSign paper)
- ROUGE-L (secondary metric)
- dominant_ratio: frequency of most common prediction / total samples
- generic_ratio: frequency of "he said that he..." hallucinations

OUTPUT FORMAT
-------------
Results are printed to stdout and saved to:
    artifacts/phase9_decoder_finetune/<run_name>/benchmark_<split>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter

import sacrebleu
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.arch_v2_redesign import (
    D_MODEL, DIM_FEEDFORWARD, ENCODER_DROPOUT,
    ENCODER_LAYERS, NHEAD, SRC_DIM, T5_NAME, TARGET_FRAMES,
)
from training.train_stage_c_v2 import WPPTranslationModel
from utils.io import ensure_dir, save_json
from utils.seed import set_global_seed


# ============================================================================
# Dataset for evaluation (same as train_stage_c_v2.py)
# ============================================================================

class TranslationDataset(Dataset):
    """Dataset that returns pose features + text string for evaluation."""

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

            self.samples.append({
                "uid": uid,
                "text": clean_text,
                "src_np": src_np,
                "mask_np": mask_np,
            })

        print(f"[BenchmarkDataset] Loaded {len(self.samples)}, skipped {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        sample = self.samples[i]
        return {
            "uid": sample["uid"],
            "text": sample["text"],
            "src": torch.from_numpy(sample["src_np"]),
            "mask": torch.from_numpy(sample["mask_np"]),
        }


def _collate_eval(batch: list[dict]) -> dict:
    src = torch.stack([b["src"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    return {
        "src": src,
        "mask": mask,
        "text": [b["text"] for b in batch],
        "uids": [b["uid"] for b in batch],
    }


# ============================================================================
# Metrics (same as train_stage_c_v2.py)
# ============================================================================

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
    """Compute benchmark metrics for comparison with iSign paper."""
    if not preds or not refs or len(preds) != len(refs):
        return {"bleu": 0.0, "rouge_l": 0.0, "dominant_ratio": 0.0, "generic_ratio": 0.0}

    # BLEU-4 using sacrebleu (standard for iSign benchmark)
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl").score

    # ROUGE-L (F1-based)
    rouge = statistics.mean(_rouge_l(p, r) for p, r in zip(preds, refs)) * 100.0

    # Dominant ratio: measure of collapse (lower is better)
    most_common_count = Counter(preds).most_common(1)[0][1]
    dominant_ratio = most_common_count / len(preds)

    # Generic ratio: "he said that he..." hallucinations (lower is better)
    generic_count = sum(
        1 for p in preds
        if re.match(r"^he said that he", p.lower())
    )
    generic_ratio = generic_count / len(preds)

    return {
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4),
        "dominant_ratio": round(dominant_ratio, 4),
        "generic_ratio": round(generic_ratio, 4),
        "num_samples": len(preds),
    }


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "val",
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Evaluate a Stage C checkpoint on the specified split.

    Args:
        checkpoint_path: Path to best_bleu_model.pt or best_loss_model.pt
        split: "val" or "test"
        batch_size: Batch size for decoding
        num_workers: DataLoader workers
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing all benchmark metrics
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=" * 70)
    print(f"BENCHMARK EVALUATION")
    print(f"=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print(f"=" * 70)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?")
    print(f"\nCheckpoint epoch: {epoch}")

    # Extract run name from checkpoint path
    run_dir = os.path.dirname(checkpoint_path)
    run_name = os.path.basename(run_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)

    # Load splits
    splits_path = os.path.join(PROJECT_ROOT, "artifacts", "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    # Determine which UIDs to use
    if split == "val":
        uids = list(splits.get("val_uids", splits.get("val", [])))
    elif split == "test":
        uids = list(splits.get("test_uids", splits.get("test", [])))
    else:
        raise ValueError(f"Unknown split: {split}. Use 'val' or 'test'.")

    print(f"\nEvaluating on {len(uids)} {split} samples")

    # Load cache
    cache_path = os.path.join(PROJECT_ROOT, "artifacts", "dataset_cache", f"isl_tf{TARGET_FRAMES}.pt")
    print(f"Loading cache from: {cache_path}")
    t0 = time.time()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = payload["samples"]
    print(f"Cache loaded in {time.time()-t0:.1f}s ({len(cache)} entries)")

    # Load text mappings
    csv_path = os.path.join(PROJECT_ROOT, "iSign_v1.1.csv")
    uid_to_text: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv as _csv
        for row in _csv.DictReader(f):
            uid = row.get("uid", "").strip()
            text = row.get("text", "").strip()
            if uid:
                uid_to_text[uid] = text

    # Build dataset
    eval_dataset = TranslationDataset(uids, uid_to_text, cache)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_eval,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    # Build model from checkpoint architecture
    ckpt_sd = ckpt["model_state_dict"]

    # Infer vocab size from wpp_head weight
    vocab_size = ckpt_sd.get("wpp_head.weight", torch.empty(0)).shape[0]
    if vocab_size == 0:
        # Try alternate key
        vocab_size = ckpt_sd.get("wpp_head.0.weight", torch.empty(0)).shape[0]
    if vocab_size == 0:
        # Default
        vocab_size = 150
        print(f"Warning: Could not infer vocab size, using default {vocab_size}")
    else:
        print(f"Inferred vocab size: {vocab_size}")

    model = WPPTranslationModel(
        src_dim=SRC_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=ENCODER_DROPOUT,
        t5_name=T5_NAME,
        vocab_size=vocab_size,
        freeze_t5=False,  # Allow decoder to be used
    ).to(device)

    model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    print(f"Model loaded and set to eval mode")

    # Run evaluation
    print(f"\nDecoding all {split} samples...")
    all_preds: list[str] = []
    all_refs: list[str] = []
    all_uids: list[str] = []

    total_batches = len(eval_loader)
    decode_t0 = time.time()

    for batch_idx, batch in enumerate(eval_loader, start=1):
        src = batch["src"].to(device)
        mask = batch["mask"].to(device)
        refs = batch["text"]
        uids = batch["uids"]

        # Beam search decode
        pred_ids = model.generate(
            src, mask,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(refs)
        all_uids.extend(uids)

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            elapsed = time.time() - decode_t0
            samples_done = batch_idx * batch_size
            print(f"  Batch {batch_idx}/{total_batches} | {samples_done}/{len(eval_dataset)} samples | {elapsed:.0f}s elapsed")

    decode_time = time.time() - decode_t0
    print(f"\nDecoding complete in {decode_time:.1f}s ({len(all_preds)/decode_time:.1f} samples/sec)")

    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = _compute_translation_metrics(all_preds, all_refs)

    # Save detailed results
    detailed_results = {
        "checkpoint": checkpoint_path,
        "epoch": epoch,
        "split": split,
        "num_samples": len(all_preds),
        "metrics": metrics,
        "decode_time_sec": round(decode_time, 2),
        "samples_per_sec": round(len(all_preds) / decode_time, 2) if decode_time > 0 else 0,
        "predictions": [
            {"uid": uid, "reference": ref, "prediction": pred}
            for uid, ref, pred in zip(all_uids, all_refs, all_preds)
        ],
    }

    # Save results to run directory
    results_filename = f"benchmark_{split}.json"
    results_path = os.path.join(run_dir, results_filename)
    save_json(results_path, detailed_results)
    print(f"\nDetailed results saved to: {results_path}")

    return metrics


def print_results_table(metrics: dict[str, float], split: str) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS — {split.upper()} SET")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Value':>15} {'Paper Baseline':>20}")
    print(f"  {'─' * 25} {'─' * 15} {'─' * 20}")
    print(f"  {'BLEU-4':<25} {metrics['bleu']:>15.4f} {1.47:>20.4f}")
    print(f"  {'ROUGE-L':<25} {metrics['rouge_l']:>15.4f} {16.67:>20.2f}")
    print(f"  {'Dominant Ratio':<25} {metrics['dominant_ratio']:>15.4f} {'(lower better)':>20}")
    print(f"  {'Generic Ratio':<25} {metrics['generic_ratio']:>15.4f} {'(lower better)':>20}")
    print(f"  {'Samples':<25} {metrics.get('num_samples', 'N/A'):>15}")
    print("=" * 70)

    # Compare to paper baseline
    bleu = metrics['bleu']
    rouge = metrics['rouge_l']
    print(f"\nComparison to iSign Paper Baseline:")
    print(f"  BLEU-4:  {bleu:.4f} vs 1.47 (target: > 1.47) {'✓ PASS' if bleu > 1.47 else '✗ BELOW'}")
    print(f"  ROUGE-L: {rouge:.2f} vs 16.67 (target: > 16.67) {'✓ PASS' if rouge > 16.67 else '✗ BELOW'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark evaluation for Stage C v2 models"
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint (best_bleu_model.pt or best_loss_model.pt)"
    )
    p.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Dataset split to evaluate on (default: val)"
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for decoding (default: 32)"
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Run evaluation
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Print results
    print_results_table(metrics, args.split)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
