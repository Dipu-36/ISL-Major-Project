#!/usr/bin/env python3
"""word_inventory.py — Build and cache the WPP (Word Presence Prediction) vocabulary.

This utility scans the iSign training split and produces the top-N most
frequent content words.  The result is cached as a JSON file so training
scripts do not re-scan the CSV on every run.

Output schema (artifacts/wpp_vocab.json):
    {
        "vocab": ["word1", "word2", ...],   // ordered by frequency, high→low
        "word_to_idx": {"word1": 0, ...},
        "frequencies": {"word1": 12345, ...},
        "total_train_samples": 103276
    }

Usage:
    python training/word_inventory.py                          # build and cache
    python training/word_inventory.py --vocab-size 200        # larger vocab
    python training/word_inventory.py --output custom_path.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.arch_v2_redesign import WPP_MIN_WORD_FREQ, WPP_VOCAB_CACHE, WPP_VOCAB_SIZE

# ── English stopword list (extended) ──────────────────────────────────────────
# These words are filtered even if frequent, as they carry no sign-level content.
STOPWORDS: frozenset = frozenset({
    # Articles, determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "no", "all", "both", "few", "more", "most", "other",
    "such", "what", "which", "whose",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "against", "along", "around", "near",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "whether", "as", "if", "because", "although", "while", "until", "since",
    "unless", "than", "even", "when", "where", "how",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should",
    "may", "might", "must", "can", "could",
    # Common function words
    "not", "also", "just", "only", "very", "too", "more", "much",
    "now", "here", "there", "then", "back", "way", "well",
    "still", "even", "already", "however", "therefore",
    # Frequent but semantically weak verbs for ISL
    "said", "say", "says", "saying", "told", "tell", "tells", "telling",
    "get", "got", "gets", "getting", "going", "went", "go", "come",
    "came", "comes", "coming", "make", "made", "makes", "making",
    "take", "took", "takes", "taking", "like", "liked", "likes",
    "know", "knew", "knows", "think", "thought", "thinks",
    "want", "wanted", "wants", "see", "saw", "seen", "look", "looked",
    "use", "used", "uses", "give", "gave", "given", "gives",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third",
    # Punctuation-adjacent tokens
    "s", "t", "d", "ll", "ve", "re", "m",
})


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    lowered = text.lower()
    # Remove possessives and contractions
    lowered = re.sub(r"'s\b", "", lowered)
    lowered = re.sub(r"n't\b", "", lowered)
    lowered = re.sub(r"'[a-z]+\b", "", lowered)
    # Remove non-alpha characters (keep hyphen within words)
    cleaned = re.sub(r"[^a-z\s-]", " ", lowered)
    # Split on whitespace and hyphens
    tokens = re.split(r"[\s-]+", cleaned)
    return [t for t in tokens if len(t) >= 3]  # minimum length filter


def build_wpp_vocabulary(
    csv_path: str,
    splits_json: str,
    vocab_size: int = WPP_VOCAB_SIZE,
    min_freq: int = WPP_MIN_WORD_FREQ,
    output_path: str = "",
) -> Dict:
    """Scan training split of iSign CSV and build WPP vocabulary.

    Args:
        csv_path:   Path to iSign_v1.1.csv.
        splits_json: Path to artifacts/splits.json (for train/val/test UIDs).
        vocab_size: Maximum vocabulary size (top-N words).
        min_freq:   Minimum frequency to include a word.
        output_path: Where to save the JSON file.  Uses WPP_VOCAB_CACHE if empty.

    Returns:
        dict with keys: vocab, word_to_idx, frequencies, total_train_samples.
    """
    if not output_path:
        output_path = os.path.join(PROJECT_ROOT, WPP_VOCAB_CACHE)

    # Load train UIDs from splits
    with open(splits_json) as f:
        splits = json.load(f)

    train_uid_list = splits.get("train_uids") or splits.get("train") or []
    train_uids: set = set(train_uid_list)
    if not train_uids:
        raise ValueError(
            f"No training UIDs found in {splits_json}. "
            "Expected one of: train_uids, train"
        )
    print(f"[word_inventory] {len(train_uids)} training UIDs loaded from splits.json")

    # Read CSV
    import csv
    uid_col, text_col = None, None
    counter: Counter = Counter()
    total_samples = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Detect column indices (handle various header formats)
        header_lower = [h.lower().strip() for h in header]
        for i, h in enumerate(header_lower):
            if h in ("uid", "id", "video_id", "name"):
                uid_col = i
            if h in ("text", "sentence", "caption", "translation", "gloss_english"):
                text_col = i
        if uid_col is None or text_col is None:
            # Fallback: assume first two columns are uid, text
            uid_col, text_col = 0, 1
            print(f"[word_inventory] WARNING: could not detect uid/text columns from header={header}; assuming cols 0,1")

        for row in reader:
            if len(row) <= max(uid_col, text_col):
                continue
            uid = row[uid_col].strip()
            text = row[text_col].strip()
            if uid not in train_uids:
                continue
            total_samples += 1
            tokens = _tokenize(text)
            tokens = [t for t in tokens if t not in STOPWORDS]
            counter.update(tokens)

    if total_samples == 0:
        raise ValueError(
            "WPP vocabulary scan matched zero training samples. "
            "Check that splits.json UIDs match iSign_v1.1.csv UID values."
        )

    print(f"[word_inventory] Scanned {total_samples} training samples")
    print(f"[word_inventory] Unique content words found: {len(counter)}")

    # Filter by min frequency
    filtered = {w: c for w, c in counter.items() if c >= min_freq}
    print(f"[word_inventory] After min_freq={min_freq} filter: {len(filtered)} words")

    # Take top-N by frequency
    top_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    vocab = [w for w, _ in top_words]
    frequencies = {w: c for w, c in top_words}

    if not vocab:
        raise ValueError(
            "WPP vocabulary is empty after frequency filtering. "
            f"Try lowering min_freq (current={min_freq}) or inspect tokenisation."
        )

    result = {
        "vocab": vocab,
        "word_to_idx": {w: i for i, w in enumerate(vocab)},
        "frequencies": frequencies,
        "total_train_samples": total_samples,
        "vocab_size": len(vocab),
        "min_freq": min_freq,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[word_inventory] Saved vocabulary ({len(vocab)} words) → {output_path}")
    print(f"[word_inventory] Top-20 words: {vocab[:20]}")

    return result


def load_wpp_vocabulary(vocab_path: str = "") -> Dict:
    """Load a previously built WPP vocabulary from JSON.

    Args:
        vocab_path: Path to the JSON file.  Defaults to WPP_VOCAB_CACHE.
    Returns:
        dict with keys: vocab, word_to_idx, frequencies, total_train_samples.
    Raises:
        FileNotFoundError: if the vocabulary file does not exist.
    """
    if not vocab_path:
        vocab_path = os.path.join(PROJECT_ROOT, WPP_VOCAB_CACHE)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"WPP vocabulary not found at {vocab_path}.\n"
            f"Run: python training/word_inventory.py"
        )
    with open(vocab_path) as f:
        result = json.load(f)
    if not result.get("vocab"):
        raise ValueError(
            f"WPP vocabulary at {vocab_path} is empty. "
            "Delete it and rebuild with python training/word_inventory.py."
        )
    return result


def make_wpp_labels(texts: List[str], word_to_idx: Dict[str, int], device: "torch.device" = None) -> "torch.Tensor":
    """Build multi-label binary targets for a batch of texts.

    Args:
        texts:       List of B reference strings.
        word_to_idx: Mapping from content word to vocabulary index.
        device:      Target device for the tensor (creates on CPU if None).

    Returns:
        Float tensor of shape [B, V] with 1.0 where the word appears in the
        corresponding text, 0.0 otherwise.  Import torch inside to keep this
        module importable without torch.
    """
    import torch
    V = len(word_to_idx)
    labels = torch.zeros(len(texts), V, device=device)
    for i, text in enumerate(texts):
        tokens = set(_tokenize(text))
        for token in tokens:
            if token in word_to_idx:
                labels[i, word_to_idx[token]] = 1.0
    return labels


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build WPP vocabulary from iSign training split")
    p.add_argument("--csv", default="iSign_v1.1.csv",
                   help="Path to iSign_v1.1.csv (default: project root)")
    p.add_argument("--splits", default="artifacts/splits.json",
                   help="Path to splits.json")
    p.add_argument("--vocab-size", type=int, default=WPP_VOCAB_SIZE,
                   help=f"Max vocabulary size (default: {WPP_VOCAB_SIZE})")
    p.add_argument("--min-freq", type=int, default=WPP_MIN_WORD_FREQ,
                   help=f"Min word frequency (default: {WPP_MIN_WORD_FREQ})")
    p.add_argument("--output", default="",
                   help="Output JSON path (default: artifacts/wpp_vocab.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(PROJECT_ROOT, args.csv)
    splits_path = args.splits if os.path.isabs(args.splits) else os.path.join(PROJECT_ROOT, args.splits)
    out_path = args.output if args.output else ""

    result = build_wpp_vocabulary(
        csv_path=csv_path,
        splits_json=splits_path,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        output_path=out_path,
    )
    print(f"\n[word_inventory] Done. Vocabulary size: {result['vocab_size']}")
