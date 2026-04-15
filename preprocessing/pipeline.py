from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from pose_format import Pose

BODY_START, BODY_END = 0, 33
LHAND_START, LHAND_END = 501, 522
RHAND_START, RHAND_END = 522, 543

L_SHOULDER_IDX = 11
R_SHOULDER_IDX = 12


@dataclass(frozen=True)
class PreprocessingConfig:
    target_frames: int = 96
    use_conf_mask: bool = True
    conf_threshold: float = 0.2
    exp_smoothing_alpha: float = 0.5
    max_text_length: int = 128


@dataclass(frozen=True)
class PreprocessedSample:
    uid: str
    text: str
    features: np.ndarray
    attention_mask: np.ndarray
    valid_length: int
    raw_length: int


def compute_attention_mask(target_frames: int, valid_length: int) -> np.ndarray:
    """Create attention mask where 1=valid frame, 0=padding.

    Args:
        target_frames: Total sequence length (including padding)
        valid_length: Number of valid (non-padding) frames

    Returns:
        Binary attention mask of shape (target_frames,)
    """
    mask = np.zeros((target_frames,), dtype=np.int64)
    mask[:min(valid_length, target_frames)] = 1
    return mask


def clean_text(text: str, max_chars: int = 256) -> str:
    lowered = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9'.,?!\s-]", " ", lowered)
    collapsed = re.sub(r"\s+", " ", normalized).strip()
    return collapsed[:max_chars]


def load_pose_body_hands(pose_path: str, cfg: PreprocessingConfig) -> np.ndarray:
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Missing pose file: {pose_path}")

    with open(pose_path, "rb") as handle:
        pose = Pose.read(handle.read())

    data = pose.body.data[:, 0, :, :]
    conf = pose.body.confidence[:, 0, :]

    xyz = data.filled(0.0) if hasattr(data, "filled") else np.asarray(data)
    confidence = np.asarray(conf)

    if cfg.use_conf_mask:
        # Soft confidence weighting: down-weight unreliable joints proportionally.
        # Using conf_threshold as a floor avoids completely zeroing any joint,
        # preserving spatial structure even for low-confidence detections.
        conf_weights = np.clip(confidence, cfg.conf_threshold, 1.0).astype(np.float32)
        xyz = xyz * conf_weights[..., None]

    body = xyz[:, BODY_START:BODY_END, :]
    lhand = xyz[:, LHAND_START:LHAND_END, :]
    rhand = xyz[:, RHAND_START:RHAND_END, :]
    merged = np.concatenate([body, lhand, rhand], axis=1)
    return merged.astype(np.float32)


def normalize_spatial(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)

    shoulders_mid = (x[:, L_SHOULDER_IDX, :] + x[:, R_SHOULDER_IDX, :]) / 2.0
    centered = x - shoulders_mid[:, None, :]

    shoulder_vec = centered[:, L_SHOULDER_IDX, :] - centered[:, R_SHOULDER_IDX, :]
    shoulder_scale = np.linalg.norm(shoulder_vec, axis=-1)

    valid = shoulder_scale > eps
    fallback = float(np.median(shoulder_scale[valid])) if np.any(valid) else 1.0
    safe_scale = np.where(valid, shoulder_scale, fallback)
    safe_scale = np.where(safe_scale > eps, safe_scale, 1.0)

    normalized = centered / safe_scale[:, None, None]
    return normalized.astype(np.float32)


def smooth_exponential(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply exponential moving average (EMA) smoothing along the time axis.

    Unlike a boxcar (moving-average) filter, EMA weights recent frames more
    heavily, better preserving sharp motion onsets while still attenuating
    high-frequency noise.  ``alpha=1.0`` leaves the signal unchanged;
    ``alpha→0`` produces a heavily smoothed (low-pass) result.
    """
    if x.shape[0] < 2:
        return x.astype(np.float32)

    smoothed = np.empty_like(x, dtype=np.float32)
    smoothed[0] = x[0]
    one_minus_alpha = 1.0 - alpha
    for t in range(1, x.shape[0]):
        smoothed[t] = alpha * x[t] + one_minus_alpha * smoothed[t - 1]
    return smoothed


def temporal_sample_keyframes(x: np.ndarray, target_frames: int) -> tuple[np.ndarray, int]:
    """Select *target_frames* frames by motion importance, preserving temporal order.

    Velocity magnitude (mean across all joints) scores each frame.  The top-K
    most dynamic frames are selected, with the first and last frame always
    included to preserve boundary context.  When the clip is too short the
    sequence is right-padded with its last frame.

    Returns:
        Tuple of (sampled_features, valid_length) where valid_length is the
        number of non-padding frames (original frames before padding).
    """
    T = x.shape[0]
    if T == 0:
        return (
            np.zeros((target_frames, x.shape[1] if x.ndim > 1 else 75, x.shape[2] if x.ndim > 2 else 3), dtype=np.float32),
            0
        )

    if T <= target_frames:
        pad = target_frames - T
        padded = np.concatenate([x, np.tile(x[-1:], (pad,) + (1,) * (x.ndim - 1))], axis=0).astype(np.float32)
        return padded, T  # valid_length = original length before padding

    # Frame-level velocity magnitude: mean L2 norm of per-joint displacements
    importance = np.zeros(T, dtype=np.float32)
    if T > 1:
        diff = x[1:] - x[:-1]                         # [T-1, joints, coords]
        mag = np.linalg.norm(diff, axis=-1).mean(axis=-1)  # [T-1]
        importance[1:] = mag

    # Always keep first and last frames
    importance[0] = importance.max() + 1.0
    importance[-1] = importance.max() + 1.0

    top_idx = np.argpartition(importance, -target_frames)[-target_frames:]
    selected_idx = np.sort(top_idx)  # restore temporal order
    # When downsampling, all target_frames are "valid" (real content)
    return x[selected_idx].astype(np.float32), target_frames


def add_velocity_features(x: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] > 1:
        velocity[1:] = x[1:] - x[:-1]
    merged = np.concatenate([x, velocity], axis=-1)
    return merged.reshape(merged.shape[0], -1).astype(np.float32)


def preprocess_single_sample(
    uid: str,
    raw_text: str,
    pose_path: str,
    cfg: Optional[PreprocessingConfig] = None,
) -> PreprocessedSample:
    active_cfg = cfg or PreprocessingConfig()
    text = clean_text(raw_text, max_chars=active_cfg.max_text_length)

    pose = load_pose_body_hands(pose_path, active_cfg)
    raw_len = int(pose.shape[0])

    normalized = normalize_spatial(pose)
    denoised = smooth_exponential(normalized, alpha=active_cfg.exp_smoothing_alpha)
    sampled, valid_len = temporal_sample_keyframes(denoised, active_cfg.target_frames)
    features = add_velocity_features(sampled)

    # Generate proper attention mask: 1 for valid frames, 0 for padding
    attention_mask = compute_attention_mask(active_cfg.target_frames, valid_len)

    return PreprocessedSample(
        uid=uid,
        text=text,
        features=features,
        attention_mask=attention_mask,
        valid_length=valid_len,
        raw_length=raw_len,
    )


def preprocess_uid_batch(
    uids: list[str],
    uid_to_text: Dict[str, str],
    pose_dir: str,
    cfg: Optional[PreprocessingConfig] = None,
) -> list[PreprocessedSample]:
    active_cfg = cfg or PreprocessingConfig()
    samples: list[PreprocessedSample] = []

    for uid in uids:
        text = uid_to_text.get(uid)
        if text is None:
            continue
        pose_path = os.path.join(pose_dir, uid + ".pose")
        sample = preprocess_single_sample(uid=uid, raw_text=text, pose_path=pose_path, cfg=active_cfg)
        samples.append(sample)
    return samples
