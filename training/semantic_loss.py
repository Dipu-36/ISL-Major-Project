"""Semantic grounding losses for ISL translation.

Two complementary losses:

1. SemanticGroundingLoss (cosine distance)
   ─ Computes attention-weighted sentence embeddings for predicted and target text.
   ─ Loss = mean(1 - cosine_similarity(pred_emb, tgt_emb)).

2. ContrastivePairLoss (InfoNCE)
   ─ Pulls together each (visual_i, text_i) positive pair.
   ─ Pushes apart all (visual_i, text_j) negatives using symmetric InfoNCE.
   ─ Operates on bridge output (already in T5 embedding space).

Both losses are fully differentiable through the model's trainable parameters;
neither modifies the frozen T5 backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


# ──────────────────────────────────────────────────────────────────────────────
# Shared pooling utilities
# ──────────────────────────────────────────────────────────────────────────────

def _mean_pool_with_mask(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``hidden`` [B, T, D] over valid positions in ``mask`` [B, T].

    Returns [B, D].  All-masked rows fall back to zero vector.
    """
    mask_f = mask.unsqueeze(-1).float()          # [B, T, 1]
    denom = mask_f.sum(dim=1).clamp(min=1.0)     # [B, 1]
    return (hidden * mask_f).sum(dim=1) / denom  # [B, D]


class AttentionPool(nn.Module):
    """Single-headed attention pooling over a token sequence.

    Learns a scalar score for each position so the pooled vector concentrates
    on the most semantically important tokens rather than averaging uniformly.
    Differentiable and handles variable-length sequences via masking.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, D] — token representations.
            mask:   [B, T]    — 1 for valid positions, 0 for padding.

        Returns:
            pooled: [B, D]
        """
        scores = self.score(hidden).squeeze(-1)          # [B, T]
        scores = scores.masked_fill(mask == 0, -1e4)     # mask padding
        weights = torch.softmax(scores, dim=-1)          # [B, T]
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]


# ──────────────────────────────────────────────────────────────────────────────
# Loss 1 — Semantic cosine-distance loss
# ──────────────────────────────────────────────────────────────────────────────

class SemanticGroundingLoss(nn.Module):
    """Cosine-distance loss using T5's embedding space with attention pooling.

    Replaces mean pooling with a learned attention-weighted pooling to produce
    richer sentence representations that emphasise semantically important tokens.

    Args:
        t5_model:    Frozen T5 model; shared embedding weights and encoder are
                     borrowed for computing target embeddings.
        temperature: Softmax temperature applied to logits during soft aggregation.
    """

    def __init__(self, t5_model: PreTrainedModel, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.t5 = t5_model
        self.temperature = temperature
        d_model: int = t5_model.config.d_model
        # Independent attention heads for prediction and target — they see
        # different input distributions and benefit from separate weights.
        self.pred_pool = AttentionPool(d_model)
        self.tgt_pool = AttentionPool(d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        labels_for_loss: torch.Tensor,
        labels_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:          T5 decoder logits, shape [B, T, vocab_size].
            labels_for_loss: Target IDs with -100 for padding, shape [B, T].
            labels_clean:    Target IDs with pad_token_id, shape [B, T].

        Returns:
            Scalar cosine-distance loss in [0, 2].
        """
        valid_mask = labels_for_loss.ne(-100)  # [B, T]

        pred_emb = self._soft_prediction_embedding(logits, valid_mask)    # [B, D]
        tgt_emb = self._target_text_embedding(labels_clean, valid_mask)   # [B, D]

        similarity = F.cosine_similarity(pred_emb, tgt_emb.detach(), dim=-1)  # [B]
        return (1.0 - similarity).mean()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _soft_prediction_embedding(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable sentence embedding via soft token aggregation.

        soft[b, t] = sum_v softmax(logits[b, t] / temp)[v] * emb[v]
        Then attention-pool over valid positions.
        """
        probs = F.softmax(logits / self.temperature, dim=-1)  # [B, T, V]
        emb_weight = self.t5.shared.weight                     # [V, D]
        soft_hidden = torch.matmul(probs, emb_weight)          # [B, T, D]
        return self.pred_pool(soft_hidden, valid_mask)         # [B, D]

    @torch.no_grad()
    def _target_text_embedding(
        self,
        labels_clean: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode target text with T5 encoder + attention pool (no gradient)."""
        encoder_out = self.t5.encoder(
            input_ids=labels_clean,
            attention_mask=valid_mask.long(),
        )
        return self.tgt_pool(encoder_out.last_hidden_state, valid_mask)  # [B, D]


# ──────────────────────────────────────────────────────────────────────────────
# Loss 2 — InfoNCE contrastive alignment loss
# ──────────────────────────────────────────────────────────────────────────────

class ContrastivePairLoss(nn.Module):
    """Symmetric InfoNCE loss for visual–text pair alignment.

    For a batch of B pairs (pose_i, text_i):
      - B positive pairs: matched (visual_i, text_i)
      - B*(B-1) negatives: mismatched (visual_i, text_j), i≠j

    Visual embeddings are projected from the bridge output (already in T5
    space), so no architectural changes to T5 are required.  Text embeddings
    come from the FROZEN T5 encoder (no_grad).

    Args:
        t5_model:    Frozen T5 model (encoder borrowed for text embeddings).
        d_model:     Dimension of the visual bridge output (= T5 d_model when
                     the bridge maps to T5 space).
        proj_dim:    Projection head output dimension for contrastive space.
        temperature: InfoNCE temperature τ; smaller = harder negatives.
    """

    def __init__(
        self,
        t5_model: PreTrainedModel,
        d_model: int,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.t5 = t5_model
        self.temperature = temperature

        # Lightweight projection heads — two separate linear+norm modules
        # that map each modality into a shared unit-sphere space.
        self.proj_visual = nn.Sequential(
            nn.Linear(d_model, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(t5_model.config.d_model, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
        )
        # Attention pooling for visual encoder outputs
        self.visual_pool = AttentionPool(d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_hidden: torch.Tensor,
        visual_mask: torch.Tensor,
        labels_clean: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_hidden: [B, T, d_model]  — bridge output from encode_visual.
            visual_mask:   [B, T]           — 1 for valid visual positions.
            labels_clean:  [B, T_text]      — target token IDs (no -100).
            text_mask:     [B, T_text]      — 1 for valid text positions.

        Returns:
            Scalar symmetric InfoNCE loss.  Returns 0.0 for batch_size < 2.
        """
        B = visual_hidden.shape[0]
        if B < 2:
            return visual_hidden.new_zeros(())

        # ── Visual side (gradient flows back through bridge) ────────
        v_pooled = self.visual_pool(visual_hidden, visual_mask)       # [B, d_model]
        v = F.normalize(self.proj_visual(v_pooled), dim=-1)           # [B, proj_dim]

        # ── Text side (frozen T5 encoder, no gradient) ──────────────
        with torch.no_grad():
            text_out = self.t5.encoder(
                input_ids=labels_clean,
                attention_mask=text_mask.long(),
            )
        t_hidden = text_out.last_hidden_state                         # [B, T_text, D]
        t_pooled = _mean_pool_with_mask(t_hidden, text_mask)          # [B, D]
        t = F.normalize(self.proj_text(t_pooled), dim=-1)             # [B, proj_dim]

        # ── Symmetric InfoNCE ────────────────────────────────────────
        labels = torch.arange(B, device=v.device)
        sim = torch.matmul(v, t.T) / self.temperature                 # [B, B]

        # Clamp similarity matrix for numerical safety
        sim = sim.clamp(-50.0, 50.0)

        loss_v2t = F.cross_entropy(sim, labels)       # visual → text direction
        loss_t2v = F.cross_entropy(sim.T, labels)     # text → visual direction
        return (loss_v2t + loss_t2v) / 2.0
