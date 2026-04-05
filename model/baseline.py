from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    weighted = sequence * weights
    denom = torch.clamp(weights.sum(dim=1), min=1.0)
    return weighted.sum(dim=1) / denom


class EncoderLinearModel(nn.Module):
    def __init__(self, src_dim: int, vocab_size: int, d_model: int = 256, max_tgt_len: int = 64):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(src_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_tgt_len, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.vocab_head = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        src_tokens = self.src_proj(src)
        context = masked_mean(src_tokens, attention_mask)

        length = decoder_input_ids.shape[1]
        positions = torch.arange(length, device=decoder_input_ids.device).unsqueeze(0)

        token_state = self.token_emb(decoder_input_ids)
        pos_state = self.pos_emb(positions)
        fused = token_state + pos_state + context.unsqueeze(1)

        hidden = self.decoder(fused)
        logits = self.vocab_head(hidden)
        return logits
