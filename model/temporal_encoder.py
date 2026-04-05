from __future__ import annotations

import math

import torch
import torch.nn as nn

from .baseline import masked_mean


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Expects input of shape ``[B, T, d_model]`` (batch_first=True).
    The encoding is pre-computed once and stored as a non-trainable buffer.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the model but has no gradients
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] — add position-dependent offsets
        return x + self.pe[:, : x.size(1), :]


class TemporalVisualEncoder(nn.Module):
    def __init__(self, src_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(src_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_states = self.input_proj(src)
        token_states = self.pos_enc(token_states)  # inject sinusoidal position information
        key_padding_mask = attention_mask == 0
        encoded = self.encoder(token_states, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded), attention_mask


class TemporalEncoderLinearModel(nn.Module):
    def __init__(self, src_dim: int, vocab_size: int, d_model: int = 256, max_tgt_len: int = 64):
        super().__init__()
        self.visual_encoder = TemporalVisualEncoder(src_dim=src_dim, d_model=d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_tgt_len, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        self.vocab_head = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        encoded, encoded_mask = self.visual_encoder(src, attention_mask)
        context = masked_mean(encoded, encoded_mask)

        length = decoder_input_ids.shape[1]
        positions = torch.arange(length, device=decoder_input_ids.device).unsqueeze(0)

        token_state = self.token_emb(decoder_input_ids)
        pos_state = self.pos_emb(positions)
        fused = token_state + pos_state + context.unsqueeze(1)

        hidden = self.decoder(fused)
        logits = self.vocab_head(hidden)
        return logits
