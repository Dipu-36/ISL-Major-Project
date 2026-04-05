from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import BaseModelOutput

from .temporal_encoder import TemporalVisualEncoder


class T5BridgeModel(nn.Module):
    def __init__(
        self,
        src_dim: int,
        t5_name: str = "t5-small",
        temporal_hidden: int = 256,
        freeze_t5: bool = True,
    ):
        super().__init__()
        self.temporal_encoder = TemporalVisualEncoder(src_dim=src_dim, d_model=temporal_hidden)
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)

        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False

        # Strengthened bridge: Linear → GELU → LayerNorm
        # The additional non-linearity gives the bridge capacity to adapt
        # the temporal encoder's feature distribution to T5's embedding space.
        self.bridge = nn.Sequential(
            nn.Linear(temporal_hidden, self.t5.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.t5.config.d_model),
        )

    def encode_visual(self, src: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pose features and project them into T5 embedding space.

        Returns:
            encoder_hidden: [B, T, d_model]  — bridged visual features.
            reduced_mask:   [B, T]           — valid position mask.
        """
        encoded, reduced_mask = self.temporal_encoder(src, attention_mask)
        bridged = self.bridge(encoded)
        return bridged, reduced_mask

    def encode_visual_pooled(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode visual features and return both the sequence and its pooled embedding.

        Computes the encoding once and derives the mean-pooled sentence vector
        without a second forward pass.

        Returns:
            encoder_hidden: [B, T, d_model]
            reduced_mask:   [B, T]
            pooled_emb:     [B, d_model]  — mean-pooled over valid positions.
        """
        encoder_hidden, reduced_mask = self.encode_visual(src, attention_mask)
        mask_f = reduced_mask.unsqueeze(-1).float()          # [B, T, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)             # [B, 1]
        pooled_emb = (encoder_hidden * mask_f).sum(dim=1) / denom  # [B, d_model]
        return encoder_hidden, reduced_mask, pooled_emb

    def unfreeze_decoder_top_n(self, n: int) -> list[str]:
        """Unfreeze the top-n decoder layers and the decoder final LayerNorm.

        Args:
            n: Number of decoder layers to unfreeze, counting from the output side.
               E.g. n=1 unfreezes the last decoder block + final_layer_norm.

        Returns:
            Names of all newly unfrozen parameters.
        """
        unfrozen: list[str] = []
        num_layers = len(self.t5.decoder.block)
        for i in range(num_layers - n, num_layers):
            for name, param in self.t5.decoder.block[i].named_parameters():
                param.requires_grad = True
                unfrozen.append(f"t5.decoder.block.{i}.{name}")
        if n > 0:
            for name, param in self.t5.decoder.final_layer_norm.named_parameters():
                param.requires_grad = True
                unfrozen.append(f"t5.decoder.final_layer_norm.{name}")
        return unfrozen

    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Seq2SeqLMOutput:
        encoder_hidden, reduced_mask = self.encode_visual(src, attention_mask)
        outputs = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=reduced_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 48,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        encoder_hidden, reduced_mask = self.encode_visual(src, attention_mask)
        generated = self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=reduced_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
        return generated
