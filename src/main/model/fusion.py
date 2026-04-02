import torch
import torch.nn as nn
from typing import Tuple

class CrossModalFusion(nn.Module):
    def __init__(self, audio_dim: int = 768, text_dim: int = 768, fused_dim: int = 768, num_heads: int = 8, num_layers: int = 2):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, fused_dim) if audio_dim != fused_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, fused_dim) if text_dim != fused_dim else nn.Identity()

        self.audio_type_embed = nn.Parameter(torch.zeros(1, 1, fused_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, fused_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=num_heads,
            dim_feedforward=fused_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, audio_hidden: torch.Tensor, text_hidden: torch.Tensor, audio_mask: torch.Tensor = None, text_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = audio_hidden.size(0)
        device = audio_hidden.device

        audio_h = self.audio_proj(audio_hidden) + self.audio_type_embed
        text_h = self.text_proj(text_hidden) + self.text_type_embed

        fused = torch.cat([audio_h, text_h], dim=1)

        if audio_mask is None:
            audio_mask = torch.ones(B, audio_hidden.size(1), device=device)
        if text_mask is None:
            text_mask = torch.ones(B, text_hidden.size(1), device=device)

        combined_mask = torch.cat([audio_mask, text_mask], dim=1)

        padding_mask = (combined_mask == 0)
        fused = self.fusion_transformer(fused, src_key_padding_mask=padding_mask)

        return fused, combined_mask
