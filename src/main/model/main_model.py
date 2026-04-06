import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers import BartTokenizer, BartForConditionalGeneration

# Importing the components
from .fusion import CrossModalFusion
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .citation_head import CitationHead

# Output dataclass for the model
@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


# Model Definition
class MainModel(nn.Module):
    def __init__(self,
        whispher_model: str = "openai/whisper-small",
        bart_model: str = "facebook/bart-base",
        freeze_audio: bool = True,
        freeze_text: bool = False,
        fused_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()

        self.audio_encoder = AudioEncoder(model_name=whispher_model, freeze=freeze_audio)
        self.text_encoder = TextEncoder(model_name=bart_model, freeze=freeze_text)
        self.fusion = CrossModalFusion(
            audio_dim=self.audio_encoder.hidden_size,
            text_dim=self.text_encoder.hidden_size,
            fused_dim=fused_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        bart = BartForConditionalGeneration.from_pretrained(bart_model)
        self.decoder = bart.model.decoder
        self.citation_head = CitationHead(model_name=bart_model)
        self.config = bart.config

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.config.decoder_start_token_id

        shifted[shifted == -100] = self.config.pad_token_id
        return shifted
    
    def forward(self,
        audio_features: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        audio_hidden = self.audio_encoder(audio_features)
        text_hidden = self.text_encoder(text_input_ids, text_attention_mask)

        fused_hidden, fused_mask = self.fusion(
            audio_hidden, text_hidden, text_mask=text_attention_mask
        )

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused_hidden,
            encoder_attention_mask=fused_mask
        )

        logits = self.citation_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits=logits,
            encoder_hidden_states=fused_hidden
        )
    
    @torch.no_grad()
    def generate(self,
        audio_features: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        max_length: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Autoregressive decoding without allocating a separate BartForConditionalGeneration."""
        audio_hidden = self.audio_encoder(audio_features)
        text_hidden = self.text_encoder(text_input_ids, text_attention_mask)

        fused_hidden, fused_mask = self.fusion(
            audio_hidden, text_hidden, text_mask=text_attention_mask
        )

        B = audio_features.shape[0]
        device = audio_features.device
        decoder_ids = torch.full(
            (B, 1), self.config.decoder_start_token_id,
            dtype=torch.long, device=device
        )
        eos_id = self.config.eos_token_id
        pad_id = self.config.pad_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            decoder_out = self.decoder(
                input_ids=decoder_ids,
                encoder_hidden_states=fused_hidden,
                encoder_attention_mask=fused_mask,
            )
            next_logits = self.citation_head(decoder_out.last_hidden_state[:, -1:, :]).squeeze(1)

            if do_sample and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Replace tokens for already-finished sequences with pad
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_id)
            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return decoder_ids
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MainModel":
        model = cls(*kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
