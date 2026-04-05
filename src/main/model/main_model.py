import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutput
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
    
    def generate(self,
        audio_features: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        **generate_kwargs
    ) -> torch.Tensor:
        audio_hidden = self.audio_encoder(audio_features)
        text_hidden = self.text_encoder(text_input_ids, text_attention_mask)

        fused_hidden, fused_mask = self.fusion(
            audio_hidden, text_hidden, text_mask=text_attention_mask
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden)

        device = fused_hidden.device
        bart_shell = BartForConditionalGeneration(self.config).to(device)
        bart_shell.model.decoder = self.decoder
        bart_shell.lm_head = self.citation_head.lm_head
        bart_shell.register_buffer(
            'final_logits_bias', self.citation_head.final_logits_bias
        )

        return bart_shell.generate(
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=fused_mask,
            **generate_kwargs
        )
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MainModel":
        model = cls(*kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
