import torch
import torch.nn as nn
from transformers import BartModel

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/bart-base", freeze: bool = True):
        super().__init__()
        bart = BartModel.from_pretrained(model_name)
        self.encoder = bart.encoder
        self.hidden_size = self.encoder.config.d_model

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state