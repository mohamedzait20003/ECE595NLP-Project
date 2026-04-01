import torch
import torch.nn as nn
from transformers import WhisperModel

class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/whisper-small", freeze: bool = True):
        super().__init__()
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        self.hidden_size = self.encoder.config.d_model

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_features=input_features)
        return outputs.last_hidden_state