import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class CitationHead(nn.Module):
    def __init__(self, model_name: str = "facebook/bart-base"):
        super().__init__()
        bart = BartForConditionalGeneration.from_pretrained(model_name)

        self.lm_head = bart.lm_head
        self.final_logits_bias = bart.final_logits_bias

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(decoder_hidden) + self.final_logits_bias
        return logits