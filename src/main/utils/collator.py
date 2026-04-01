import torch
from transformers import BartTokenizer

class Collator:
    def __init__(self, bart_model: str = "facebook/bart-base"):
        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, features: list) -> dict:
        audio_features = torch.stack([f["audio_features"] for f in features])  # [B, 80, 3000]

        text_input_ids = [f["text_input_ids"] for f in features]
        text_attention_mask = [f["text_attention_mask"] for f in features]

        max_text_len = max(ids.size(0) for ids in text_input_ids)

        padded_text_ids = []
        padded_text_mask = []

        for ids, mask in zip(text_input_ids, text_attention_mask):
            pad_len = max_text_len - ids.size(0)
            padded_text_ids.append(
                torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
            )
            
            padded_text_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        text_input_ids = torch.stack(padded_text_ids)
        text_attention_mask = torch.stack(padded_text_mask)

        labels_list = [f["labels"] for f in features]
        max_label_len = max(l.size(0) for l in labels_list)

        padded_labels = []
        for lbl in labels_list:
            pad_len = max_label_len - lbl.size(0)
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )

        labels = torch.stack(padded_labels)

        return {
            "audio_features": audio_features,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "labels": labels,
        }