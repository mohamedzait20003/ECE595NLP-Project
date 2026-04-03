import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import WhisperProcessor, BartTokenizer

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class CustomDataset(Dataset):
    def __init__(self, manifest_path: str, 
        whisper_model: str = "openai/whisper-small",
        bart_model: str = "facebook/bart-base",
        max_audio_len: int = 480000,
        max_text_len: int = 512,
        max_target_len: int = 64,
        sample_rate: int = 16000
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.manifest = [e for e in raw if os.path.exists(e["audio_path"])]
        if len(self.manifest) < len(raw):
            print(f"[Dataset] Skipped {len(raw) - len(self.manifest)} entries with missing audio files.")

        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.bart_tokenizer = BartTokenizer.from_pretrained(bart_model)

        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.max_target_len = max_target_len
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        audio_path = entry["audio_path"]

        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        if len(waveform) > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]

        audio_features = self.whisper_processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        context = f"{entry['source_title']} </s> {entry['source_abstract']}"
        text_encoding = self.bart_tokenizer(
            context,
            max_length=self.max_text_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_encoding["input_ids"].squeeze(0)
        text_attention_mask = text_encoding["attention_mask"].squeeze(0)

        target_encoding = self.bart_tokenizer(
            entry["citation_string"],
            max_length=self.max_target_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"].squeeze(0) 

        return {
            "audio_features": audio_features,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "labels": labels
        }

