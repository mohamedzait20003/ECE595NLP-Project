"""
CiteMind Inference Pipeline
End-to-end: audio file + paper context -> ranked citation candidates
"""
import torch
import librosa
from pathlib import Path
from collections import Counter
from transformers import BartTokenizer, WhisperProcessor

from src.main.model.main_model import MainModel


class CitationPipeline:
    def __init__(
        self,
        checkpoint_path: str,
        whisper_model: str = "openai/whisper-small",
        bart_model: str = "facebook/bart-base",
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = MainModel(
            whispher_model=whisper_model,
            bart_model=bart_model,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        step = ckpt.get("step", "?")
        val_loss = ckpt.get("val_loss", None)
        best_reward = ckpt.get("best_reward", None)
        info = f"step={step}"
        if val_loss is not None:
            info += f", val_loss={val_loss:.4f}"
        if best_reward is not None:
            info += f", best_reward={best_reward:.4f}"
        print(f"[CitationPipeline] Loaded ({info}) on {self.device}")

        self.tokenizer = BartTokenizer.from_pretrained(bart_model)
        self.processor = WhisperProcessor.from_pretrained(whisper_model)

    def _encode_inputs(self, audio_path: str, source_title: str, source_abstract: str):
        waveform, _ = librosa.load(audio_path, sr=16000)
        audio_features = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        ctx = f"{source_title} </s> {source_abstract}"
        enc = self.tokenizer(ctx, return_tensors="pt", max_length=512, truncation=True)
        text_ids = enc["input_ids"].to(self.device)
        text_mask = enc["attention_mask"].to(self.device)

        return audio_features, text_ids, text_mask

    @torch.no_grad()
    def predict(
        self,
        audio_path: str,
        source_title: str,
        source_abstract: str,
        num_candidates: int = 5,
        max_length: int = 64,
        temperature: float = 0.8,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Generate citation candidates for a spoken academic claim.

        Args:
            audio_path: Path to .wav audio file of the spoken claim.
            source_title: Title of the citing paper.
            source_abstract: Abstract of the citing paper.
            num_candidates: Number of citation candidates to generate.
            max_length: Max token length of generated citation.
            temperature: Sampling temperature (higher = more diverse).
            deduplicate: Remove duplicate candidates from output.

        Returns:
            List of dicts sorted by confidence:
            [{"citation": str, "confidence": float, "count": int}]
        """
        audio_features, text_ids, text_mask = self._encode_inputs(
            audio_path, source_title, source_abstract
        )

        candidates = []

        # 1 greedy decode (highest probability)
        greedy_out = self.model.generate(
            audio_features=audio_features,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            max_length=max_length,
            do_sample=False,
        )
        candidates.append(self.tokenizer.decode(greedy_out[0], skip_special_tokens=True))

        # Remaining candidates via temperature sampling
        for _ in range(num_candidates - 1):
            out = self.model.generate(
                audio_features=audio_features,
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
            )
            candidates.append(self.tokenizer.decode(out[0], skip_special_tokens=True))

        counts = Counter(candidates)
        total = len(candidates)

        if deduplicate:
            seen, unique = set(), []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            candidates = unique

        results = [
            {"citation": c, "confidence": counts[c] / total, "count": counts[c]}
            for c in candidates
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    @torch.no_grad()
    def predict_batch(
        self,
        entries: list[dict],
        num_candidates: int = 5,
        temperature: float = 0.8,
    ) -> list[list[dict]]:
        """
        Run predict() on a list of entries.
        Each entry must have keys: audio_path, source_title, source_abstract.
        """
        return [
            self.predict(
                e["audio_path"], e["source_title"], e["source_abstract"],
                num_candidates=num_candidates, temperature=temperature,
            )
            for e in entries
        ]

    def predict_from_text(
        self,
        masked_sentence: str,
        source_title: str,
        source_abstract: str,
        num_candidates: int = 5,
        tts_rate: int = 150,
        **kwargs,
    ) -> list[dict]:
        """
        Generate citation from a masked text sentence (synthesizes audio on-the-fly).
        Requires pyttsx3.

        Args:
            masked_sentence: Sentence with [MASK] where the citation appears.
            source_title: Title of the citing paper.
            source_abstract: Abstract of the citing paper.
            num_candidates: Number of candidates to return.
            tts_rate: TTS speech rate (words per minute).
        """
        import tempfile
        import pyttsx3

        spoken = masked_sentence.replace("[MASK]", "citation needed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", tts_rate)
            engine.save_to_file(spoken, tmp_path)
            engine.runAndWait()
            return self.predict(
                tmp_path, source_title, source_abstract, num_candidates, **kwargs
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
