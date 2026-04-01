import os
import json
import pyttsx3
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Synthesizer:
    def __init__(self, sample_rate: int = 16000, rate: int = 150, volume: float = 1.0, output_dir: str = "src/data/audio"):
        self.sample_rate = sample_rate
        self.rate = rate
        self.volume = volume
        self.output_dir = output_dir or str(PROJECT_ROOT / "src" / "data" / "audio")
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_engine(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        return engine
    
    def synthesize_one(self, text: str, filename: str):
        spoken_text = text.replace("[MASK]", "citation needed")

        output_path = os.path.join(self.output_dir, f"{filename}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        temp_path = output_path + ".tmp.wav"
        engine = self._init_engine()
        engine.save_to_file(spoken_text, temp_path)
        engine.runAndWait()
        engine.stop()

        self._resample(temp_path, output_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return output_path
    
    def _resample(self, input_path: str, output_path: str):
        data, orig_sr = sf.read(input_path)

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        if orig_sr != self.sample_rate:
            duration = len(data) / orig_sr
            target_len = int(duration * self.sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data
            ).astype(np.float32)

        max_val = np.abs(data).max()
        if max_val > 0:
            data = data / max_val

        sf.write(output_path, data, self.sample_rate)

    def synthesize_split(self, split_path: str, split_name: str = "train") -> str:
        with open(split_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        split_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        manifest = []

        for i, example in enumerate(tqdm(examples, desc=f"Synthesizing {split_name}")):
            filename = os.path.join(split_name, f"{split_name}_{i:06d}")
            try:
                audio_path = self.synthesize_one(
                    text=example["masked_sentence"],
                    filename=filename
                )
                manifest.append({
                    "index": i,
                    "audio_path": audio_path,
                    "masked_sentence": example["masked_sentence"],
                    "citation_string": example["citation_string"],
                    "source_title": example["source_title"],
                    "source_abstract": example["source_abstract"],
                })
            except Exception as e:
                print(f"Failed on example {i}: {e}")
                continue

        manifest_path = os.path.join(self.output_dir, f"{split_name}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Synthesized {len(manifest)}/{len(examples)} examples -> {split_dir}")
        print(f"Manifest saved to {manifest_path}")

        return manifest_path