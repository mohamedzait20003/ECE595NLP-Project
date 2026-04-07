import os
import json
import pyttsx3
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _synthesize_worker(args):
    """Worker function for multiprocessing TTS synthesis."""
    text, output_path, temp_path, rate, volume, sample_rate = args

    try:
        spoken_text = text.replace("[MASK]", "citation needed")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.save_to_file(spoken_text, temp_path)
        engine.runAndWait()
        engine.stop()
        del engine

        # Resample
        data, orig_sr = sf.read(temp_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if orig_sr != sample_rate:
            duration = len(data) / orig_sr
            target_len = int(duration * sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        max_val = np.abs(data).max()
        if max_val > 0:
            data = data / max_val
        sf.write(output_path, data, sample_rate)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return output_path, None
    except Exception as e:
        return None, str(e)


class Synthesizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        rate: int = 150,
        volume: float = 1.0,
        output_dir: str = "src/data/audio",
        num_workers: int = 0,
    ):
        self.sample_rate = sample_rate
        self.rate = rate
        self.volume = volume
        self.output_dir = output_dir or str(PROJECT_ROOT / "src" / "data" / "audio")
        # 0 = auto (use 75% of CPUs), negative = sequential
        if num_workers == 0:
            self.num_workers = max(1, int(cpu_count() * 0.75))
        else:
            self.num_workers = num_workers
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
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
                data,
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

        # Check which files already exist (resume support)
        existing = set()
        manifest_path = os.path.join(self.output_dir, f"{split_name}_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
            existing = {
                e["index"] for e in old_manifest
                if os.path.exists(e.get("audio_path", ""))
            }
            if existing:
                print(f"  Resuming: {len(existing)}/{len(examples)} already synthesized")

        # Build work items for unsynthesized examples
        work_items = []
        for i, example in enumerate(examples):
            if i in existing:
                continue
            filename = os.path.join(split_name, f"{split_name}_{i:06d}")
            output_path = os.path.join(self.output_dir, f"{filename}.wav")
            temp_path = output_path + f".tmp_{i}.wav"
            work_items.append((
                example["masked_sentence"],
                output_path,
                temp_path,
                self.rate,
                self.volume,
                self.sample_rate,
            ))

        if not work_items:
            print(f"  All {len(examples)} examples already synthesized.")
            return manifest_path

        print(f"  Synthesizing {len(work_items)} examples using {self.num_workers} workers...")

        # Parallel synthesis
        results = {}
        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_synthesize_worker, work_items),
                    total=len(work_items),
                    desc=f"Synthesizing {split_name}",
                ):
                    audio_path, _ = result
                    if audio_path:
                        results[audio_path] = True
        else:
            for item in tqdm(work_items, desc=f"Synthesizing {split_name}"):
                audio_path, _ = _synthesize_worker(item)
                if audio_path:
                    results[audio_path] = True

        # Build manifest from all files (existing + new)
        manifest = []
        for i, example in enumerate(examples):
            filename = os.path.join(split_name, f"{split_name}_{i:06d}")
            audio_path = os.path.join(self.output_dir, f"{filename}.wav")
            if os.path.exists(audio_path):
                manifest.append({
                    "index": i,
                    "audio_path": audio_path,
                    "masked_sentence": example["masked_sentence"],
                    "citation_string": example["citation_string"],
                    "source_title": example["source_title"],
                    "source_abstract": example["source_abstract"],
                })

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Synthesized {len(manifest)}/{len(examples)} examples -> {split_dir}")
        print(f"Manifest saved to {manifest_path}")

        return manifest_path
