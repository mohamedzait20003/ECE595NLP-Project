import sys
import json
import torch
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.main.utils.dataset import CustomDataset
from src.main.utils.collator import Collator


def create_dummy_manifest(tmp_dir: Path, num_samples: int = 4) -> str:
    manifest = []
    audio_dir = tmp_dir / "audio"
    audio_dir.mkdir()

    for i in range(num_samples):
        waveform = np.zeros(16000, dtype=np.float32)
        audio_path = str(audio_dir / f"sample_{i}.wav")
        sf.write(audio_path, waveform, 16000)

        manifest.append({
            "index": i,
            "audio_path": audio_path,
            "masked_sentence": f"This is a test claim [MASK] about topic {i}.",
            "citation_string": f"Author{i} et al., 202{i}",
            "source_title": f"Test Paper Title {i}",
            "source_abstract": f"This is the abstract of test paper {i}. " * 5,
        })

    manifest_path = str(tmp_dir / "test_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest_path


def test_dataset_item_shapes():
    """Each item should have correct tensor shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = create_dummy_manifest(Path(tmp))
        dataset = CustomDataset(manifest_path=manifest_path)

        assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"

        item = dataset[0]
        assert "audio_features" in item
        assert "text_input_ids" in item
        assert "text_attention_mask" in item
        assert "labels" in item

        assert item["audio_features"].shape == (80, 3000), \
            f"audio_features shape: {item['audio_features'].shape}"

        assert item["text_input_ids"].dim() == 1
        assert item["text_attention_mask"].dim() == 1
        assert item["labels"].dim() == 1

        print(f"  audio_features:     {item['audio_features'].shape}")
        print(f"  text_input_ids:     {item['text_input_ids'].shape}")
        print(f"  text_attention_mask:{item['text_attention_mask'].shape}")
        print(f"  labels:             {item['labels'].shape}")

    print("PASSED: test_dataset_item_shapes")


def test_collator_batch_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = create_dummy_manifest(Path(tmp), num_samples=4)
        dataset = CustomDataset(manifest_path=manifest_path)
        collator = Collator()
        loader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(loader))

        assert batch["audio_features"].shape == (4, 80, 3000), \
            f"audio batch shape: {batch['audio_features'].shape}"
        assert batch["text_input_ids"].dim() == 2
        assert batch["text_attention_mask"].dim() == 2
        assert batch["labels"].dim() == 2

        assert batch["text_input_ids"].shape == batch["text_attention_mask"].shape
        assert (batch["labels"] == -100).any() or batch["labels"].shape[1] > 0

        print(f"  audio_features:      {batch['audio_features'].shape}")
        print(f"  text_input_ids:      {batch['text_input_ids'].shape}")
        print(f"  text_attention_mask: {batch['text_attention_mask'].shape}")
        print(f"  labels:              {batch['labels'].shape}")

    print("PASSED: test_collator_batch_shapes")


def test_dataloader_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = create_dummy_manifest(Path(tmp), num_samples=8)
        dataset = CustomDataset(manifest_path=manifest_path)
        collator = Collator()
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

        batches = list(loader)
        assert len(batches) == 4, f"Expected 4 batches, got {len(batches)}"

    print("PASSED: test_dataloader_iteration")


if __name__ == "__main__":
    print("Running dataset tests...\n")
    test_dataset_item_shapes()
    test_collator_batch_shapes()
    test_dataloader_iteration()
    print("\nAll tests passed.")
