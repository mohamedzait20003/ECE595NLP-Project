import json
import torch
import librosa
from tqdm import tqdm
from pathlib import Path
from transformers import BartTokenizer, WhisperProcessor

from src.main.model.main_model import MainModel
from src.main.evaluation.metrics import CitationMetrics


def load_model(checkpoint_path: str, device: torch.device) -> MainModel:
    """Load a MainModel from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MainModel(
        whispher_model="openai/whisper-small",
        bart_model="facebook/bart-base",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", None)
    best_reward = ckpt.get("best_reward", None)
    info = f"step={step}"
    if val_loss is not None:
        info += f", val_loss={val_loss:.4f}"
    if best_reward is not None:
        info += f", best_reward={best_reward:.4f}"
    print(f"  Loaded checkpoint: {checkpoint_path} ({info})")
    return model


def generate_predictions(
    model: MainModel,
    test_entries: list[dict],
    tokenizer: BartTokenizer,
    processor: WhisperProcessor,
    device: torch.device,
    max_length: int = 64,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> tuple[list[str], list[str]]:
    """Run generation on test entries, return (generated, references)."""
    generated = []
    references = []

    for entry in tqdm(test_entries, desc="Generating"):
        try:
            waveform, _ = librosa.load(entry["audio_path"], sr=16000)
        except Exception as e:
            print(f"  Skipping {entry['audio_path']}: {e}")
            continue

        audio_features = processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        ctx = f"{entry['source_title']} </s> {entry['source_abstract']}"
        enc = tokenizer(ctx, return_tensors="pt", max_length=512, truncation=True)
        text_ids = enc["input_ids"].to(device)
        text_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate(
                audio_features=audio_features,
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        generated.append(pred)
        references.append(entry["citation_string"])

    return generated, references


def evaluate_checkpoint(
    checkpoint_path: str,
    test_manifest_path: str,
    device: torch.device,
    max_samples: int = 0,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> dict:
    """Evaluate a single checkpoint on the test set.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        test_manifest_path: Path to test_manifest.json.
        device: torch device.
        max_samples: Limit test samples (0 = use all).
        do_sample: Use sampling instead of greedy decoding.
        temperature: Sampling temperature (only used if do_sample=True).

    Returns:
        Dict with 'averages', 'per_sample', 'predictions'.
    """
    model = load_model(checkpoint_path, device)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    with open(test_manifest_path, "r") as f:
        test_entries = json.load(f)

    if max_samples > 0:
        test_entries = test_entries[:max_samples]

    print(f"  Evaluating on {len(test_entries)} test samples...")

    generated, references = generate_predictions(
        model, test_entries, tokenizer, processor, device,
        do_sample=do_sample, temperature=temperature,
    )

    metrics = CitationMetrics()
    results = metrics(generated, references)

    results["predictions"] = [
        {"generated": g, "reference": r}
        for g, r in zip(generated, references)
    ]

    return results


def compare_checkpoints(
    checkpoint_paths: dict[str, str],
    test_manifest_path: str,
    device: torch.device,
    max_samples: int = 0,
    do_sample: bool = False,
    temperature: float = 0.7,
    output_path: str = None,
) -> dict[str, dict]:
    """Evaluate and compare multiple checkpoints.

    Args:
        checkpoint_paths: Dict mapping name → checkpoint path.
        test_manifest_path: Path to test_manifest.json.
        device: torch device.
        max_samples: Limit test samples (0 = use all).
        do_sample: Use sampling instead of greedy decoding.
        temperature: Sampling temperature (only used if do_sample=True).
        output_path: Optional path to save JSON results.

    Returns:
        Dict mapping name → evaluation results.
    """
    all_results = {}

    for name, ckpt_path in checkpoint_paths.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        results = evaluate_checkpoint(
            ckpt_path, test_manifest_path, device, max_samples,
            do_sample=do_sample, temperature=temperature,
        )
        all_results[name] = results

        print(f"\n  Results for {name}:")
        for metric, value in results["averages"].items():
            print(f"    {metric:20s}: {value:.4f}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")

    names = list(all_results.keys())
    header = f"{'Metric':20s}" + "".join(f"  {n:>14s}" for n in names)
    print(header)
    print("-" * len(header))

    metric_names = CitationMetrics.METRIC_NAMES
    for metric in metric_names:
        row = f"{metric:20s}"
        for name in names:
            val = all_results[name]["averages"][metric]
            row += f"  {val:>14.4f}"
        print(row)

    if output_path:
        # Save only averages and predictions (per_sample can be large)
        save_data = {}
        for name, res in all_results.items():
            save_data[name] = {
                "averages": res["averages"],
                "predictions": res["predictions"],
            }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results
