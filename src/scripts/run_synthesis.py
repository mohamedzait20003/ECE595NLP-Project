import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.main.utils import Synthesizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio from processed citation data")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing train.json, val.json, test.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save .wav files and manifests")
    parser.add_argument("--rate", type=int, default=150,
                        help="Speaking speed in words per minute")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to synthesize")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "src" / "data" / "processed"
    output_dir = args.output_dir or str(PROJECT_ROOT / "src" / "data" / "audio")

    synthesizer = Synthesizer(rate=args.rate, output_dir=output_dir)

    for split in args.splits:
        split_path = data_dir / f"{split}.json"
        if not split_path.exists():
            print(f"Skipping {split}: {split_path} not found")
            continue
        print(f"\nSynthesizing {split} split...")
        synthesizer.synthesize_split(str(split_path), split_name=split)


if __name__ == "__main__":
    main()
