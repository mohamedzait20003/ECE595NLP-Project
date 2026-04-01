import sys
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.main.utils import Processor


def main():
    parser = argparse.ArgumentParser(description="Process raw Semantic Scholar data into training examples")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Directory containing papers.json and citation_contexts.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save train/val/test splits")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum citation context sentence length")
    parser.add_argument("--max_length", type=int, default=500,
                        help="Maximum citation context sentence length")
    args = parser.parse_args()

    processor = Processor(raw_data_dir=args.raw_dir, processed_data_dir=args.output_dir)
    df = processor.process_all(min_context_length=args.min_length, max_context_length=args.max_length)

    print("\n--- Summary ---")
    print(f"Total examples: {len(df)}")
    print(f"Unique citations: {df['citation_string'].nunique()}")
    print(f"Unique source papers: {df['citing_paper_id'].nunique()}")
    print(f"Output directory: {args.output_dir or 'src/data/processed (default)'}")


if __name__ == "__main__":
    main()
