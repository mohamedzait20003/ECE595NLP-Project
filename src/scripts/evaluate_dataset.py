import os
import re
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUTHOR_YEAR_PATTERN = re.compile(
    r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'
    r'(?:\s*,\s*\d{4}))\)'
    r'|'
    r'([A-Z][a-z]+\s+(?:et\s+al\.?\s+)?\(\d{4}\))'
)

NUMERIC_PATTERN = re.compile(
    r'\[(\d+(?:\s*[,;]\s*\d+)*)\]'
    r'|\[(\d+)\s*[-–]\s*(\d+)\]'
)


def evaluate_raw(raw_dir: str):
    papers_path = os.path.join(raw_dir, "papers.json")
    contexts_path = os.path.join(raw_dir, "citation_contexts.json")

    if not os.path.exists(papers_path):
        print(f"  [MISSING] {papers_path}")
        return
    if not os.path.exists(contexts_path):
        print(f"  [MISSING] {contexts_path}")
        return

    with open(papers_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    print(f"Papers      : {len(papers)}")

    no_abstract = sum(1 for p in papers if not p.get("abstract"))
    no_year = sum(1 for p in papers if not p.get("year"))
    print(f"  No abstract : {no_abstract}")
    print(f"  No year     : {no_year}")

    print(f"\nContexts    : {len(contexts)}")

    unique_cited = len(set(c["cited_paper_id"] for c in contexts))
    print(f"  Unique papers cited: {unique_cited}")
    print(f"  Avg per paper      : {len(contexts)/max(unique_cited,1):.1f}")

    no_abstract = sum(1 for c in contexts if not c.get("citing_paper_abstract"))
    short = sum(1 for c in contexts if len(c.get("citation_context", "")) < 50)
    long_ = sum(1 for c in contexts if len(c.get("citation_context", "")) > 500)

    has_author_year = sum(1 for c in contexts if AUTHOR_YEAR_PATTERN.search(c.get("citation_context", "")))
    has_numeric = sum(1 for c in contexts if NUMERIC_PATTERN.search(c.get("citation_context", "")))
    has_any = sum(1 for c in contexts if (
        AUTHOR_YEAR_PATTERN.search(c.get("citation_context", "")) or
        NUMERIC_PATTERN.search(c.get("citation_context", ""))
    ))

    print(f"  No abstract         : {no_abstract} ({100*no_abstract/max(len(contexts),1):.1f}%)")
    print(f"  Too short (<50)     : {short} ({100*short/max(len(contexts),1):.1f}%)")
    print(f"  Too long  (>500)    : {long_} ({100*long_/max(len(contexts),1):.1f}%)")
    print(f"  Author-year pattern : {has_author_year} ({100*has_author_year/max(len(contexts),1):.1f}%)")
    print(f"  Numeric pattern     : {has_numeric} ({100*has_numeric/max(len(contexts),1):.1f}%)")
    print(f"  Either pattern      : {has_any} ({100*has_any/max(len(contexts),1):.1f}%)")

    usable = sum(
        1 for c in contexts
        if c.get("citing_paper_abstract")
        and 50 <= len(c.get("citation_context", "")) <= 500
        and (
            AUTHOR_YEAR_PATTERN.search(c.get("citation_context", "")) or
            NUMERIC_PATTERN.search(c.get("citation_context", ""))
        )
    )
    print(f"\n  Est. training examples: ~{usable}")


def evaluate_counts(processed_dir: str, audio_dir: str):
    """Quick sample counts — matches pretrain.py startup output exactly."""
    found_audio = False

    for split in ["train", "val", "test"]:
        manifest_path = os.path.join(audio_dir, f"{split}_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            total = len(entries)
            valid = sum(1 for e in entries if os.path.exists(e.get("audio_path", "")))
            skipped = total - valid

            label = f"{split.capitalize():>5s} samples"
            print(f"  {label}: {valid}")

            if skipped > 0:
                print(f"  [Dataset] Skipped {skipped} entries with missing audio files.")

            found_audio = True

    if found_audio:
        return

    found_processed = False
    for split in ["train", "val", "test"]:
        path = os.path.join(processed_dir, f"{split}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            print(f"  {split.capitalize():>5s} samples: {count}")
            found_processed = True

    if not found_processed:
        print("No processed or audio data found. Run run_processor.py first.")


def evaluate_processed(processed_dir: str):
    """Validate processed splits against training requirements."""
    REQUIRED_FIELDS = [
        "masked_sentence", "citation_string", "citation_full",
        "cited_paper_id", "source_title", "source_abstract",
        "citing_paper_id", "author", "year", "extraction_source",
    ]
    YEAR_RE = re.compile(r'(?:19|20)\d{2}')

    for split in ["train", "val", "test"]:
        path = os.path.join(processed_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        total = len(examples)
        issues = {
            "missing_fields": 0,
            "no_mask_token": 0,
            "sentence_too_short": 0,
            "sentence_too_long": 0,
            "no_author": 0,
            "no_year": 0,
            "citation_too_long": 0,
            "empty_abstract": 0,
            "empty_title": 0,
        }
        source_counts = {"author_year": 0, "numeric": 0, "other": 0}
        valid = 0

        for ex in examples:
            # Check required fields
            if any(ex.get(f) is None for f in REQUIRED_FIELDS):
                issues["missing_fields"] += 1
                continue

            ok = True

            # [MASK] token
            if "[MASK]" not in ex["masked_sentence"]:
                issues["no_mask_token"] += 1
                ok = False

            # Sentence length
            slen = len(ex["masked_sentence"])
            if slen < 50:
                issues["sentence_too_short"] += 1
                ok = False
            elif slen > 500:
                issues["sentence_too_long"] += 1
                ok = False

            # Citation string: capital author
            cs = ex["citation_string"]
            if not cs or not cs[0].isupper():
                issues["no_author"] += 1
                ok = False

            # Citation string: 4-digit year
            if not YEAR_RE.search(cs):
                issues["no_year"] += 1
                ok = False

            # Citation string length (proxy for tokenizer limit)
            if len(cs) > 64:
                issues["citation_too_long"] += 1
                ok = False

            # Non-empty abstract
            if not ex["source_abstract"].strip():
                issues["empty_abstract"] += 1
                ok = False

            # Non-empty title
            if not ex["source_title"].strip():
                issues["empty_title"] += 1
                ok = False

            if ok:
                valid += 1

            src = ex.get("extraction_source", "other")
            if src in source_counts:
                source_counts[src] += 1
            else:
                source_counts["other"] += 1

        pct = lambda n: f"{100*n/max(total,1):.1f}%"
        print(f"\n{split.capitalize()} split  ({total} examples)")
        print(f"  Valid (all checks pass) : {valid} ({pct(valid)})")
        print(f"  --- Issues ---")
        for k, v in issues.items():
            if v > 0:
                print(f"  {k:<22}: {v} ({pct(v)})")
        print(f"  --- Source distribution ---")
        print(f"  author_year : {source_counts['author_year']} ({pct(source_counts['author_year'])})")
        print(f"  numeric     : {source_counts['numeric']} ({pct(source_counts['numeric'])})")
        if source_counts["other"] > 0:
            print(f"  other       : {source_counts['other']} ({pct(source_counts['other'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dataset before training")
    parser.add_argument("--stage", type=str, default="counts",
                        choices=["counts", "raw", "processed", "all"],
                        help="Which stage to evaluate (default: counts)")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    args = parser.parse_args()

    raw_dir = args.raw_dir or str(PROJECT_ROOT / "src" / "data" / "raw")
    processed_dir = args.processed_dir or str(PROJECT_ROOT / "src" / "data" / "processed")
    audio_dir = args.audio_dir or str(PROJECT_ROOT / "src" / "data" / "audio")

    if args.stage in ("raw", "all"):
        evaluate_raw(raw_dir)

    if args.stage in ("processed", "all"):
        if args.stage == "all":
            print()
        print("=== Processed split validation ===")
        evaluate_processed(processed_dir)

    if args.stage in ("counts", "all"):
        if args.stage == "all":
            print()
        evaluate_counts(processed_dir, audio_dir)
