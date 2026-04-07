import os
import re
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Author-year patterns: (Wu et al., 2023) or Wu et al. (2023)
AUTHOR_YEAR_PATTERN = re.compile(
    r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'
    r'(?:\s*,\s*\d{4}))\)'
    r'|'
    r'([A-Z][a-z]+\s+(?:et\s+al\.?\s+)?\(\d{4}\))'
)

# Numeric citation patterns: [9], [9,16], [9, 16, 18], [BBL+17]
NUMERIC_PATTERN = re.compile(
    r'\[(\d+(?:\s*[,;]\s*\d+)*)\]'   # [9], [9,16], [9; 16]
    r'|\[(\d+)\s*[-–]\s*(\d+)\]'      # [9-12]
)


def _make_citation_string(authors: list, year) -> str:
    """Build 'Author et al., YEAR' from author list and year."""
    if not authors:
        return ""
    first = authors[0].split()[-1]  # last name of first author
    suffix = " et al." if len(authors) > 1 else ""
    year_str = f", {year}" if year else ""
    return f"{first}{suffix}{year_str}"


class Processor:
    def __init__(self, raw_data_dir: str = None, processed_data_dir: str = None):
        raw_data_dir = raw_data_dir or str(PROJECT_ROOT / "src" / "data" / "raw")
        processed_data_dir = processed_data_dir or str(PROJECT_ROOT / "src" / "data" / "processed")
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        with open(os.path.join(self.raw_data_dir, "papers.json"), "r", encoding="utf-8") as f:
            papers = json.load(f)
        with open(os.path.join(self.raw_data_dir, "citation_contexts.json"), "r", encoding="utf-8") as f:
            contexts = json.load(f)
        return papers, contexts

    def build_paper_lookup(self, papers: List[Dict]) -> Dict[str, Dict]:
        return {p["paperId"]: p for p in papers}

    def _extract_author_year(self, context: str) -> List[Dict]:
        """Extract author-year citations like (Wu et al., 2023)."""
        results = []
        for match in AUTHOR_YEAR_PATTERN.finditer(context):
            citation_text = match.group(1) or match.group(2)
            if not citation_text:
                continue
            full_match = match.group(0)
            masked = context.replace(full_match, "[MASK]", 1)
            year_m = re.search(r'(\d{4})', citation_text)
            author_m = re.match(r'([A-Z][a-z]+)', citation_text)
            results.append({
                "masked_sentence": masked,
                "citation_string": citation_text,
                "citation_full": full_match,
                "author": author_m.group(1) if author_m else "",
                "year": year_m.group(1) if year_m else "",
                "source": "author_year",
            })
        return results

    def _extract_numeric(self, context: str, cited_paper: Dict) -> List[Dict]:
        """Extract numeric citations like [9] using cited paper metadata."""
        citation_string = _make_citation_string(
            cited_paper.get("authors", []),
            cited_paper.get("year"),
        )
        if not citation_string:
            return []

        results = []
        for match in NUMERIC_PATTERN.finditer(context):
            full_match = match.group(0)
            masked = context.replace(full_match, "[MASK]", 1)
            year_str = str(cited_paper.get("year", "")) if cited_paper.get("year") else ""
            authors = cited_paper.get("authors", [])
            author_str = authors[0].split()[-1] if authors else ""
            results.append({
                "masked_sentence": masked,
                "citation_string": citation_string,
                "citation_full": full_match,
                "author": author_str,
                "year": year_str,
                "source": "numeric",
            })
            break  # one extraction per context to avoid duplicates
        return results

    def process_all(self, min_context_length: int = 50, max_context_length: int = 500) -> pd.DataFrame:
        papers, contexts = self.load_raw_data()
        paper_lookup = self.build_paper_lookup(papers)

        examples = []
        stats = {"author_year": 0, "numeric": 0, "skipped": 0}

        for ctx in tqdm(contexts, desc="Processing citation contexts"):
            if not ctx.get("citing_paper_abstract"):
                stats["skipped"] += 1
                continue

            sentence = ctx["citation_context"]
            if len(sentence) < min_context_length or len(sentence) > max_context_length:
                stats["skipped"] += 1
                continue

            cited_paper = paper_lookup.get(ctx["cited_paper_id"], {})

            # Try author-year first
            extractions = self._extract_author_year(sentence)

            # Fall back to numeric if no author-year found
            if not extractions:
                extractions = self._extract_numeric(sentence, cited_paper)

            for ext in extractions:
                examples.append({
                    "masked_sentence": ext["masked_sentence"],
                    "citation_string": ext["citation_string"],
                    "citation_full": ext["citation_full"],
                    "cited_paper_id": ctx["cited_paper_id"],
                    "source_title": ctx["citing_paper_title"],
                    "source_abstract": ctx["citing_paper_abstract"],
                    "citing_paper_id": ctx["citing_paper_id"],
                    "author": ext["author"],
                    "year": ext["year"],
                    "extraction_source": ext["source"],
                })
                stats[ext["source"]] += 1

        df = pd.DataFrame(examples)
        df = df.drop_duplicates(subset=["masked_sentence", "citation_string"])

        print(f"\nExtraction stats:")
        print(f"  Author-year matches : {stats['author_year']}")
        print(f"  Numeric matches     : {stats['numeric']}")
        print(f"  Skipped             : {stats['skipped']}")
        print(f"  Total after dedup   : {len(df)}")

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_df = df[:int(0.8 * n)]
        val_df = df[int(0.8 * n):int(0.9 * n)]
        test_df = df[int(0.9 * n):]

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = os.path.join(self.processed_data_dir, f"{split_name}.json")
            split_df.to_json(path, orient="records", indent=2, force_ascii=False)
            print(f"  {split_name}: {len(split_df)} examples -> {path}")

        return df


if __name__ == "__main__":
    processor = Processor()
    df = processor.process_all()
