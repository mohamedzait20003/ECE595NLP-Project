import os
import re
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple

CITATION_PATTERN = re.compile(
    r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'
    r'(?:\s*,\s*\d{4}))\)'
    r'|'
    r'\[([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'
    r'(?:\s*,\s*\d{4}))\]'
)

CITATION_PATTERN_ALT = re.compile(
    r'([A-Z][a-z]+\s+(?:et\s+al\.?\s+)?\(\d{4}\))'
)

class Processor:
    def __init__(self, raw_data_dir: str = "src/data/raw", processed_data_dir: str = "src/data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        with open(os.path.join(self.raw_data_dir, "papers.json"), "r") as f:
            papers = json.load(f)

        with open(os.path.join(self.raw_data_dir, "citation_contexts.json"), "r") as f:
            contexts = json.load(f)

        return papers, contexts
    
    def build_paper_lookup(self, papers: List[Dict]) -> Dict[str, Dict]:
        return {p["paperId"]: p for p in papers}
    
    def extract_citation_from_context(self, context: str) -> List[Dict]:
        results = []
        
        for pattern in [CITATION_PATTERN, CITATION_PATTERN_ALT]:
            for match in pattern.finditer(context):
                citation_text = match.group(1) or match.group(2) if match.lastindex >= 2 else match.group(1)
                if not citation_text:
                    continue
                
                full_match = match.group(0)
                masked_sentence = context.replace(full_match, "[MASK]", 1)
                
                year_match = re.search(r'(\d{4})', citation_text)
                author_match = re.match(r'([A-Z][a-z]+)', citation_text)
                
                results.append({
                    "original_sentence": context,
                    "masked_sentence": masked_sentence,
                    "citation_string": citation_text,
                    "citation_full": full_match,
                    "author": author_match.group(1) if author_match else "",
                    "year": year_match.group(1) if year_match else ""
                })
        
        return results
    
    def process_all(self, min_context_length: int = 50, max_context_length: int = 500) -> pd.DataFrame:
        papers, contexts = self.load_raw_data()
        paper_lookup = self.build_paper_lookup(papers)
        
        examples = []
        
        for ctx in tqdm(contexts, desc="Processing citation contexts"):
            if not ctx.get("citing_paper_abstract"):
                continue
            
            sentence = ctx["citation_context"]
            
            if len(sentence) < min_context_length or len(sentence) > max_context_length:
                continue
            
            extractions = self.extract_citation_from_context(sentence)
            
            for ext in extractions:
                examples.append({
                    "masked_sentence": ext["masked_sentence"],
                    "original_sentence": ext["original_sentence"],
                    
                    "citation_string": ext["citation_string"],
                    "citation_full": ext["citation_full"],
                    "cited_paper_id": ctx["cited_paper_id"],
                    
                    "source_title": ctx["citing_paper_title"],
                    "source_abstract": ctx["citing_paper_abstract"],
                    
                    "citing_paper_id": ctx["citing_paper_id"],
                    "author": ext["author"],
                    "year": ext["year"]
                })
        
        df = pd.DataFrame(examples)
        df = df.drop_duplicates(subset=["masked_sentence", "citation_string"])
        
        print(f"Total examples after processing: {len(df)}")
        
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
