import os
import json
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from semanticscholar import SemanticScholar

# Project root: two levels up from src/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def download_papers(api_key: str = None, query: str = "natural language processing", output_dir: str = None, fields: list = None, max_papers: int = 5000):
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "src" / "data" / "raw")

    sch = SemanticScholar(api_key=api_key)
    
    if fields is None:
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "fieldsOfStudy"
        ]

    os.makedirs(output_dir, exist_ok=True)

    # Resume: load existing papers and track seen IDs
    output_path = os.path.join(output_dir, "papers.json")
    all_papers = []
    seen_ids = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_papers = json.load(f)
        seen_ids = {p["paperId"] for p in all_papers}
        print(f"Resuming: loaded {len(all_papers)} existing papers")

    if len(all_papers) >= max_papers:
        print(f"Already have {len(all_papers)} papers (>= {max_papers}), skipping download.")
        return all_papers

    print(f"Downloading up to {max_papers} papers for query: '{query}'")

    try:
        results = sch.search_paper(
            query,
            limit=100,
            bulk=True,
            fields_of_study=["Computer Science"],
            fields=fields,
            min_citation_count=1
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return all_papers

    count = 0
    for paper in tqdm(results, total=min(max_papers, results.total), desc="Fetching papers"):
        try:
            count += 1
            if paper.abstract and paper.citationCount and paper.citationCount > 0 and paper.paperId not in seen_ids:
                all_papers.append({
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "authors": [a["name"] for a in (paper.authors or [])],
                    "citationCount": paper.citationCount
                })
                seen_ids.add(paper.paperId)

                # Checkpoint every 500 papers
                if len(all_papers) % 500 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(all_papers, f, indent=2, ensure_ascii=False)
                    print(f"\n  Checkpoint: saved {len(all_papers)} papers")

                if len(all_papers) >= max_papers:
                    break
        except Exception as e:
            print(f"Error processing paper: {e}")
            # Save progress before potentially crashing
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_papers, f, indent=2, ensure_ascii=False)
            time.sleep(5)
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_papers)} papers to {output_path}")
    return all_papers


def download_citation_contexts(paper_ids: list, api_key: str = None, output_dir: str = None, max_citations_per_paper: int = 50):
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "src" / "data" / "raw")

    sch = SemanticScholar(api_key=api_key)

    output_path = os.path.join(output_dir, "citation_contexts.json")

    all_contexts = []
    done_pids = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_contexts = json.load(f)
            
        done_pids = {c["cited_paper_id"] for c in all_contexts}
        print(f"Resuming: loaded {len(all_contexts)} existing contexts from {len(done_pids)} papers")

    remaining = [pid for pid in paper_ids if pid not in done_pids]
    print(f"Fetching citation contexts for {len(remaining)} papers ({len(done_pids)} already done)...")

    for pid in tqdm(remaining):
        try:
            citations = sch.get_paper_citations(
                pid, 
                fields=["contexts", "intents", "citingPaper.paperId",
                         "citingPaper.title", "citingPaper.abstract",
                         "citingPaper.authors", "citingPaper.year"]
            )
            
            cite_count = 0
            for citation in citations:
                if cite_count >= max_citations_per_paper:
                    break
                
                if citation.contexts:
                    for ctx in citation.contexts:
                        if ctx and len(ctx) > 30:
                            citing = citation.paper
                            all_contexts.append({
                                "cited_paper_id": pid,
                                "citing_paper_id": citing.paperId,
                                "citing_paper_title": citing.title,
                                "citing_paper_abstract": citing.abstract,
                                "citing_paper_authors": [
                                    a["name"] for a in (citing.authors or [])
                                ],
                                "citing_paper_year": citing.year,
                                "citation_context": ctx,
                                "intents": citation.intents
                            })
                            cite_count += 1

            # Checkpoint after each paper
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_contexts, f, indent=2, ensure_ascii=False)

            time.sleep(1.0 if api_key else 3.0)

        except Exception as e:
            print(f"Error for paper {pid}: {e}")
            time.sleep(5)
            continue
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_contexts, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_contexts)} citation contexts to {output_path}")
    return all_contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None, help="Semantic Scholar API key (optional, increases rate limit)")
    parser.add_argument("--query", type=str, default="natural language processing")
    parser.add_argument("--max_papers", type=int, default=5000, help="Maximum number of papers to download")
    parser.add_argument("--max_citations_per_paper", type=int, default=100, help="Max citation contexts per paper")
    parser.add_argument("--top_k_for_citations", type=int, default=2000, help="Fetch citations for the top-K most cited papers")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    papers = download_papers(
        api_key=args.api_key, query=args.query,
        output_dir=args.output_dir, max_papers=args.max_papers
    )

    top_papers = sorted(papers, key=lambda p: p["citationCount"] or 0, reverse=True)
    top_ids = [p["paperId"] for p in top_papers[:args.top_k_for_citations]]

    download_citation_contexts(
        paper_ids=top_ids, api_key=args.api_key, output_dir=args.output_dir,
        max_citations_per_paper=args.max_citations_per_paper
    )