import os
import json
import time
import argparse
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from semanticscholar import SemanticScholar

# Project root: two levels up from src/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env for API key
load_dotenv(PROJECT_ROOT / ".env")


def download_papers(
    api_key: str = None,
    query: str = "natural language processing",
    output_dir: str = None,
    fields: list = None,
    max_papers: int = 5000,
):
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "src" / "data" / "raw")

    sch = SemanticScholar(api_key=api_key)

    if fields is None:
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "fieldsOfStudy",
        ]

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "papers.json")
    all_papers = []
    seen_ids = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_papers = json.load(f)
        seen_ids = {p["paperId"] for p in all_papers}
        print(f"Resuming: loaded {len(all_papers)} existing papers")

    if len(all_papers) >= max_papers:
        print(f"Already have {len(all_papers)} papers (>= {max_papers}), skipping.")
        return all_papers

    print(f"Downloading up to {max_papers} papers for query: '{query}'")

    try:
        results = sch.search_paper(
            query,
            limit=100,
            bulk=True,
            fields_of_study=["Computer Science"],
            fields=fields,
            min_citation_count=1,
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return all_papers

    for paper in tqdm(results, total=min(max_papers, results.total), desc="Fetching papers"):
        try:
            if (
                paper.abstract
                and paper.citationCount
                and paper.citationCount > 0
                and paper.paperId not in seen_ids
            ):
                all_papers.append({
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "authors": [a["name"] for a in (paper.authors or [])],
                    "citationCount": paper.citationCount,
                })
                seen_ids.add(paper.paperId)

                if len(all_papers) % 500 == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(all_papers, f, indent=2, ensure_ascii=False)
                    print(f"\n  Checkpoint: saved {len(all_papers)} papers")

                if len(all_papers) >= max_papers:
                    break
        except Exception as e:
            print(f"Error processing paper: {e}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_papers, f, indent=2, ensure_ascii=False)
            time.sleep(2)
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_papers)} papers to {output_path}")
    return all_papers


def _fetch_one_paper_citations(args):
    """Worker function for concurrent citation fetching."""
    pid, api_key, max_citations = args
    sch = SemanticScholar(api_key=api_key)
    contexts = []
    try:
        citations = sch.get_paper_citations(
            pid,
            fields=[
                "contexts", "intents",
                "citingPaper.paperId", "citingPaper.title",
                "citingPaper.abstract", "citingPaper.authors",
                "citingPaper.year",
            ],
        )

        cite_count = 0
        for citation in citations:
            if cite_count >= max_citations:
                break
            if citation.contexts:
                for ctx in citation.contexts:
                    if ctx and len(ctx) > 30:
                        citing = citation.paper
                        contexts.append({
                            "cited_paper_id": pid,
                            "citing_paper_id": citing.paperId,
                            "citing_paper_title": citing.title,
                            "citing_paper_abstract": citing.abstract,
                            "citing_paper_authors": [
                                a["name"] for a in (citing.authors or [])
                            ],
                            "citing_paper_year": citing.year,
                            "citation_context": ctx,
                            "intents": citation.intents,
                        })
                        cite_count += 1
    except Exception:
        pass  # silently skip failed papers; progress tracked in main
    return pid, contexts


def download_citation_contexts(
    paper_ids: list,
    api_key: str = None,
    output_dir: str = None,
    max_citations_per_paper: int = 50,
    max_workers: int = 5,
):
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "src" / "data" / "raw")

    output_path = os.path.join(output_dir, "citation_contexts.json")

    all_contexts = []
    done_pids = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_contexts = json.load(f)
        done_pids = {c["cited_paper_id"] for c in all_contexts}
        print(f"Resuming: loaded {len(all_contexts)} contexts from {len(done_pids)} papers")

    remaining = [pid for pid in paper_ids if pid not in done_pids]
    print(f"Fetching citation contexts for {len(remaining)} papers "
          f"({len(done_pids)} already done) using {max_workers} workers...")

    # Prepare work items
    work_items = [(pid, api_key, max_citations_per_paper) for pid in remaining]

    checkpoint_interval = 100
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_paper_citations, item): item[0]
            for item in work_items
        }

        with tqdm(total=len(remaining), desc="Citation contexts") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pid, contexts = future.result()
                all_contexts.extend(contexts)
                done_pids.add(pid)
                completed += 1
                pbar.update(1)
                pbar.set_postfix(contexts=len(all_contexts))

                # Checkpoint periodically
                if completed % checkpoint_interval == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(all_contexts, f, indent=2, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_contexts, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_contexts)} citation contexts to {output_path}")
    return all_contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None,
                        help="Semantic Scholar API key (reads from .env if not set)")
    parser.add_argument("--query", type=str, default="natural language processing")
    parser.add_argument("--max_papers", type=int, default=5000)
    parser.add_argument("--max_citations_per_paper", type=int, default=100)
    parser.add_argument("--top_k_for_citations", type=int, default=2000)
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Concurrent threads for citation fetching (API key: 5-8, no key: 1-2)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Use API key from args, or .env, or None
    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        print(f"Using API key: {api_key[:8]}...")
    else:
        print("No API key — rate limits will be lower. Set SEMANTIC_SCHOLAR_API_KEY in .env")

    papers = download_papers(
        api_key=api_key, query=args.query,
        output_dir=args.output_dir, max_papers=args.max_papers,
    )

    top_papers = sorted(papers, key=lambda p: p["citationCount"] or 0, reverse=True)
    top_ids = [p["paperId"] for p in top_papers[:args.top_k_for_citations]]

    download_citation_contexts(
        paper_ids=top_ids, api_key=api_key, output_dir=args.output_dir,
        max_citations_per_paper=args.max_citations_per_paper,
        max_workers=args.max_workers,
    )
