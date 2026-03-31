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
            "citationCount", "references", "fieldsOfStudy"
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    all_papers = []

    print(f"Downloading up to {max_papers} papers for query: '{query}'")

    try:
        results = sch.search_paper(
            query,
            limit=100,
            fields_of_study=["Computer Science"],
            fields=fields
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return all_papers

    for paper in tqdm(results, total=min(max_papers, results.total), desc="Fetching papers"):
        try:
            if (paper.abstract
                and paper.references
                and len(paper.references) > 0):
                all_papers.append({
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "authors": [a["name"] for a in (paper.authors or [])],
                    "references": [
                        {"paperId": r.paperId, "title": r.title}
                        for r in (paper.references or [])
                    ],
                    "citationCount": paper.citationCount
                })
                if len(all_papers) >= max_papers:
                    break
        except Exception as e:
            print(f"Error processing paper: {e}")
            continue

    output_path = os.path.join(output_dir, "papers.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_papers)} papers to {output_path}")
    return all_papers


def download_citation_contexts(paper_ids: list, api_key: str = None, output_dir: str = None):
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "src" / "data" / "raw")

    sch = SemanticScholar(api_key=api_key)
    all_contexts = []
    
    print(f"Fetching citation contexts for {len(paper_ids)} papers...")
    
    for pid in tqdm(paper_ids):
        try:
            citations = sch.get_paper_citations(
                pid, 
                fields=["contexts", "intents", "citingPaper.paperId",
                         "citingPaper.title", "citingPaper.abstract",
                         "citingPaper.authors", "citingPaper.year"]
            )
            
            for citation in citations:
                if citation.contexts:
                    for ctx in citation.contexts:
                        if ctx and len(ctx) > 30:
                            citing = citation.citingPaper
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
            
            time.sleep(1.0 if api_key else 3.0)
            
        except Exception as e:
            print(f"Error for paper {pid}: {e}")
            time.sleep(5)
            continue
    
    output_path = os.path.join(output_dir, "citation_contexts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_contexts, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_contexts)} citation contexts to {output_path}")
    return all_contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None, help="Semantic Scholar API key (optional, increases rate limit)")
    parser.add_argument("--query", type=str, default="natural language processing")
    parser.add_argument("--max_papers", type=int, default=5000, help="Maximum number of papers to download")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    papers = download_papers(
        api_key=args.api_key, query=args.query,
        output_dir=args.output_dir, max_papers=args.max_papers
    )

    top_papers = sorted(papers, key=lambda p: p["citationCount"] or 0, reverse=True)
    top_ids = [p["paperId"] for p in top_papers[:2000]]
    
    download_citation_contexts(
        paper_ids=top_ids, api_key=args.api_key, output_dir=args.output_dir
    )