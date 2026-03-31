import os
import json
import sqlite3
import argparse
from difflib import SequenceMatcher

def build_citation_db(papers_path: str = "src/data/raw/papers.json", db_path: str = "src/data/citation_db/citations.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with open(papers_path, "r") as f:
        papers = json.load(f)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            citation_key TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            paper_id TEXT,
            abstract TEXT
        )
    """)
    
    inserted = 0
    for paper in papers:
        if not paper.get("authors") or not paper.get("year"):
            continue
        
        first_author = paper["authors"][0].split()[-1]  # Last name
        year = paper["year"]
        
        # Build citation key: "Author et al., YYYY" or "Author, YYYY"
        if len(paper["authors"]) > 2:
            citation_key = f"{first_author} et al., {year}"
        elif len(paper["authors"]) == 2:
            second = paper["authors"][1].split()[-1]
            citation_key = f"{first_author} and {second}, {year}"
        else:
            citation_key = f"{first_author}, {year}"
        
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                (citation_key, paper["title"],
                 json.dumps(paper["authors"]), year,
                 paper["paperId"], paper.get("abstract", ""))
            )
            inserted += 1
        except Exception as e:
            continue
    
    conn.commit()
    conn.close()
    print(f"Inserted {inserted} papers into {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_path", default="src/data/raw/papers.json")
    parser.add_argument("--db_path", default="src/data/citation_db/citations.db")
    args = parser.parse_args()
    build_citation_db(args.papers_path, args.db_path)

