"""Precompute NER skills for the fixed job corpus, once, and cache them.

The recommend flow must not run NER over every posting on every query (that is
what made the app slow). Job postings are a fixed corpus, so we extract their
skills a single time here and store them in the ``skills`` column of
``data/processed/job_postings.csv``. At query time only the résumé is extracted
live; postings are looked up.

Run:  python scripts/precompute_skills.py
Re-run whenever the corpus or the extractor changes.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from matcher.recommend import CORPUS_CSV, load_job_corpus  # noqa: E402
from matcher.scoring.skill_extraction import extract_skills  # noqa: E402


def main() -> None:
    postings = load_job_corpus()
    print(f"Precomputing skills for {len(postings)} postings...")

    rows = []
    start = time.time()
    for i, p in enumerate(postings, 1):
        skills = extract_skills(p.text)
        rows.append({"id": p.id, "text": p.text, "title": p.title, "skills": ";".join(skills)})
        if i % 50 == 0:
            print(f"  {i}/{len(postings)}  ({time.time() - start:.0f}s)")

    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "title", "skills"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} postings with precomputed skills to {CORPUS_CSV} "
          f"in {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
