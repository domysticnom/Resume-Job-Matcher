from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DIR = Path("data/processed")

CLEANED_JOBS_FILE = PROCESSED_DIR / "cleaned_jobs.csv"
CLEANED_RESUMES_FILE = PROCESSED_DIR / "cleaned_resumes.csv"
OUTPUT_FILE = PROCESSED_DIR / "resume_job_tfidf_matches.csv"


def main():
    print("Loading cleaned jobs and resumes...")

    jobs = pd.read_csv(CLEANED_JOBS_FILE)
    resumes = pd.read_csv(CLEANED_RESUMES_FILE)

    print(f"Jobs shape: {jobs.shape}")
    print(f"Resumes shape: {resumes.shape}")

    # Use a smaller sample first so it runs quickly.
    # You can increase these later.
    sample_resumes = resumes.head(1000).copy()
    sample_jobs = jobs.head(200).copy()

    print(f"\nUsing {len(sample_resumes)} resumes and {len(sample_jobs)} jobs for TF-IDF baseline.")

    sample_resumes["clean_resume_text"] = sample_resumes["clean_resume_text"].fillna("")
    sample_jobs["job_text_for_matching"] = (
        sample_jobs["clean_job_title"].fillna("")
        + " "
        + sample_jobs["clean_job_description"].fillna("")
        + " "
        + sample_jobs["clean_job_skills"].fillna("")
    )

    resume_texts = sample_resumes["clean_resume_text"].tolist()
    job_texts = sample_jobs["job_text_for_matching"].tolist()

    all_texts = resume_texts + job_texts

    print("\nBuilding TF-IDF vectors...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)

    resume_vectors = tfidf_matrix[: len(resume_texts)]
    job_vectors = tfidf_matrix[len(resume_texts) :]

    print("Calculating cosine similarity...")

    similarity_matrix = cosine_similarity(resume_vectors, job_vectors)

    print("Creating result table...")

    results = []

    for resume_index, resume_row in sample_resumes.reset_index(drop=True).iterrows():
        for job_index, job_row in sample_jobs.reset_index(drop=True).iterrows():
            similarity_score = similarity_matrix[resume_index, job_index]

            results.append(
                {
                    "resume_id": resume_row["resume_id"],
                    "job_id": job_row["job_id"],
                    "resume_position": resume_row.get("job_position_name", ""),
                    "job_title": job_row["job_title"],
                    "job_category": job_row["category"],
                    "tfidf_similarity_score": round(float(similarity_score), 4),
                }
            )

    results_df = pd.DataFrame(results)

    print(f"\nSaving TF-IDF results to: {OUTPUT_FILE}")
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Output shape: {results_df.shape}")

    print("\nTop 10 TF-IDF matches:")
    print(
        results_df.sort_values("tfidf_similarity_score", ascending=False)
        .head(10)
    )


if __name__ == "__main__":
    main()