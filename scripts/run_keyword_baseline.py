from pathlib import Path

import pandas as pd

from matcher.scoring.skill_extraction import extract_skills
from matcher.scoring.keyword_overlap import keyword_overlap_score


PROCESSED_DIR = Path("data/processed")

CLEANED_JOBS_FILE = PROCESSED_DIR / "cleaned_jobs.csv"
CLEANED_RESUMES_FILE = PROCESSED_DIR / "cleaned_resumes.csv"
OUTPUT_FILE = PROCESSED_DIR / "resume_job_skill_matches.csv"


def main():
    print("Loading cleaned jobs and resumes...")

    jobs = pd.read_csv(CLEANED_JOBS_FILE)
    resumes = pd.read_csv(CLEANED_RESUMES_FILE)

    print(f"Jobs shape: {jobs.shape}")
    print(f"Resumes shape: {resumes.shape}")

    print("\nExtracting skills from jobs...")

    jobs["extracted_job_skills"] = jobs[
        "clean_job_description"
    ].fillna("").apply(extract_skills)

    jobs["extracted_job_skill_set"] = jobs[
        "clean_job_skills"
    ].fillna("").apply(extract_skills)

    # Combine skills found in job description and job_skill_set.
    jobs["all_job_skills"] = jobs.apply(
        lambda row: sorted(
            set(row["extracted_job_skills"])
            .union(set(row["extracted_job_skill_set"]))
        ),
        axis=1,
    )

    print("Extracting skills from resumes...")

    resumes["extracted_resume_skills"] = resumes[
        "clean_resume_text"
    ].fillna("").apply(extract_skills)

    print("\nCreating resume-job skill matches...")

    results = []

    # To keep this manageable for class-project baseline,
    # we match each resume against the first 200 jobs.
    # You can increase this later if needed.
    sample_jobs = jobs.head(200)

    for _, resume_row in resumes.iterrows():
        resume_id = resume_row["resume_id"]
        resume_position = resume_row.get("job_position_name", "")
        resume_text = resume_row.get("clean_resume_text", "")
        resume_skills = resume_row["extracted_resume_skills"]

        for _, job_row in sample_jobs.iterrows():
            job_id = job_row["job_id"]
            job_title = job_row["job_title"]
            job_category = job_row["category"]
            job_skills = job_row["all_job_skills"]

            match_result = keyword_overlap_score(
                resume_skills=resume_skills,
                job_skills=job_skills,
            )

            results.append(
                {
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "resume_position": resume_position,
                    "job_title": job_title,
                    "job_category": job_category,
                    "resume_skills": ", ".join(resume_skills),
                    "job_skills": ", ".join(job_skills),
                    "match_score": match_result["score"],
                    "matched_skills": ", ".join(match_result["matched_skills"]),
                    "missing_skills": ", ".join(match_result["missing_skills"]),
                }
            )

    results_df = pd.DataFrame(results)

    print(f"\nSaving results to: {OUTPUT_FILE}")
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Output shape: {results_df.shape}")

    print("\nTop 10 matches:")
    print(
        results_df.sort_values("match_score", ascending=False)
        .head(10)[
            [
                "resume_id",
                "job_id",
                "resume_position",
                "job_title",
                "match_score",
                "matched_skills",
                "missing_skills",
            ]
        ]
    )


if __name__ == "__main__":
    main()