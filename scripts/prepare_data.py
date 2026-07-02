from pathlib import Path
import ast
import re

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

JOBS_FILE = RAW_DIR / "all_job_post.csv"
RESUMES_FILE = RAW_DIR / "resume_data.csv"

CLEANED_JOBS_FILE = PROCESSED_DIR / "cleaned_jobs.csv"
CLEANED_RESUMES_FILE = PROCESSED_DIR / "cleaned_resumes.csv"


def clean_text(value):
    """
    Basic text cleaning:
    - handles missing values
    - converts text to lowercase
    - removes strange characters
    - keeps letters, numbers, +, #, and .
    - removes extra spaces
    """
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_skill_list(value):
    """
    The job_skill_set column looks like a Python list stored as text:
    ['python', 'sql', 'communication']

    This function converts it into clean comma-separated text.
    """
    if pd.isna(value):
        return ""

    try:
        parsed_value = ast.literal_eval(str(value))

        if isinstance(parsed_value, list):
            return ", ".join([clean_text(skill) for skill in parsed_value])

    except Exception:
        pass

    return clean_text(value)


def load_and_clean_jobs():
    jobs = pd.read_csv(JOBS_FILE)

    jobs = jobs.copy()

    jobs["job_id"] = jobs["job_id"].astype(str)
    jobs["category"] = jobs["category"].fillna("")
    jobs["job_title"] = jobs["job_title"].fillna("")
    jobs["job_description"] = jobs["job_description"].fillna("")
    jobs["job_skill_set"] = jobs["job_skill_set"].fillna("")

    jobs["clean_category"] = jobs["category"].apply(clean_text)
    jobs["clean_job_title"] = jobs["job_title"].apply(clean_text)
    jobs["clean_job_description"] = jobs["job_description"].apply(clean_text)
    jobs["clean_job_skills"] = jobs["job_skill_set"].apply(clean_skill_list)

    cleaned_jobs = jobs[
        [
            "job_id",
            "category",
            "job_title",
            "job_description",
            "job_skill_set",
            "clean_category",
            "clean_job_title",
            "clean_job_description",
            "clean_job_skills",
        ]
    ]

    return cleaned_jobs


def load_and_clean_resumes():
    resumes = pd.read_csv(RESUMES_FILE)

    resumes = resumes.copy()

    # This file has a strange hidden character in the job position column name.
    # We rename it to something easier to use.
    resumes = resumes.rename(columns={"\ufeffjob_position_name": "job_position_name"})

    # Create a resume_id because the dataset does not appear to have one.
    resumes["resume_id"] = range(1, len(resumes) + 1)

    # Fill missing values for important columns.
    important_columns = [
        "career_objective",
        "skills",
        "degree_names",
        "major_field_of_studies",
        "professional_company_names",
        "related_skils_in_job",
        "positions",
        "responsibilities",
        "languages",
        "certification_skills",
        "job_position_name",
        "skills_required",
        "matched_score",
    ]

    for col in important_columns:
        if col in resumes.columns:
            resumes[col] = resumes[col].fillna("")

    # Combine resume-side information into one text field.
    resumes["resume_text"] = (
        resumes["career_objective"].astype(str)
        + " "
        + resumes["skills"].astype(str)
        + " "
        + resumes["degree_names"].astype(str)
        + " "
        + resumes["major_field_of_studies"].astype(str)
        + " "
        + resumes["professional_company_names"].astype(str)
        + " "
        + resumes["related_skils_in_job"].astype(str)
        + " "
        + resumes["positions"].astype(str)
        + " "
        + resumes["responsibilities"].astype(str)
        + " "
        + resumes["languages"].astype(str)
        + " "
        + resumes["certification_skills"].astype(str)
    )

    resumes["clean_resume_text"] = resumes["resume_text"].apply(clean_text)
    resumes["clean_job_position_name"] = resumes["job_position_name"].apply(clean_text)
    resumes["clean_skills"] = resumes["skills"].apply(clean_text)
    resumes["clean_skills_required"] = resumes["skills_required"].apply(clean_text)

    cleaned_resumes = resumes[
        [
            "resume_id",
            "job_position_name",
            "skills",
            "skills_required",
            "matched_score",
            "resume_text",
            "clean_resume_text",
            "clean_job_position_name",
            "clean_skills",
            "clean_skills_required",
        ]
    ]

    return cleaned_resumes


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning job data...")
    cleaned_jobs = load_and_clean_jobs()
    cleaned_jobs.to_csv(CLEANED_JOBS_FILE, index=False)
    print(f"Saved cleaned jobs to: {CLEANED_JOBS_FILE}")
    print(f"Cleaned jobs shape: {cleaned_jobs.shape}")

    print()

    print("Loading and cleaning resume data...")
    cleaned_resumes = load_and_clean_resumes()
    cleaned_resumes.to_csv(CLEANED_RESUMES_FILE, index=False)
    print(f"Saved cleaned resumes to: {CLEANED_RESUMES_FILE}")
    print(f"Cleaned resumes shape: {cleaned_resumes.shape}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()