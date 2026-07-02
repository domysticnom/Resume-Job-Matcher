"""
Keyword-overlap scoring for resume-to-job matching.

This compares extracted resume skills with required job skills.
"""


def keyword_overlap_score(resume_skills, job_skills):
    """
    Calculate how many job skills are found in the resume.

    Formula:
        score = number of matched job skills / total number of job skills

    Parameters
    ----------
    resume_skills : list
        Skills extracted from the resume.

    job_skills : list
        Skills extracted from the job posting.

    Returns
    -------
    dict
        Dictionary containing score, matched skills, and missing skills.
    """
    resume_skills = set(resume_skills)
    job_skills = set(job_skills)

    if len(job_skills) == 0:
        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
        }

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills.difference(resume_skills)

    score = len(matched_skills) / len(job_skills)

    return {
        "score": round(score, 3),
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
    }