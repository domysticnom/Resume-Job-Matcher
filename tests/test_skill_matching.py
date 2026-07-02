from matcher.scoring.skill_extraction import extract_skills
from matcher.scoring.keyword_overlap import keyword_overlap_score


def test_extract_skills():
    text = "I have experience with Python, SQL, Excel, and Power BI."
    skills = extract_skills(text)

    assert "python" in skills
    assert "sql" in skills
    assert "excel" in skills
    assert "power bi" in skills


def test_keyword_overlap_score():
    resume_skills = ["python", "sql", "excel"]
    job_skills = ["python", "sql", "tableau"]

    result = keyword_overlap_score(resume_skills, job_skills)

    assert result["score"] == 0.667
    assert result["matched_skills"] == ["python", "sql"]
    assert result["missing_skills"] == ["tableau"]