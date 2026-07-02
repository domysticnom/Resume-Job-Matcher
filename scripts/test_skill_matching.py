from matcher.scoring.skill_extraction import extract_skills
from matcher.scoring.keyword_overlap import keyword_overlap_score


resume_text = """
I have experience with Python, SQL, Excel, Power BI, data analysis,
communication, and project management.
"""

job_text = """
We are looking for a data analyst with Python, SQL, Tableau,
data visualization, and communication skills.
"""

resume_skills = extract_skills(resume_text)
job_skills = extract_skills(job_text)

result = keyword_overlap_score(resume_skills, job_skills)

print("Resume skills:")
print(resume_skills)

print("\nJob skills:")
print(job_skills)

print("\nMatch result:")
print(result)