"""
Skill extraction utilities for the Resume-to-Job Skill Matcher.

This file uses a simple keyword dictionary approach.
It checks whether known skills appear in resume text or job text.
"""


SKILL_KEYWORDS = [
    # Programming and data
    "python",
    "sql",
    "java",
    "javascript",
    "c++",
    "c#",
    "r",
    "scala",

    # Data analysis and machine learning
    "data analysis",
    "data analytics",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "statistics",
    "predictive modeling",
    "classification",
    "regression",

    # Python libraries
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "pytorch",

    # BI and visualization
    "excel",
    "power bi",
    "tableau",
    "data visualization",
    "dashboard",
    "reporting",

    # Databases and cloud
    "database",
    "mysql",
    "postgresql",
    "sql server",
    "mongodb",
    "aws",
    "azure",
    "google cloud",
    "cloud",

    # Big data and tools
    "spark",
    "hadoop",
    "databricks",
    "git",
    "github",
    "docker",
    "linux",

    # Business and soft skills
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "critical thinking",
    "project management",
    "time management",
    "customer service",
    "training",
    "management",
    "recruitment",
    "sales",
    "marketing",
    "human resources",
]


def extract_skills(text, skill_list=None):
    """
    Extract known skills from a text string.

    Parameters
    ----------
    text : str
        Resume text or job description text.

    skill_list : list, optional
        List of skills to search for.
        If no list is provided, SKILL_KEYWORDS is used.

    Returns
    -------
    list
        Sorted list of matched skills.
    """
    if skill_list is None:
        skill_list = SKILL_KEYWORDS

    if text is None:
        return []

    text = str(text).lower()

    found_skills = []

    for skill in skill_list:
        skill_lower = skill.lower()

        if skill_lower in text:
            found_skills.append(skill_lower)

    return sorted(set(found_skills))