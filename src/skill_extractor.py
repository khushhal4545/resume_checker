"""
src/skill_extractor.py
Extracts skills from text using a predefined skill taxonomy.
Also maps skills to job roles.
"""

from src.preprocessing import clean_text

# ── Master skill taxonomy ─────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    # Programming Languages
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "kotlin": ["kotlin"],
    "r": [" r "],          # space-padded to avoid false positives
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "go": [" go "],
    "rust": ["rust"],
    "swift": ["swift"],
    "php": ["php"],
    "scala": ["scala"],
    "bash": ["bash", "shell scripting"],

    # ML / AI
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "statistics": ["statistics", "statistical analysis"],
    "feature engineering": ["feature engineering"],
    "mlops": ["mlops"],
    "mlflow": ["mlflow"],

    # Data
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "pl/sql"],
    "spark": ["apache spark", "pyspark", "spark"],
    "hadoop": ["hadoop"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "ms excel"],
    "data visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
    "snowflake": ["snowflake"],
    "etl": ["etl"],

    # Web
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "nodejs": ["node.js", "nodejs"],
    "django": ["django"],
    "flask": ["flask"],
    "rest api": ["rest api", "restful", "rest"],
    "graphql": ["graphql"],
    "next.js": ["next.js", "nextjs"],
    "tailwind": ["tailwind", "tailwindcss"],
    "bootstrap": ["bootstrap"],
    "redux": ["redux"],

    # DevOps / Cloud
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "ci/cd": ["ci/cd", "jenkins", "github actions", "gitlab ci"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "git": ["git", "github", "gitlab", "version control"],
    "linux": ["linux", "ubuntu", "centos"],

    # Android
    "android sdk": ["android sdk", "android development"],
    "jetpack compose": ["jetpack compose"],
    "firebase": ["firebase"],
    "mvvm": ["mvvm"],
    "retrofit": ["retrofit"],

    # Security
    "ethical hacking": ["ethical hacking", "penetration testing", "pentest"],
    "network security": ["network security", "cybersecurity"],
    "owasp": ["owasp"],
    "siem": ["siem"],
    "vulnerability assessment": ["vulnerability assessment"],
    "cryptography": ["cryptography", "encryption"],

    # Soft / Methodology
    "agile": ["agile", "scrum", "kanban"],
    "project management": ["project management", "pmp"],
    "communication": ["communication", "presentation"],
    "leadership": ["leadership", "team lead"],
}

# Roles → required skills mapping for gap analysis
ROLE_REQUIRED_SKILLS = {
    "Data Scientist": [
        "python","machine learning","deep learning","statistics","sql",
        "pandas","numpy","scikit-learn","data visualization","feature engineering",
    ],
    "Web Developer": [
        "html","css","javascript","react","nodejs","rest api","git","sql","docker",
    ],
    "DevOps Engineer": [
        "docker","kubernetes","aws","ci/cd","git","linux","terraform","ansible",
    ],
    "Android Developer": [
        "java","kotlin","android sdk","rest api","firebase","git","mvvm",
    ],
    "Machine Learning Engineer": [
        "python","machine learning","deep learning","mlops","docker","aws",
        "scikit-learn","tensorflow","git","sql",
    ],
    "Data Analyst": [
        "sql","excel","tableau","power bi","python","data visualization","statistics",
    ],
    "UI/UX Designer": [
        "figma","prototyping","wireframing","user research","css","html",
    ],
    "Cybersecurity Analyst": [
        "network security","ethical hacking","python","linux","owasp",
        "vulnerability assessment","siem","cryptography",
    ],
}


def extract_skills(text: str) -> list:
    """
    Returns a de-duplicated list of canonical skill names found in `text`.
    Matching is case-insensitive and uses alias expansion.
    """
    text_lower = " " + text.lower() + " "   # pad for boundary matching
    found = set()

    for canonical, aliases in SKILL_TAXONOMY.items():
        for alias in aliases:
            # Use word boundaries for short tokens to reduce false positives
            if len(alias) <= 3:
                pattern = f" {alias} "
            else:
                pattern = alias
            if pattern in text_lower:
                found.add(canonical)
                break

    return sorted(found)


def get_required_skills(role: str) -> list:
    """Return the required skill list for a given job role."""
    return ROLE_REQUIRED_SKILLS.get(role, [])


def get_all_roles() -> list:
    return list(ROLE_REQUIRED_SKILLS.keys())


if __name__ == "__main__":
    sample = "Python developer with expertise in Machine Learning, TensorFlow, SQL, Docker and AWS."
    print("Extracted skills:", extract_skills(sample))
