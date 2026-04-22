"""
src/recommender.py
Skill-gap analyser + resume improvement recommender.
"""

from src.skill_extractor import (
    extract_skills,
    get_required_skills,
    ROLE_REQUIRED_SKILLS,
    SKILL_TAXONOMY,
)


# ── Skill Gap Analysis ────────────────────────────────────────────────────────

def analyze_skill_gap(resume_text: str, job_description: str, role: str = None) -> dict:
    """
    Compares skills in resume vs. job description (and optional role requirements).

    Returns:
        {
            'resume_skills': [...],
            'jd_skills': [...],
            'required_skills': [...],   # from role taxonomy
            'matched_skills': [...],
            'missing_from_jd': [...],
            'missing_from_role': [...],
            'skill_coverage': float,    # % of JD skills covered
        }
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))
    required_skills = set(get_required_skills(role)) if role else set()

    matched = resume_skills & jd_skills
    missing_from_jd = jd_skills - resume_skills
    missing_from_role = required_skills - resume_skills

    coverage = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "required_skills": sorted(required_skills),
        "matched_skills": sorted(matched),
        "missing_from_jd": sorted(missing_from_jd),
        "missing_from_role": sorted(missing_from_role),
        "skill_coverage": round(coverage, 1),
    }


# ── Resume Improvement Tips ───────────────────────────────────────────────────

def generate_recommendations(gap_result: dict, ats_score: float, match_score: float) -> list:
    """
    Return a list of actionable recommendation strings based on
    gap analysis + score inputs.
    """
    tips = []

    # ATS-score based tips
    if ats_score < 50:
        tips.append("📄 Your ATS score is low. Add more industry-specific keywords to your resume.")
        tips.append("📐 Use a simple, clean resume format — avoid tables and graphics that ATS cannot parse.")
    elif ats_score < 70:
        tips.append("🔑 Sprinkle more role-specific keywords throughout your experience bullet points.")
        tips.append("📊 Quantify your achievements (e.g., 'Improved model accuracy by 12%').")

    # Match score based tips
    if match_score < 40:
        tips.append("⚠️  Your resume matches poorly with this job description. Tailor it specifically for this role.")
        tips.append("🎯 Mirror the language and terminology used in the job description.")
    elif match_score < 65:
        tips.append("✍️  Rewrite your summary/objective to closely align with this job's requirements.")

    # Missing skill tips
    missing = gap_result.get("missing_from_jd", [])
    if missing:
        skills_str = ", ".join(missing[:6])
        tips.append(f"🛠️  Add these missing skills (if you have them): **{skills_str}**")
        tips.append("📚 Consider online courses (Coursera, Udemy, LinkedIn Learning) to upskill in missing areas.")

    # Coverage tips
    coverage = gap_result.get("skill_coverage", 0)
    if coverage < 50:
        tips.append("📉 Skill coverage against this JD is below 50%. Revamp the skills section entirely.")
    elif coverage < 75:
        tips.append("📈 Skill coverage is moderate. Focus on adding the top required skills from the JD.")

    # General best practices
    tips += [
        "✅ Use action verbs (built, designed, led, optimised, reduced, increased).",
        "📋 Keep resume to 1–2 pages; prioritise recent and relevant experience.",
        "🔗 Add GitHub / LinkedIn / Portfolio links if not present.",
        "💬 Include a strong 2–3 line professional summary at the top.",
    ]

    return tips


def keyword_suggestions(job_description: str, resume_text: str, top_n: int = 10) -> list:
    """
    Suggest top keywords from JD that are not yet in the resume.
    Uses simple word frequency (no TF-IDF) for transparency.
    """
    from collections import Counter
    from src.preprocessing import clean_text
    import re

    jd_clean = clean_text(job_description)
    resume_clean = clean_text(resume_text)

    jd_words = Counter(jd_clean.split())
    resume_words = set(resume_clean.split())

    # Filter: not in resume, length > 3
    suggestions = [
        (w, freq) for w, freq in jd_words.most_common(50)
        if w not in resume_words and len(w) > 3
    ]

    return [w for w, _ in suggestions[:top_n]]


def role_based_tips(role: str) -> list:
    """Return role-specific improvement advice."""
    TIPS = {
        "Data Scientist": [
            "🧪 Include links to Kaggle notebooks or GitHub ML projects.",
            "📊 Mention specific datasets or competitions you've worked with.",
            "📝 List publications or technical blog posts if any.",
        ],
        "Web Developer": [
            "🌐 Add links to live projects or your portfolio website.",
            "⚡ Highlight performance improvements you made (e.g., load-time reduction).",
            "📱 Mention responsive design and mobile-first experience.",
        ],
        "DevOps Engineer": [
            "🔧 List specific infrastructure scale (e.g., 'managed 200-node K8s cluster').",
            "📈 Highlight uptime improvements and incident reduction metrics.",
            "🛡️ Include security compliance experience (SOC 2, ISO 27001).",
        ],
        "Android Developer": [
            "📱 Link to apps on Google Play Store.",
            "⭐ Mention app ratings and download counts if impressive.",
            "🔋 Highlight performance optimisation (battery, memory).",
        ],
        "Machine Learning Engineer": [
            "🚀 Highlight model deployment experience (latency, throughput).",
            "📦 Mention MLOps tools (MLflow, Airflow, SageMaker) used in production.",
            "🔢 Quantify model improvements (accuracy delta, F1 score).",
        ],
        "Data Analyst": [
            "📊 Showcase dashboards with specific business impact.",
            "💡 Describe a data-driven decision you enabled.",
            "🏢 Mention stakeholder communication and presentation skills.",
        ],
        "UI/UX Designer": [
            "🎨 Include Figma/Behance/Dribbble portfolio link.",
            "🔬 Describe user research methods and sample sizes.",
            "📐 Mention A/B test results or usability improvements.",
        ],
        "Cybersecurity Analyst": [
            "🏆 List certifications (CEH, CISSP, CompTIA Security+, OSCP).",
            "🔍 Describe specific vulnerabilities discovered and remediated.",
            "🛡️ Mention compliance frameworks (NIST, PCI-DSS, GDPR) handled.",
        ],
    }
    return TIPS.get(role, ["📌 Tailor your resume to highlight the most relevant experiences for this role."])


if __name__ == "__main__":
    resume = "Python machine learning tensorflow sql pandas scikit-learn git docker"
    jd = "We need a data scientist with Python, machine learning, NLP, tensorflow, spark, tableau, docker"
    gap = analyze_skill_gap(resume, jd, role="Data Scientist")
    print("Gap Analysis:", gap)
    tips = generate_recommendations(gap, ats_score=62, match_score=55)
    print("\nRecommendations:")
    for t in tips:
        print(" ", t)
