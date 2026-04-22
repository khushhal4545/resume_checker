"""
src/job_matcher.py
Job-Resume matching using TF-IDF + Cosine Similarity.
Also supports ranking multiple resumes against a single JD.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text


def _build_vectorizer(texts: list) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a corpus."""
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    vec.fit(texts)
    return vec


def compute_match_score(resume_text: str, job_description: str) -> float:
    """
    Returns cosine similarity (0–100 %) between resume and job description.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    corpus = [cleaned_resume, cleaned_jd]
    vec = _build_vectorizer(corpus)
    vectors = vec.transform(corpus)

    sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(sim) * 100, 2)


def rank_resumes(resumes: list, job_description: str) -> list:
    """
    Rank a list of resume texts against a job description.

    Args:
        resumes: list of (label, text) tuples  e.g. [("Resume 1", "..."), ...]
        job_description: str

    Returns:
        Sorted list of dicts: [{'rank', 'label', 'score'}, ...]
    """
    if not resumes:
        return []

    cleaned_jd = clean_text(job_description)
    cleaned_resumes = [(label, clean_text(text)) for label, text in resumes]

    corpus = [cleaned_jd] + [r[1] for r in cleaned_resumes]
    vec = _build_vectorizer(corpus)
    vectors = vec.transform(corpus)

    jd_vec = vectors[0]
    results = []
    for i, (label, _) in enumerate(cleaned_resumes, start=1):
        sim = cosine_similarity(jd_vec, vectors[i])[0][0]
        results.append({
            "label": label,
            "score": round(float(sim) * 100, 2),
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def match_score_interpretation(score: float) -> dict:
    """Colour and label for a match-score percentage."""
    if score >= 75:
        return {"label": "Excellent Match", "color": "#22c55e"}
    elif score >= 55:
        return {"label": "Good Match", "color": "#84cc16"}
    elif score >= 35:
        return {"label": "Moderate Match", "color": "#f59e0b"}
    elif score >= 20:
        return {"label": "Low Match", "color": "#f97316"}
    else:
        return {"label": "Poor Match", "color": "#ef4444"}


if __name__ == "__main__":
    resume = """
    Python developer with 4 years of experience in machine learning, NLP,
    TensorFlow, scikit-learn, SQL, and data visualisation.
    """
    jd = """
    We are looking for a Data Scientist skilled in Python, Machine Learning,
    NLP, TensorFlow, SQL, and statistical analysis.
    """
    score = compute_match_score(resume, jd)
    print(f"Match Score: {score}%")
    interp = match_score_interpretation(score)
    print(f"Label: {interp['label']}")
