"""
src/parser.py
Resume parsing: extracts Name, Email, Phone, Skills, Education
using regex + spaCy NER (falls back gracefully if spaCy unavailable).
"""

import re
from src.preprocessing import clean_text

# ── spaCy (optional) ─────────────────────────────────────────────────────────
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        _SPACY_AVAILABLE = False
except ImportError:
    _SPACY_AVAILABLE = False

# ── Education keyword patterns ────────────────────────────────────────────────
EDUCATION_KEYWORDS = [
    r"b\.?tech", r"m\.?tech", r"b\.?e\.?", r"m\.?e\.?",
    r"b\.?sc", r"m\.?sc", r"bca", r"mca", r"mba",
    r"b\.?com", r"m\.?com", r"ph\.?d", r"diploma",
    r"bachelor", r"master", r"doctorate",
    r"computer science", r"information technology", r"data science",
    r"software engineering", r"electronics", r"electrical",
]

EDUCATION_PATTERN = re.compile(
    "|".join(EDUCATION_KEYWORDS), re.IGNORECASE
)

# ── Regex helpers ─────────────────────────────────────────────────────────────

def extract_email(text: str) -> str:
    """Return the first email address found, or empty string."""
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Return the first phone number found, or empty string."""
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def extract_name_spacy(text: str) -> str:
    """Use spaCy NER to extract the first PERSON entity."""
    if not _SPACY_AVAILABLE:
        return ""
    doc = _nlp(text[:1000])  # only scan top of resume for speed
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""


def extract_name_heuristic(text: str) -> str:
    """
    Fallback: look for 'Name: <value>' pattern,
    or take the first capitalised two-word line.
    """
    # Pattern: "Name: Firstname Lastname"
    m = re.search(r"(?i)name\s*[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", text)
    if m:
        return m.group(1).strip()

    # First line with two capitalised words
    for line in text.splitlines():
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            return line
    return "Unknown"


def extract_name(text: str) -> str:
    name = extract_name_spacy(text)
    if not name:
        name = extract_name_heuristic(text)
    return name


def extract_education(text: str) -> list:
    """Return unique education-related lines from resume text."""
    results = []
    for line in text.splitlines():
        if EDUCATION_PATTERN.search(line):
            cleaned = line.strip()
            if cleaned and cleaned not in results:
                results.append(cleaned)
    return results


def extract_sections(text: str) -> dict:
    """
    Split resume into rough sections by common headings.
    Returns dict: {section_name: section_text}
    """
    section_headers = [
        "objective", "summary", "education", "experience",
        "skills", "projects", "certifications", "achievements",
        "publications", "languages", "interests",
    ]
    pattern = re.compile(
        r"(?im)^(" + "|".join(section_headers) + r")\s*[:\-]?\s*$"
    )
    sections = {}
    last_header = "header"
    last_pos = 0
    sections["header"] = ""

    for m in pattern.finditer(text):
        sections[last_header] = text[last_pos: m.start()].strip()
        last_header = m.group(1).lower()
        last_pos = m.end()

    sections[last_header] = text[last_pos:].strip()
    return sections


def parse_resume(text: str) -> dict:
    """
    Main entry-point. Returns a structured dict:
    {
        'name': str,
        'email': str,
        'phone': str,
        'education': [str, ...],
        'raw_text': str,
        'sections': dict
    }
    """
    sections = extract_sections(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": extract_education(text),
        "raw_text": text,
        "sections": sections,
    }


if __name__ == "__main__":
    sample = """
Aarav Sharma
aarav.sharma22@gmail.com  |  +91 9876543210

OBJECTIVE
Data Scientist with 3 years of experience in machine learning and NLP.

EDUCATION
B.Tech in Computer Science | IIT Delhi | 2020

SKILLS
Python, Machine Learning, TensorFlow, SQL, Pandas, NLP

EXPERIENCE
Data Scientist – Google (2021–2024)
  • Built recommendation systems using collaborative filtering.
"""
    result = parse_resume(sample)
    for k, v in result.items():
        if k != "sections":
            print(f"{k:12}: {v}")
