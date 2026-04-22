"""
src/preprocessing.py
Text preprocessing utilities used across the CareerPilot pipeline.
"""

import re
import string

# We'll use NLTK's stopwords; spaCy tokenizer if available
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords."""
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(filtered)


def remove_urls(text: str) -> str:
    """Strip URLs from text."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+", "", text)


def remove_phone_numbers(text: str) -> str:
    """Remove phone number patterns."""
    return re.sub(r"(\+?\d[\d\s\-]{7,}\d)", "", text)


def remove_special_characters(text: str) -> str:
    """Keep only alphanumeric characters and spaces."""
    return re.sub(r"[^a-z0-9\s]", " ", text)


def clean_text(text: str, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs, emails, phone numbers
      3. Remove punctuation & special chars
      4. Remove extra whitespace
      5. Optionally remove stopwords
    """
    if not isinstance(text, str):
        text = str(text)

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    if remove_stops:
        text = remove_stopwords(text)

    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return text.split()


if __name__ == "__main__":
    sample = "Hello! Visit https://example.com or email me at test@mail.com. Ph: +91 9876543210"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
