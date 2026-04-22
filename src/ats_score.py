"""
src/ats_score.py
ATS Score Predictor using a trained Random Forest regression model.
Provides training + inference utilities.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ats_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")


# ── Training ──────────────────────────────────────────────────────────────────

def train_ats_model(df):
    """
    Train and save an ATS score predictor.
    df must have columns: 'resume_text', 'ats_score'
    Returns (pipeline, metrics_dict)
    """
    import pandas as pd

    # Preprocess text
    X_text = df["resume_text"].apply(lambda t: clean_text(t, remove_stops=True))
    y = df["ats_score"].values

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(X_text)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model: Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 100)
    mae = mean_absolute_error(y_test, y_pred_clipped)
    r2 = r2_score(y_test, y_pred_clipped)

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"✅ ATS model saved → {MODEL_PATH}")
    print(f"   MAE: {mae:.2f}  |  R²: {r2:.3f}")

    return model, vectorizer, {"mae": mae, "r2": r2}


# ── Inference ─────────────────────────────────────────────────────────────────

def load_ats_model():
    """Load trained model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ATS model not found. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_ats_score(resume_text: str, model=None, vectorizer=None) -> float:
    """
    Predict ATS score (0–100) for a given resume text.
    Loads model from disk if not provided.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_ats_model()

    cleaned = clean_text(resume_text, remove_stops=True)
    vec = vectorizer.transform([cleaned])
    score = model.predict(vec)[0]
    return round(float(np.clip(score, 0, 100)), 1)


def score_interpretation(score: float) -> dict:
    """Return label + colour + advice based on ATS score."""
    if score >= 80:
        return {
            "label": "Excellent",
            "color": "#22c55e",
            "advice": "Your resume is highly optimised for ATS systems.",
        }
    elif score >= 65:
        return {
            "label": "Good",
            "color": "#84cc16",
            "advice": "Solid resume. Minor keyword additions can push it higher.",
        }
    elif score >= 50:
        return {
            "label": "Average",
            "color": "#f59e0b",
            "advice": "Add more role-specific keywords and quantify achievements.",
        }
    elif score >= 35:
        return {
            "label": "Below Average",
            "color": "#f97316",
            "advice": "Resume needs significant keyword enrichment and restructuring.",
        }
    else:
        return {
            "label": "Poor",
            "color": "#ef4444",
            "advice": "Major revision required. Focus on role-specific skills and formatting.",
        }


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/dataset.csv")
    train_ats_model(df)
