# 🚀 CareerPilot – AI Career Copilot

An end-to-end **Machine Learning + NLP** resume intelligence platform built with Python, scikit-learn, and Streamlit.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 Resume Parser | Extracts Name, Email, Phone, Education using regex + spaCy NER |
| 🤖 ATS Score | Predicts ATS compatibility (0–100) using Random Forest + TF-IDF |
| 🎯 Job Matching | Cosine similarity between resume and job description |
| 🛠️ Skill Extraction | Matches 80+ skills from a curated taxonomy |
| 🔍 Skill Gap Analysis | Shows matched vs missing skills |
| 💡 Recommendations | Actionable improvement tips |
| 🏆 Resume Ranking | Ranks multiple candidates against one JD |

---

## 📁 Project Structure

```
CareerPilot/
│
├── data/
│   └── dataset.csv              # 400-row synthetic resume dataset
│
├── models/
│   ├── ats_model.pkl            # Trained Random Forest model
│   └── vectorizer.pkl           # Fitted TF-IDF vectorizer
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Text cleaning pipeline
│   ├── parser.py                # Resume parser (NER + regex)
│   ├── skill_extractor.py       # Skill taxonomy + extraction
│   ├── ats_score.py             # ATS score model (train + predict)
│   ├── job_matcher.py           # Cosine similarity job matching
│   └── recommender.py           # Skill gap + recommendations
│
├── app/
│   └── app.py                   # Streamlit UI
│
├── generate_dataset.py          # Synthetic dataset generator
├── train_model.py               # One-shot training script
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download spaCy model for better NER
```bash
python -m spacy download en_core_web_sm
```

### 3. Generate dataset + train model
```bash
python train_model.py
```

### 4. Launch the app
```bash
streamlit run app/app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Tech Stack

- **Python 3.9+**
- **scikit-learn** – Random Forest, TF-IDF, Cosine Similarity
- **NLTK** – Stopwords, tokenization
- **spaCy** – Named Entity Recognition (NER)
- **Streamlit** – Web UI
- **Matplotlib** – Charts & gauges
- **Joblib** – Model persistence

---

## 🎯 How to Use

### Analyze Resume
1. Select target role from sidebar
2. Paste or upload your resume
3. Paste the job description
4. Click **⚡ Analyze Now**
5. View ATS score, match %, skill gaps, and recommendations
6. Download JSON report

### Compare Resumes
1. Go to **📊 Compare Resumes** page
2. Paste job description
3. Add 2–8 candidate resumes
4. Click **🏆 Rank Candidates**

---

## 📊 Model Details

| Attribute | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| Features | TF-IDF (3000 features, bigrams) |
| Dataset | 400 synthetic resume records |
| Target | ATS score (0–100) |

---

## 🙌 Author

Built with ❤️ as a complete ML + NLP portfolio project.
