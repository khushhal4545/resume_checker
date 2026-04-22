"""
app/app.py  –  CareerPilot: AI Career Copilot
Run: streamlit run app/app.py
"""

import os
import sys
import json
import io
import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.parser import parse_resume
from src.skill_extractor import extract_skills, get_all_roles
from src.ats_score import predict_ats_score, score_interpretation, load_ats_model
from src.job_matcher import compute_match_score, rank_resumes, match_score_interpretation
from src.recommender import (
    analyze_skill_gap,
    generate_recommendations,
    keyword_suggestions,
    role_based_tips,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPilot – AI Career Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Global ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; }

/* Main bg */
.main { background: #0f172a; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ─── Cards ──────────────────────────────────── */
.cp-card {
    background: linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
}
.cp-card-accent {
    background: linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);
    border-left: 4px solid #3b82f6;
}

/* ─── Score ring ─────────────────────────────── */
.score-ring {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:4px;
}
.score-number {
    font-size:3rem; font-weight:700; font-family:'Space Grotesk',sans-serif;
    line-height:1;
}
.score-label { font-size:.85rem; color:#94a3b8; letter-spacing:.04em; }

/* ─── Skill tags ─────────────────────────────── */
.tag-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:.4rem; }
.tag {
    padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:500;
}
.tag-green  { background:#14532d; color:#4ade80; border:1px solid #166534; }
.tag-red    { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; }
.tag-blue   { background:#172554; color:#93c5fd; border:1px solid #1e3a8a; }
.tag-yellow { background:#422006; color:#fbbf24; border:1px solid #78350f; }

/* ─── Section headings ───────────────────────── */
.sec-head {
    font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600;
    color:#e2e8f0; border-bottom:1px solid #334155; padding-bottom:.4rem;
    margin-bottom:.8rem;
}

/* ─── Sidebar ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}

/* ─── Metric overrides ───────────────────────── */
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
[data-testid="stMetricLabel"] { color:#94a3b8 !important; }

/* ─── Progress bar ───────────────────────────── */
.stProgress > div > div { border-radius:999px !important; }

/* ─── Tip box ────────────────────────────────── */
.tip-box {
    background:#1e293b; border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0; padding:.7rem 1rem;
    margin:.4rem 0; color:#cbd5e1; font-size:.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: load model (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_ats_model()
    except FileNotFoundError:
        return None, None


def render_tags(skills: list, tag_class: str) -> str:
    tags = "".join(f'<span class="tag {tag_class}">{s}</span>' for s in skills)
    return f'<div class="tag-wrap">{tags}</div>'


def gauge_chart(score: float, label: str, color: str):
    """Draw a matplotlib half-donut gauge."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    theta = np.linspace(np.pi, 0, 200)
    r_outer, r_inner = 1.0, 0.65

    # Background arc
    ax.fill_between(
        np.cos(theta), np.sin(theta),
        r_inner * np.sin(theta) / np.sin(theta),  # trick
        color="#334155", alpha=0.5,
    )
    # Draw grey track
    ax.plot(np.cos(theta), np.sin(theta), color="#334155", lw=12, solid_capstyle="round")
    ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), color="#1e293b", lw=1)

    # Filled arc proportional to score
    fill_angle = np.pi * (score / 100)
    theta_fill = np.linspace(np.pi, np.pi - fill_angle, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, lw=12, solid_capstyle="round")

    ax.text(0, 0.1, f"{score:.0f}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.22, label, ha="center", va="center",
            fontsize=9, color="#94a3b8")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def bar_chart_skills(matched: list, missing: list):
    """Horizontal bar showing matched vs missing skills count."""
    fig, ax = plt.subplots(figsize=(5, 1.8))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    total = max(len(matched) + len(missing), 1)
    ax.barh(["Skills"], [len(matched)], color="#22c55e", label="Matched", height=0.5)
    ax.barh(["Skills"], [len(missing)], left=len(matched), color="#ef4444", label="Missing", height=0.5)

    ax.set_xlim(0, total)
    ax.legend(loc="upper right", fontsize=8, facecolor="#0f172a", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 CareerPilot")
    st.markdown("<span style='color:#64748b;font-size:.85rem'>AI Career Copilot v1.0</span>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Analyze Resume", "📊 Compare Resumes", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    role = st.selectbox("🎯 Target Role", ["— Select —"] + get_all_roles())
    st.markdown("<span style='color:#64748b;font-size:.8rem'>Used for role-specific gap analysis</span>", unsafe_allow_html=True)

    st.divider()
    model, vectorizer = get_model()
    if model is None:
        st.warning("⚠️ Model not trained yet.\nRun `python train_model.py` first.")
    else:
        st.success("✅ Model loaded")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1: ANALYZE RESUME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Analyze Resume":
    st.markdown("# 🚀 CareerPilot – AI Career Copilot")
    st.markdown("<span style='color:#64748b'>Paste your resume and job description to get AI-powered insights.</span>", unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Resume")
        resume_input_mode = st.radio("Input mode", ["Paste Text", "Upload .txt/.pdf"], horizontal=True)

        resume_text = ""
        if resume_input_mode == "Paste Text":
            resume_text = st.text_area(
                "Paste your resume here",
                height=300,
                placeholder="Name:\nEmail:\nSkills:\nExperience:\n...",
                label_visibility="collapsed",
            )
        else:
            uploaded = st.file_uploader("Upload Resume", type=["txt", "pdf"])
            if uploaded:
                if uploaded.type == "application/pdf":
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(io.BytesIO(uploaded.read()))
                        resume_text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    except Exception as e:
                        st.error(f"PDF parsing error: {e}")
                else:
                    resume_text = uploaded.read().decode("utf-8", errors="ignore")

    with col_right:
        st.markdown("### 💼 Job Description")
        jd_text = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="We are looking for a Data Scientist with...",
            label_visibility="collapsed",
        )

    analyze_btn = st.button("⚡ Analyze Now", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip():
            st.error("Please provide a resume.")
            st.stop()
        if not jd_text.strip():
            st.error("Please provide a job description.")
            st.stop()

        with st.spinner("Analysing your resume…"):

            # 1. Parse
            parsed = parse_resume(resume_text)

            # 2. ATS Score
            if model:
                ats = predict_ats_score(resume_text, model, vectorizer)
            else:
                # Heuristic fallback
                skill_count = len(extract_skills(resume_text))
                ats = min(20 + skill_count * 3.5, 98)
            ats_info = score_interpretation(ats)

            # 3. Match Score
            match = compute_match_score(resume_text, jd_text)
            match_info = match_score_interpretation(match)

            # 4. Skill gap
            selected_role = role if role != "— Select —" else None
            gap = analyze_skill_gap(resume_text, jd_text, role=selected_role)

            # 5. Recommendations
            tips = generate_recommendations(gap, ats, match)
            kw_sug = keyword_suggestions(jd_text, resume_text, top_n=10)
            role_tips = role_based_tips(selected_role) if selected_role else []

        # ── Results ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Row 1: Scores
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🤖 ATS Score", f"{ats:.0f}/100", help="Predicted ATS compatibility score")
            st.progress(int(ats))
        with c2:
            st.metric("🎯 Job Match", f"{match:.1f}%", help="Cosine similarity with job description")
            st.progress(int(match))
        with c3:
            st.metric("✅ Skills Found", len(gap["resume_skills"]))
        with c4:
            st.metric("⚠️ Skills Missing", len(gap["missing_from_jd"]))

        # Row 2: Gauges
        g1, g2 = st.columns(2)
        with g1:
            fig = gauge_chart(ats, ats_info["label"], ats_info["color"])
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with g2:
            fig = gauge_chart(match, match_info["label"], match_info["color"])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Row 3: Parsed Info
        st.markdown("---")
        st.markdown("### 👤 Parsed Resume Info")
        pi1, pi2, pi3 = st.columns(3)
        with pi1:
            st.markdown(f"**Name:** {parsed['name']}")
            st.markdown(f"**Email:** {parsed['email'] or '—'}")
        with pi2:
            st.markdown(f"**Phone:** {parsed['phone'] or '—'}")
        with pi3:
            edu_list = parsed['education']
            st.markdown(f"**Education:** {edu_list[0] if edu_list else '—'}")

        # Row 4: Skills
        st.markdown("---")
        st.markdown("### 🛠️ Skills Analysis")
        sk1, sk2, sk3 = st.columns(3)
        with sk1:
            st.markdown('<p class="sec-head">✅ Your Skills</p>', unsafe_allow_html=True)
            if gap["resume_skills"]:
                st.markdown(render_tags(gap["resume_skills"], "tag-blue"), unsafe_allow_html=True)
            else:
                st.info("No skills detected")
        with sk2:
            st.markdown('<p class="sec-head">🎯 Matched Skills</p>', unsafe_allow_html=True)
            if gap["matched_skills"]:
                st.markdown(render_tags(gap["matched_skills"], "tag-green"), unsafe_allow_html=True)
            else:
                st.warning("No skill overlap detected")
        with sk3:
            st.markdown('<p class="sec-head">❌ Missing Skills (from JD)</p>', unsafe_allow_html=True)
            if gap["missing_from_jd"]:
                st.markdown(render_tags(gap["missing_from_jd"], "tag-red"), unsafe_allow_html=True)
            else:
                st.success("All JD skills covered!")

        # Skill bar chart
        if gap["resume_skills"] or gap["jd_skills"]:
            fig = bar_chart_skills(gap["matched_skills"], gap["missing_from_jd"])
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Coverage
        st.markdown(f"**Skill Coverage:** `{gap['skill_coverage']}%` of JD skills found in your resume")
        st.progress(int(gap["skill_coverage"]))

        # Row 5: Role gap
        if selected_role and gap["missing_from_role"]:
            st.markdown("---")
            st.markdown(f"### 🔍 Role Gap – {selected_role}")
            st.markdown("Skills typically required for this role that are missing from your resume:")
            st.markdown(render_tags(gap["missing_from_role"], "tag-yellow"), unsafe_allow_html=True)

        # Row 6: Keywords
        if kw_sug:
            st.markdown("---")
            st.markdown("### 🔑 Keyword Suggestions")
            st.markdown("Add these keywords from the JD into your resume (if applicable):")
            st.markdown(render_tags(kw_sug, "tag-blue"), unsafe_allow_html=True)

        # Row 7: Recommendations
        st.markdown("---")
        st.markdown("### 💡 Improvement Recommendations")
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        if role_tips:
            st.markdown(f"#### 🎯 Role-Specific Tips for {selected_role}")
            for tip in role_tips:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        # Row 8: Download
        st.markdown("---")
        report = {
            "ats_score": ats,
            "match_score": match,
            "parsed_info": {k: v for k, v in parsed.items() if k not in ("raw_text", "sections")},
            "skill_gap": gap,
            "keyword_suggestions": kw_sug,
            "recommendations": tips,
        }
        st.download_button(
            "⬇️ Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="careerpilot_report.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2: COMPARE / RANK RESUMES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Compare Resumes":
    st.markdown("# 📊 Multi-Resume Comparison")
    st.markdown("Rank multiple candidates against a single job description.")
    st.markdown("---")

    jd_compare = st.text_area("💼 Job Description", height=180,
                               placeholder="Paste the job description here…")

    st.markdown("#### 📄 Candidate Resumes")
    n_resumes = st.slider("Number of resumes to compare", 2, 8, 3)

    resumes_data = []
    cols = st.columns(min(n_resumes, 3))
    for i in range(n_resumes):
        with cols[i % 3]:
            label = st.text_input(f"Candidate {i+1} name", value=f"Candidate {i+1}", key=f"lbl_{i}")
            text = st.text_area(f"Resume {i+1}", height=160, key=f"res_{i}",
                                 placeholder="Paste resume text…")
            if text.strip():
                resumes_data.append((label, text))

    if st.button("🏆 Rank Candidates", type="primary", use_container_width=True):
        if not jd_compare.strip():
            st.error("Please provide a job description.")
        elif len(resumes_data) < 2:
            st.error("Please provide at least 2 resumes.")
        else:
            with st.spinner("Ranking candidates…"):
                rankings = rank_resumes(resumes_data, jd_compare)

            st.markdown("---")
            st.markdown("### 🏆 Ranking Results")

            # Table
            df_rank = pd.DataFrame(rankings)[["rank", "label", "score"]]
            df_rank.columns = ["Rank", "Candidate", "Match Score (%)"]
            st.dataframe(df_rank.style.highlight_max(subset=["Match Score (%)"], color="#14532d"), use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, max(2.5, len(rankings) * 0.6)))
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            labels = [r["label"] for r in rankings]
            scores = [r["score"] for r in rankings]
            colors = ["#22c55e" if s >= 60 else "#f59e0b" if s >= 35 else "#ef4444" for s in scores]
            bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.5)
            ax.set_xlabel("Match Score (%)", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            ax.set_xlim(0, 100)
            for bar, score in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{score:.1f}%", va="center", color="#e2e8f0", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Winner highlight
            winner = rankings[0]
            st.success(f"🥇 **{winner['label']}** is the best match with a score of **{winner['score']:.1f}%**")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About CareerPilot")
    st.markdown("""
    **CareerPilot – AI Career Copilot** is an end-to-end ML + NLP resume intelligence platform.

    ### 🧠 How It Works
    | Component | Technology |
    |-----------|-----------|
    | Text Preprocessing | NLTK stopwords, regex |
    | Resume Parsing | Regex + spaCy NER |
    | Skill Extraction | Custom taxonomy matching |
    | ATS Score Prediction | Random Forest Regressor + TF-IDF |
    | Job Matching | Cosine Similarity on TF-IDF vectors |
    | Skill Gap Analysis | Set difference on extracted skills |
    | Recommendations | Rule-based heuristics |

    ### 📁 Project Structure
    ```
    CareerPilot/
    ├── data/               Dataset
    ├── models/             Trained model + vectorizer
    ├── src/                Core modules
    │   ├── preprocessing.py
    │   ├── parser.py
    │   ├── skill_extractor.py
    │   ├── ats_score.py
    │   ├── job_matcher.py
    │   └── recommender.py
    ├── app/app.py          Streamlit UI
    ├── train_model.py      Training script
    ├── generate_dataset.py Dataset generator
    └── requirements.txt
    ```

    ### 🚀 Getting Started
    ```bash
    pip install -r requirements.txt
    python train_model.py       # generate data + train model
    streamlit run app/app.py    # launch UI
    ```
    """)
