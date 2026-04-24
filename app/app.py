# """
# app/app.py  –  CareerPilot: AI Career Copilot
# Run: streamlit run app/app.py
# """

# import os
# import sys
# import json
# import io
# import textwrap

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ── Path setup ────────────────────────────────────────────────────────────────
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT)

# from src.parser import parse_resume
# from src.skill_extractor import extract_skills, get_all_roles
# from src.ats_score import predict_ats_score, score_interpretation, load_ats_model
# from src.job_matcher import compute_match_score, rank_resumes, match_score_interpretation
# from src.recommender import (
#     analyze_skill_gap,
#     generate_recommendations,
#     keyword_suggestions,
#     role_based_tips,
# )

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="CareerPilot – AI Career Copilot",
#     page_icon="🚀",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── Custom CSS ────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# /* ─── Global ─────────────────────────────────── */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
# h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; }

# /* Main bg */
# .main { background: #0f172a; }
# .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

# /* ─── Cards ──────────────────────────────────── */
# .cp-card {
#     background: linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
#     border: 1px solid #334155;
#     border-radius: 16px;
#     padding: 1.4rem 1.6rem;
#     margin-bottom: 1rem;
#     box-shadow: 0 4px 24px rgba(0,0,0,.35);
# }
# .cp-card-accent {
#     background: linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);
#     border-left: 4px solid #3b82f6;
# }

# /* ─── Score ring ─────────────────────────────── */
# .score-ring {
#     display:flex; flex-direction:column; align-items:center;
#     justify-content:center; gap:4px;
# }
# .score-number {
#     font-size:3rem; font-weight:700; font-family:'Space Grotesk',sans-serif;
#     line-height:1;
# }
# .score-label { font-size:.85rem; color:#94a3b8; letter-spacing:.04em; }

# /* ─── Skill tags ─────────────────────────────── */
# .tag-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:.4rem; }
# .tag {
#     padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:500;
# }
# .tag-green  { background:#14532d; color:#4ade80; border:1px solid #166534; }
# .tag-red    { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; }
# .tag-blue   { background:#172554; color:#93c5fd; border:1px solid #1e3a8a; }
# .tag-yellow { background:#422006; color:#fbbf24; border:1px solid #78350f; }

# /* ─── Section headings ───────────────────────── */
# .sec-head {
#     font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600;
#     color:#e2e8f0; border-bottom:1px solid #334155; padding-bottom:.4rem;
#     margin-bottom:.8rem;
# }

# /* ─── Sidebar ────────────────────────────────── */
# section[data-testid="stSidebar"] {
#     background: #0f172a !important;
#     border-right: 1px solid #1e293b;
# }

# /* ─── Metric overrides ───────────────────────── */
# [data-testid="stMetricValue"] { color:#e2e8f0 !important; }
# [data-testid="stMetricLabel"] { color:#94a3b8 !important; }

# /* ─── Progress bar ───────────────────────────── */
# .stProgress > div > div { border-radius:999px !important; }

# /* ─── Tip box ────────────────────────────────── */
# .tip-box {
#     background:#1e293b; border-left:3px solid #3b82f6;
#     border-radius:0 8px 8px 0; padding:.7rem 1rem;
#     margin:.4rem 0; color:#cbd5e1; font-size:.9rem;
# }
# </style>
# """, unsafe_allow_html=True)


# # ── Helper: load model (cached) ───────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def get_model():
#     try:
#         return load_ats_model()
#     except FileNotFoundError:
#         return None, None


# def render_tags(skills: list, tag_class: str) -> str:
#     tags = "".join(f'<span class="tag {tag_class}">{s}</span>' for s in skills)
#     return f'<div class="tag-wrap">{tags}</div>'


# def gauge_chart(score: float, label: str, color: str):
#     """Draw a matplotlib half-donut gauge."""
#     fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw=dict(aspect="equal"))
#     fig.patch.set_facecolor("#1e293b")
#     ax.set_facecolor("#1e293b")

#     theta = np.linspace(np.pi, 0, 200)
#     r_outer, r_inner = 1.0, 0.65

#     # Background arc
#     ax.fill_between(
#         np.cos(theta), np.sin(theta),
#         r_inner * np.sin(theta) / np.sin(theta),  # trick
#         color="#334155", alpha=0.5,
#     )
#     # Draw grey track
#     ax.plot(np.cos(theta), np.sin(theta), color="#334155", lw=12, solid_capstyle="round")
#     ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), color="#1e293b", lw=1)

#     # Filled arc proportional to score
#     fill_angle = np.pi * (score / 100)
#     theta_fill = np.linspace(np.pi, np.pi - fill_angle, 200)
#     ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, lw=12, solid_capstyle="round")

#     ax.text(0, 0.1, f"{score:.0f}", ha="center", va="center",
#             fontsize=26, fontweight="bold", color=color)
#     ax.text(0, -0.22, label, ha="center", va="center",
#             fontsize=9, color="#94a3b8")

#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-0.4, 1.2)
#     ax.axis("off")
#     plt.tight_layout(pad=0)
#     return fig


# def bar_chart_skills(matched: list, missing: list):
#     """Horizontal bar showing matched vs missing skills count."""
#     fig, ax = plt.subplots(figsize=(5, 1.8))
#     fig.patch.set_facecolor("#1e293b")
#     ax.set_facecolor("#1e293b")

#     total = max(len(matched) + len(missing), 1)
#     ax.barh(["Skills"], [len(matched)], color="#22c55e", label="Matched", height=0.5)
#     ax.barh(["Skills"], [len(missing)], left=len(matched), color="#ef4444", label="Missing", height=0.5)

#     ax.set_xlim(0, total)
#     ax.legend(loc="upper right", fontsize=8, facecolor="#0f172a", labelcolor="#e2e8f0")
#     ax.tick_params(colors="#94a3b8")
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#334155")
#     ax.xaxis.label.set_color("#94a3b8")
#     ax.yaxis.label.set_color("#94a3b8")
#     plt.tight_layout()
#     return fig


# # ─────────────────────────────────────────────────────────────────────────────
# #  SIDEBAR
# # ─────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🚀 CareerPilot")
#     st.markdown("<span style='color:#64748b;font-size:.85rem'>AI Career Copilot v1.0</span>", unsafe_allow_html=True)
#     st.divider()

#     page = st.radio(
#         "Navigate",
#         ["🏠 Analyze Resume", "📊 Compare Resumes", "ℹ️ About"],
#         label_visibility="collapsed",
#     )
#     st.divider()

#     role = st.selectbox("🎯 Target Role", ["— Select —"] + get_all_roles())
#     st.markdown("<span style='color:#64748b;font-size:.8rem'>Used for role-specific gap analysis</span>", unsafe_allow_html=True)

#     st.divider()
#     model, vectorizer = get_model()
#     if model is None:
#         st.warning("⚠️ Model not trained yet.\nRun `python train_model.py` first.")
#     else:
#         st.success("✅ Model loaded")


# # ─────────────────────────────────────────────────────────────────────────────
# #  PAGE 1: ANALYZE RESUME
# # ─────────────────────────────────────────────────────────────────────────────
# if page == "🏠 Analyze Resume":
#     st.markdown("# 🚀 CareerPilot – AI Career Copilot")
#     st.markdown("<span style='color:#64748b'>Paste your resume and job description to get AI-powered insights.</span>", unsafe_allow_html=True)
#     st.markdown("---")

#     col_left, col_right = st.columns([1, 1], gap="large")

#     with col_left:
#         st.markdown("### 📄 Resume")
#         resume_input_mode = st.radio("Input mode", ["Paste Text", "Upload .txt/.pdf"], horizontal=True)

#         resume_text = ""
#         if resume_input_mode == "Paste Text":
#             resume_text = st.text_area(
#                 "Paste your resume here",
#                 height=300,
#                 placeholder="Name:\nEmail:\nSkills:\nExperience:\n...",
#                 label_visibility="collapsed",
#             )
#         else:
#             uploaded = st.file_uploader("Upload Resume", type=["txt", "pdf"])
#             if uploaded:
#                 if uploaded.type == "application/pdf":
#                     try:
#                         import PyPDF2
#                         reader = PyPDF2.PdfReader(io.BytesIO(uploaded.read()))
#                         resume_text = "\n".join(p.extract_text() or "" for p in reader.pages)
#                     except Exception as e:
#                         st.error(f"PDF parsing error: {e}")
#                 else:
#                     resume_text = uploaded.read().decode("utf-8", errors="ignore")

#     with col_right:
#         st.markdown("### 💼 Job Description")
#         jd_text = st.text_area(
#             "Paste the job description here",
#             height=300,
#             placeholder="We are looking for a Data Scientist with...",
#             label_visibility="collapsed",
#         )

#     analyze_btn = st.button("⚡ Analyze Now", type="primary", use_container_width=True)

#     if analyze_btn:
#         if not resume_text.strip():
#             st.error("Please provide a resume.")
#             st.stop()
#         if not jd_text.strip():
#             st.error("Please provide a job description.")
#             st.stop()

#         with st.spinner("Analysing your resume…"):

#             # 1. Parse
#             parsed = parse_resume(resume_text)

#             # 2. ATS Score
#             if model:
#                 ats = predict_ats_score(resume_text, model, vectorizer)
#             else:
#                 # Heuristic fallback
#                 skill_count = len(extract_skills(resume_text))
#                 ats = min(20 + skill_count * 3.5, 98)
#             ats_info = score_interpretation(ats)

#             # 3. Match Score
#             match = compute_match_score(resume_text, jd_text)
#             match_info = match_score_interpretation(match)

#             # 4. Skill gap
#             selected_role = role if role != "— Select —" else None
#             gap = analyze_skill_gap(resume_text, jd_text, role=selected_role)

#             # 5. Recommendations
#             tips = generate_recommendations(gap, ats, match)
#             kw_sug = keyword_suggestions(jd_text, resume_text, top_n=10)
#             role_tips = role_based_tips(selected_role) if selected_role else []

#         # ── Results ───────────────────────────────────────────────────────
#         st.markdown("---")
#         st.markdown("## 📊 Analysis Results")

#         # Row 1: Scores
#         c1, c2, c3, c4 = st.columns(4)
#         with c1:
#             st.metric("🤖 ATS Score", f"{ats:.0f}/100", help="Predicted ATS compatibility score")
#             st.progress(int(ats))
#         with c2:
#             st.metric("🎯 Job Match", f"{match:.1f}%", help="Cosine similarity with job description")
#             st.progress(int(match))
#         with c3:
#             st.metric("✅ Skills Found", len(gap["resume_skills"]))
#         with c4:
#             st.metric("⚠️ Skills Missing", len(gap["missing_from_jd"]))

#         # Row 2: Gauges
#         g1, g2 = st.columns(2)
#         with g1:
#             fig = gauge_chart(ats, ats_info["label"], ats_info["color"])
#             st.pyplot(fig, use_container_width=True)
#             plt.close()
#         with g2:
#             fig = gauge_chart(match, match_info["label"], match_info["color"])
#             st.pyplot(fig, use_container_width=True)
#             plt.close()

#         # Row 3: Parsed Info
#         st.markdown("---")
#         st.markdown("### 👤 Parsed Resume Info")
#         pi1, pi2, pi3 = st.columns(3)
#         with pi1:
#             st.markdown(f"**Name:** {parsed['name']}")
#             st.markdown(f"**Email:** {parsed['email'] or '—'}")
#         with pi2:
#             st.markdown(f"**Phone:** {parsed['phone'] or '—'}")
#         with pi3:
#             edu_list = parsed['education']
#             st.markdown(f"**Education:** {edu_list[0] if edu_list else '—'}")

#         # Row 4: Skills
#         st.markdown("---")
#         st.markdown("### 🛠️ Skills Analysis")
#         sk1, sk2, sk3 = st.columns(3)
#         with sk1:
#             st.markdown('<p class="sec-head">✅ Your Skills</p>', unsafe_allow_html=True)
#             if gap["resume_skills"]:
#                 st.markdown(render_tags(gap["resume_skills"], "tag-blue"), unsafe_allow_html=True)
#             else:
#                 st.info("No skills detected")
#         with sk2:
#             st.markdown('<p class="sec-head">🎯 Matched Skills</p>', unsafe_allow_html=True)
#             if gap["matched_skills"]:
#                 st.markdown(render_tags(gap["matched_skills"], "tag-green"), unsafe_allow_html=True)
#             else:
#                 st.warning("No skill overlap detected")
#         with sk3:
#             st.markdown('<p class="sec-head">❌ Missing Skills (from JD)</p>', unsafe_allow_html=True)
#             if gap["missing_from_jd"]:
#                 st.markdown(render_tags(gap["missing_from_jd"], "tag-red"), unsafe_allow_html=True)
#             else:
#                 st.success("All JD skills covered!")

#         # Skill bar chart
#         if gap["resume_skills"] or gap["jd_skills"]:
#             fig = bar_chart_skills(gap["matched_skills"], gap["missing_from_jd"])
#             st.pyplot(fig, use_container_width=True)
#             plt.close()

#         # Coverage
#         st.markdown(f"**Skill Coverage:** `{gap['skill_coverage']}%` of JD skills found in your resume")
#         st.progress(int(gap["skill_coverage"]))

#         # Row 5: Role gap
#         if selected_role and gap["missing_from_role"]:
#             st.markdown("---")
#             st.markdown(f"### 🔍 Role Gap – {selected_role}")
#             st.markdown("Skills typically required for this role that are missing from your resume:")
#             st.markdown(render_tags(gap["missing_from_role"], "tag-yellow"), unsafe_allow_html=True)

#         # Row 6: Keywords
#         if kw_sug:
#             st.markdown("---")
#             st.markdown("### 🔑 Keyword Suggestions")
#             st.markdown("Add these keywords from the JD into your resume (if applicable):")
#             st.markdown(render_tags(kw_sug, "tag-blue"), unsafe_allow_html=True)

#         # Row 7: Recommendations
#         st.markdown("---")
#         st.markdown("### 💡 Improvement Recommendations")
#         for tip in tips:
#             st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

#         if role_tips:
#             st.markdown(f"#### 🎯 Role-Specific Tips for {selected_role}")
#             for tip in role_tips:
#                 st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

#         # Row 8: Download
#         st.markdown("---")
#         report = {
#             "ats_score": ats,
#             "match_score": match,
#             "parsed_info": {k: v for k, v in parsed.items() if k not in ("raw_text", "sections")},
#             "skill_gap": gap,
#             "keyword_suggestions": kw_sug,
#             "recommendations": tips,
#         }
#         st.download_button(
#             "⬇️ Download Report (JSON)",
#             data=json.dumps(report, indent=2),
#             file_name="careerpilot_report.json",
#             mime="application/json",
#         )


# # ─────────────────────────────────────────────────────────────────────────────
# #  PAGE 2: COMPARE / RANK RESUMES
# # ─────────────────────────────────────────────────────────────────────────────
# elif page == "📊 Compare Resumes":
#     st.markdown("# 📊 Multi-Resume Comparison")
#     st.markdown("Rank multiple candidates against a single job description.")
#     st.markdown("---")

#     jd_compare = st.text_area("💼 Job Description", height=180,
#                                placeholder="Paste the job description here…")

#     st.markdown("#### 📄 Candidate Resumes")
#     n_resumes = st.slider("Number of resumes to compare", 2, 8, 3)

#     resumes_data = []
#     cols = st.columns(min(n_resumes, 3))
#     for i in range(n_resumes):
#         with cols[i % 3]:
#             label = st.text_input(f"Candidate {i+1} name", value=f"Candidate {i+1}", key=f"lbl_{i}")
#             text = st.text_area(f"Resume {i+1}", height=160, key=f"res_{i}",
#                                  placeholder="Paste resume text…")
#             if text.strip():
#                 resumes_data.append((label, text))

#     if st.button("🏆 Rank Candidates", type="primary", use_container_width=True):
#         if not jd_compare.strip():
#             st.error("Please provide a job description.")
#         elif len(resumes_data) < 2:
#             st.error("Please provide at least 2 resumes.")
#         else:
#             with st.spinner("Ranking candidates…"):
#                 rankings = rank_resumes(resumes_data, jd_compare)

#             st.markdown("---")
#             st.markdown("### 🏆 Ranking Results")

#             # Table
#             df_rank = pd.DataFrame(rankings)[["rank", "label", "score"]]
#             df_rank.columns = ["Rank", "Candidate", "Match Score (%)"]
#             st.dataframe(df_rank.style.highlight_max(subset=["Match Score (%)"], color="#14532d"), use_container_width=True)

#             # Bar chart
#             fig, ax = plt.subplots(figsize=(7, max(2.5, len(rankings) * 0.6)))
#             fig.patch.set_facecolor("#1e293b")
#             ax.set_facecolor("#1e293b")
#             labels = [r["label"] for r in rankings]
#             scores = [r["score"] for r in rankings]
#             colors = ["#22c55e" if s >= 60 else "#f59e0b" if s >= 35 else "#ef4444" for s in scores]
#             bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.5)
#             ax.set_xlabel("Match Score (%)", color="#94a3b8")
#             ax.tick_params(colors="#94a3b8")
#             for spine in ax.spines.values():
#                 spine.set_edgecolor("#334155")
#             ax.set_xlim(0, 100)
#             for bar, score in zip(bars, scores[::-1]):
#                 ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
#                         f"{score:.1f}%", va="center", color="#e2e8f0", fontsize=9)
#             plt.tight_layout()
#             st.pyplot(fig, use_container_width=True)
#             plt.close()

#             # Winner highlight
#             winner = rankings[0]
#             st.success(f"🥇 **{winner['label']}** is the best match with a score of **{winner['score']:.1f}%**")


# # ─────────────────────────────────────────────────────────────────────────────
# #  PAGE 3: ABOUT
# # ─────────────────────────────────────────────────────────────────────────────
# elif page == "ℹ️ About":
#     st.markdown("# ℹ️ About CareerPilot")
#     st.markdown("""
#     **CareerPilot – AI Career Copilot** is an end-to-end ML + NLP resume intelligence platform.

#     ### 🧠 How It Works
#     | Component | Technology |
#     |-----------|-----------|
#     | Text Preprocessing | NLTK stopwords, regex |
#     | Resume Parsing | Regex + spaCy NER |
#     | Skill Extraction | Custom taxonomy matching |
#     | ATS Score Prediction | Random Forest Regressor + TF-IDF |
#     | Job Matching | Cosine Similarity on TF-IDF vectors |
#     | Skill Gap Analysis | Set difference on extracted skills |
#     | Recommendations | Rule-based heuristics |

#     ### 📁 Project Structure
#     ```
#     CareerPilot/
#     ├── data/               Dataset
#     ├── models/             Trained model + vectorizer
#     ├── src/                Core modules
#     │   ├── preprocessing.py
#     │   ├── parser.py
#     │   ├── skill_extractor.py
#     │   ├── ats_score.py
#     │   ├── job_matcher.py
#     │   └── recommender.py
#     ├── app/app.py          Streamlit UI
#     ├── train_model.py      Training script
#     ├── generate_dataset.py Dataset generator
#     └── requirements.txt
#     ```

#     ### 🚀 Getting Started
#     ```bash
#     pip install -r requirements.txt
#     python train_model.py       # generate data + train model
#     streamlit run app/app.py    # launch UI
#     ```
#     """)



"""
app/app.py – CareerPilot: AI Career Copilot
Run: streamlit run app/app.py
"""

import os
import sys
import json
import io
import html
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

st.set_page_config(
    page_title="CareerPilot – AI Career Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── CSS ─────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
    --bg:#07111f;
    --panel:#0f1b2e;
    --panel2:#111f35;
    --muted:#94a3b8;
    --text:#e5eefb;
    --line:rgba(148,163,184,.18);
    --brand:#38bdf8;
    --brand2:#8b5cf6;
    --green:#22c55e;
    --red:#ef4444;
    --yellow:#f59e0b;
}

html, body, [class*="css"] {font-family:'Inter',sans-serif;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; letter-spacing:-.02em;}
.stApp {
    background:
      radial-gradient(circle at top left, rgba(56,189,248,.18), transparent 32%),
      radial-gradient(circle at top right, rgba(139,92,246,.16), transparent 30%),
      linear-gradient(180deg,#07111f 0%,#0b1220 100%);
    color:var(--text);
}
.block-container {padding-top:1.2rem; padding-bottom:2.5rem; max-width:1280px;}

/* sidebar */
section[data-testid="stSidebar"] {background:rgba(5,12,23,.92)!important; border-right:1px solid var(--line);}
section[data-testid="stSidebar"] * {color:#dbeafe;}

/* inputs */
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background:#0b1628!important; color:#e5eefb!important; border:1px solid rgba(148,163,184,.25)!important;
    border-radius:14px!important;
}
.stTextArea textarea:focus, .stTextInput input:focus {border-color:#38bdf8!important; box-shadow:0 0 0 2px rgba(56,189,248,.16)!important;}

/* buttons */
.stButton > button, .stDownloadButton > button {
    border-radius:14px!important; min-height:45px; font-weight:700!important;
    border:1px solid rgba(56,189,248,.3)!important;
    background:linear-gradient(135deg,#38bdf8,#8b5cf6)!important;
    color:white!important; box-shadow:0 12px 30px rgba(56,189,248,.18)!important;
}
.stButton > button:hover, .stDownloadButton > button:hover {transform:translateY(-1px); filter:brightness(1.08);}

/* custom components */
.hero {
    position:relative; overflow:hidden; border:1px solid rgba(148,163,184,.2); border-radius:28px;
    padding:30px 32px; margin-bottom:22px;
    background:linear-gradient(135deg,rgba(15,27,46,.96),rgba(17,24,39,.82));
    box-shadow:0 24px 70px rgba(0,0,0,.28);
}
.hero:after{
    content:""; position:absolute; right:-100px; top:-120px; width:340px; height:340px;
    background:radial-gradient(circle,rgba(56,189,248,.38),transparent 62%);
}
.hero-title {font-family:'Space Grotesk',sans-serif; font-size:2.8rem; font-weight:800; line-height:1.05; margin:0;}
.hero-sub {color:var(--muted); font-size:1.02rem; max-width:720px; margin-top:10px;}
.pill-row {display:flex; flex-wrap:wrap; gap:10px; margin-top:18px;}
.pill {background:rgba(56,189,248,.10); border:1px solid rgba(56,189,248,.25); color:#bae6fd; padding:7px 12px; border-radius:999px; font-size:.82rem; font-weight:700;}

.glass-card {
    background:linear-gradient(180deg,rgba(15,27,46,.92),rgba(15,23,42,.78)); border:1px solid var(--line);
    border-radius:22px; padding:20px; margin-bottom:16px; box-shadow:0 18px 45px rgba(0,0,0,.18);
}
.card-title {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.12rem; margin-bottom:10px; color:#f8fafc;}
.muted {color:var(--muted);}

.kpi {
    border:1px solid var(--line); border-radius:20px; padding:18px 18px;
    background:linear-gradient(135deg,rgba(30,41,59,.86),rgba(15,23,42,.72)); min-height:128px;
}
.kpi-label {color:var(--muted); font-size:.83rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;}
.kpi-value {font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:800; margin-top:8px; color:#f8fafc;}
.kpi-foot {color:#94a3b8; font-size:.82rem; margin-top:4px;}

.tag-wrap {display:flex; flex-wrap:wrap; gap:8px; margin-top:.45rem;}
.tag {padding:6px 11px; border-radius:999px; font-size:.80rem; font-weight:700; border:1px solid transparent;}
.tag-green {background:rgba(34,197,94,.12); color:#86efac; border-color:rgba(34,197,94,.28);}
.tag-red {background:rgba(239,68,68,.12); color:#fca5a5; border-color:rgba(239,68,68,.28);}
.tag-blue {background:rgba(56,189,248,.12); color:#bae6fd; border-color:rgba(56,189,248,.28);}
.tag-yellow {background:rgba(245,158,11,.12); color:#fcd34d; border-color:rgba(245,158,11,.30);}

.tip-box {background:rgba(15,23,42,.72); border:1px solid var(--line); border-left:4px solid #38bdf8; border-radius:14px; padding:12px 14px; margin:8px 0; color:#dbeafe;}
.info-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;}
.info-item {background:rgba(15,23,42,.65); border:1px solid var(--line); border-radius:16px; padding:14px;}
.info-label {color:var(--muted); font-size:.78rem; font-weight:800; text-transform:uppercase;}
.info-value {font-size:1rem; font-weight:700; margin-top:5px; color:#f8fafc; overflow-wrap:anywhere;}

[data-testid="stMetricValue"] {color:#f8fafc!important;}
[data-testid="stMetricLabel"] {color:#94a3b8!important;}
hr {border-color:rgba(148,163,184,.18)!important;}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────── Helpers ─────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_ats_model()
    except FileNotFoundError:
        return None, None


def safe_text(value, fallback="—"):
    if value is None or value == "":
        return fallback
    return html.escape(str(value))


def render_tags(skills, tag_class):
    if not skills:
        return '<p class="muted">No skills detected</p>'
    tags = "".join(f'<span class="tag {tag_class}">{html.escape(str(s))}</span>' for s in skills)
    return f'<div class="tag-wrap">{tags}</div>'


def hero(title, subtitle, pills=None):
    pills = pills or []
    pill_html = "".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills)
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-title">{title}</p>
            <div class="hero-sub">{subtitle}</div>
            <div class="pill-row">{pill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div class="kpi-foot">{html.escape(foot)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(score, label, color):
    score = max(0, min(float(score), 100))
    fig, ax = plt.subplots(figsize=(4.2, 2.55), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0f1b2e")
    ax.set_facecolor("#0f1b2e")

    theta = np.linspace(np.pi, 0, 220)
    ax.plot(np.cos(theta), np.sin(theta), color="#243247", lw=18, solid_capstyle="round")

    fill = np.linspace(np.pi, np.pi - np.pi * score / 100, 220)
    ax.plot(np.cos(fill), np.sin(fill), color=color, lw=18, solid_capstyle="round")

    ax.text(0, 0.16, f"{score:.0f}", ha="center", va="center", fontsize=31, fontweight="bold", color="#f8fafc")
    ax.text(0, -0.18, label, ha="center", va="center", fontsize=10, color="#94a3b8")
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-0.42, 1.18)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def bar_chart_skills(matched, missing):
    fig, ax = plt.subplots(figsize=(7, 2.1))
    fig.patch.set_facecolor("#0f1b2e")
    ax.set_facecolor("#0f1b2e")
    total = max(len(matched) + len(missing), 1)
    ax.barh(["Skill Coverage"], [len(matched)], color="#22c55e", label="Matched", height=0.48)
    ax.barh(["Skill Coverage"], [len(missing)], left=len(matched), color="#ef4444", label="Missing", height=0.48)
    ax.set_xlim(0, total)
    ax.legend(loc="upper right", fontsize=9, facecolor="#0b1628", edgecolor="#334155", labelcolor="#e5eefb")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    plt.tight_layout()
    return fig


def read_resume_upload(uploaded):
    if uploaded is None:
        return ""
    if uploaded.type == "application/pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded.read()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            st.error(f"PDF parsing error: {exc}")
            return ""
    return uploaded.read().decode("utf-8", errors="ignore")


# ───────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("# 🚀 CareerPilot")
    st.caption("Premium Resume Intelligence Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Analyze Resume", "📊 Compare Resumes", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    role = st.selectbox("🎯 Target Role", ["— Select —"] + get_all_roles())
    st.caption("Role select karne se gap analysis better hota hai.")

    st.divider()
    model, vectorizer = get_model()
    if model is None:
        st.warning("Model not trained. Run: python train_model.py")
    else:
        st.success("ATS model loaded")

# ───────────────────────────── Page 1 ─────────────────────────────
if page == "🏠 Analyze Resume":
    hero(
        "CareerPilot – AI Career Copilot",
        "Paste your resume and job description to get ATS score, job matching, skill gap analysis, keyword suggestions, and improvement tips.",
        ["ATS Score", "JD Match", "Skill Gap", "Recommendations", "Report Download"],
    )

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown('<div class="glass-card"><div class="card-title">📄 Resume Input</div>', unsafe_allow_html=True)
        mode = st.radio("Input mode", ["Paste Text", "Upload .txt/.pdf"], horizontal=True)
        resume_text = ""
        if mode == "Paste Text":
            resume_text = st.text_area(
                "Paste your resume here",
                height=320,
                placeholder="Name:\nEmail:\nSkills:\nExperience:\nProjects:\n...",
                label_visibility="collapsed",
            )
        else:
            uploaded = st.file_uploader("Upload Resume", type=["txt", "pdf"])
            resume_text = read_resume_upload(uploaded)
            if resume_text:
                st.success("Resume uploaded successfully")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card"><div class="card-title">💼 Job Description</div>', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Paste job description",
            height=366,
            placeholder="We are looking for a Python Developer with ML, SQL, Git, communication skills...",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_btn = st.button("⚡ Analyze Resume", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip():
            st.error("Please provide a resume.")
            st.stop()
        if not jd_text.strip():
            st.error("Please provide a job description.")
            st.stop()

        with st.spinner("Analyzing resume and job description..."):
            parsed = parse_resume(resume_text)
            if model:
                ats = predict_ats_score(resume_text, model, vectorizer)
            else:
                ats = min(20 + len(extract_skills(resume_text)) * 3.5, 98)
            ats_info = score_interpretation(ats)
            match = compute_match_score(resume_text, jd_text)
            match_info = match_score_interpretation(match)
            selected_role = role if role != "— Select —" else None
            gap = analyze_skill_gap(resume_text, jd_text, role=selected_role)
            tips = generate_recommendations(gap, ats, match)
            kw_sug = keyword_suggestions(jd_text, resume_text, top_n=10)
            role_tips = role_based_tips(selected_role) if selected_role else []

        st.markdown("## 📊 Analysis Dashboard")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("ATS Score", f"{ats:.0f}/100", ats_info.get("label", "ATS compatibility"))
        with k2:
            kpi_card("Job Match", f"{match:.1f}%", match_info.get("label", "JD similarity"))
        with k3:
            kpi_card("Skills Found", len(gap["resume_skills"]), "Detected from resume")
        with k4:
            kpi_card("Missing Skills", len(gap["missing_from_jd"]), "Required by JD")

        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown('<div class="glass-card"><div class="card-title">🤖 ATS Health</div>', unsafe_allow_html=True)
            fig = gauge_chart(ats, ats_info.get("label", "ATS"), ats_info.get("color", "#38bdf8"))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.progress(int(max(0, min(ats, 100))))
            st.markdown('</div>', unsafe_allow_html=True)
        with g2:
            st.markdown('<div class="glass-card"><div class="card-title">🎯 JD Match Health</div>', unsafe_allow_html=True)
            fig = gauge_chart(match, match_info.get("label", "Match"), match_info.get("color", "#8b5cf6"))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.progress(int(max(0, min(match, 100))))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="card-title">👤 Parsed Resume Info</div>', unsafe_allow_html=True)
        education = parsed.get("education", [])
        edu_value = education[0] if education else "—"
        st.markdown(
            f"""
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Name</div><div class="info-value">{safe_text(parsed.get('name'))}</div></div>
                <div class="info-item"><div class="info-label">Email</div><div class="info-value">{safe_text(parsed.get('email'))}</div></div>
                <div class="info-item"><div class="info-label">Phone</div><div class="info-value">{safe_text(parsed.get('phone'))}</div></div>
                <div class="info-item"><div class="info-label">Education</div><div class="info-value">{safe_text(edu_value)}</div></div>
                <div class="info-item"><div class="info-label">Role</div><div class="info-value">{safe_text(selected_role or 'Not selected')}</div></div>
                <div class="info-item"><div class="info-label">Report Time</div><div class="info-value">{datetime.now().strftime('%d %b %Y, %I:%M %p')}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("## 🛠️ Skills Analysis")
        s1, s2, s3 = st.columns(3, gap="large")
        with s1:
            st.markdown('<div class="glass-card"><div class="card-title">🔵 Your Skills</div>', unsafe_allow_html=True)
            st.markdown(render_tags(gap["resume_skills"], "tag-blue"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="glass-card"><div class="card-title">✅ Matched Skills</div>', unsafe_allow_html=True)
            st.markdown(render_tags(gap["matched_skills"], "tag-green"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="glass-card"><div class="card-title">❌ Missing From JD</div>', unsafe_allow_html=True)
            st.markdown(render_tags(gap["missing_from_jd"], "tag-red"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if gap["resume_skills"] or gap["jd_skills"]:
            st.markdown('<div class="glass-card"><div class="card-title">📈 Skill Coverage</div>', unsafe_allow_html=True)
            fig = bar_chart_skills(gap["matched_skills"], gap["missing_from_jd"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(f"**Skill Coverage:** `{gap['skill_coverage']}%` of JD skills found in resume")
            st.progress(int(max(0, min(gap["skill_coverage"], 100))))
            st.markdown('</div>', unsafe_allow_html=True)

        if selected_role and gap["missing_from_role"]:
            st.markdown('<div class="glass-card"><div class="card-title">🔍 Role Gap Analysis</div>', unsafe_allow_html=True)
            st.write(f"Skills commonly needed for **{selected_role}** but missing in your resume:")
            st.markdown(render_tags(gap["missing_from_role"], "tag-yellow"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if kw_sug:
            st.markdown('<div class="glass-card"><div class="card-title">🔑 Keyword Suggestions</div>', unsafe_allow_html=True)
            st.caption("JD ke important keywords resume mein naturally add karo, fake mat add karna.")
            st.markdown(render_tags(kw_sug, "tag-blue"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="card-title">💡 Improvement Recommendations</div>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{html.escape(str(tip))}</div>', unsafe_allow_html=True)
        if role_tips:
            st.markdown(f"#### 🎯 Role-Specific Tips for {selected_role}")
            for tip in role_tips:
                st.markdown(f'<div class="tip-box">{html.escape(str(tip))}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        report = {
            "ats_score": ats,
            "match_score": match,
            "parsed_info": {k: v for k, v in parsed.items() if k not in ("raw_text", "sections")},
            "skill_gap": gap,
            "keyword_suggestions": kw_sug,
            "recommendations": tips,
            "generated_at": datetime.now().isoformat(),
        }
        st.download_button(
            "⬇️ Download Premium Report JSON",
            data=json.dumps(report, indent=2),
            file_name="careerpilot_report.json",
            mime="application/json",
            use_container_width=True,
        )

# ───────────────────────────── Page 2 ─────────────────────────────
elif page == "📊 Compare Resumes":
    hero(
        "Multi-Resume Ranking",
        "Paste multiple candidate resumes and one job description. CareerPilot ranks candidates by job-description match score.",
        ["Candidate Ranking", "Match Score", "Visual Comparison"],
    )

    jd_compare = st.text_area("💼 Job Description", height=180, placeholder="Paste the job description here...")
    n_resumes = st.slider("Number of resumes to compare", 2, 8, 3)

    resumes_data = []
    cols = st.columns(min(n_resumes, 3))
    for i in range(n_resumes):
        with cols[i % 3]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            label = st.text_input(f"Candidate {i + 1} name", value=f"Candidate {i + 1}", key=f"lbl_{i}")
            text = st.text_area(f"Resume {i + 1}", height=170, key=f"res_{i}", placeholder="Paste resume text...")
            st.markdown('</div>', unsafe_allow_html=True)
            if text.strip():
                resumes_data.append((label, text))

    if st.button("🏆 Rank Candidates", type="primary", use_container_width=True):
        if not jd_compare.strip():
            st.error("Please provide a job description.")
        elif len(resumes_data) < 2:
            st.error("Please provide at least 2 resumes.")
        else:
            with st.spinner("Ranking candidates..."):
                rankings = rank_resumes(resumes_data, jd_compare)

            st.markdown("## 🏆 Ranking Results")
            df_rank = pd.DataFrame(rankings)[["rank", "label", "score"]]
            df_rank.columns = ["Rank", "Candidate", "Match Score (%)"]
            st.dataframe(df_rank, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, max(2.8, len(rankings) * 0.65)))
            fig.patch.set_facecolor("#0f1b2e")
            ax.set_facecolor("#0f1b2e")
            labels = [r["label"] for r in rankings]
            scores = [r["score"] for r in rankings]
            colors = ["#22c55e" if s >= 60 else "#f59e0b" if s >= 35 else "#ef4444" for s in scores]
            bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.52)
            ax.set_xlabel("Match Score (%)", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.set_xlim(0, 100)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            for bar, score in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", color="#e5eefb", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            winner = rankings[0]
            st.success(f"🥇 {winner['label']} is the best match with {winner['score']:.1f}% score.")

# ───────────────────────────── Page 3 ─────────────────────────────
elif page == "ℹ️ About":
    hero(
        "About CareerPilot",
        "An end-to-end Machine Learning + NLP resume intelligence platform built with Python, scikit-learn, and Streamlit.",
        ["ML", "NLP", "Streamlit", "Resume Intelligence"],
    )

    st.markdown(
        """
<div class="glass-card">
<div class="card-title">🧠 How It Works</div>

| Component | Technology |
|---|---|
| Text Preprocessing | NLTK stopwords, regex |
| Resume Parsing | Regex + spaCy NER |
| Skill Extraction | Custom taxonomy matching |
| ATS Score Prediction | Random Forest Regressor + TF-IDF |
| Job Matching | Cosine Similarity on TF-IDF vectors |
| Skill Gap Analysis | Set difference on extracted skills |
| Recommendations | Rule-based heuristics |

</div>

<div class="glass-card">
<div class="card-title">🚀 Run Project</div>

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app/app.py
```
</div>
        """,
        unsafe_allow_html=True,
    )
