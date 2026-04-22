"""
generate_dataset.py - Generates synthetic resume/job dataset (dataset.csv)
Run once before training: python generate_dataset.py
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── Skill pools per role ──────────────────────────────────────────────────────
ROLE_SKILLS = {
    "Data Scientist": [
        "python","machine learning","deep learning","tensorflow","pytorch","keras",
        "scikit-learn","pandas","numpy","sql","r","statistics","data visualization",
        "matplotlib","seaborn","nlp","computer vision","big data","spark","hadoop",
        "tableau","power bi","excel","jupyter","git","docker","aws","azure","gcp",
        "feature engineering","model deployment","a/b testing","hypothesis testing",
    ],
    "Web Developer": [
        "html","css","javascript","react","angular","vue","nodejs","express",
        "typescript","rest api","graphql","mongodb","mysql","postgresql","git",
        "docker","aws","azure","ci/cd","agile","scrum","responsive design",
        "tailwind","bootstrap","webpack","jest","testing","redux","next.js",
        "php","laravel","django","flask","redis","elasticsearch",
    ],
    "DevOps Engineer": [
        "docker","kubernetes","jenkins","ansible","terraform","git","linux",
        "aws","azure","gcp","ci/cd","bash","python","monitoring","prometheus",
        "grafana","elk stack","nginx","apache","networking","security","agile",
        "chef","puppet","helm","argocd","gitlab","github actions","vagrant",
    ],
    "Android Developer": [
        "java","kotlin","android sdk","xml","retrofit","mvvm","room database",
        "jetpack compose","firebase","rest api","git","agile","unit testing",
        "gradle","material design","rxjava","coroutines","dagger","hilt",
        "google play","push notifications","sqlite","ble","nfc",
    ],
    "Machine Learning Engineer": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","mlops","docker","kubernetes","aws sagemaker","airflow",
        "mlflow","feature store","model serving","spark","sql","statistics",
        "numpy","pandas","git","linux","rest api","kafka","redis","triton",
    ],
    "Data Analyst": [
        "sql","excel","tableau","power bi","python","r","statistics","pandas",
        "numpy","data visualization","google analytics","looker","snowflake",
        "etl","data warehousing","mysql","postgresql","a/b testing","reporting",
        "dashboard","pivot tables","vlookup","google sheets","business analysis",
    ],
    "UI/UX Designer": [
        "figma","sketch","adobe xd","photoshop","illustrator","prototyping",
        "wireframing","user research","usability testing","css","html","invision",
        "zeplin","design thinking","information architecture","typography",
        "color theory","responsive design","accessibility","motion design",
    ],
    "Cybersecurity Analyst": [
        "network security","ethical hacking","penetration testing","siem",
        "vulnerability assessment","firewall","ids/ips","python","bash","linux",
        "windows server","active directory","wireshark","metasploit","nmap",
        "owasp","iso 27001","soc","incident response","threat intelligence",
        "cryptography","pki","vpn","zero trust","compliance",
    ],
}

EDUCATION_LEVELS = [
    "B.Tech in Computer Science",
    "B.E. in Information Technology",
    "M.Tech in Data Science",
    "B.Sc in Computer Science",
    "MCA",
    "BCA",
    "M.Sc in Machine Learning",
    "B.Tech in Electronics",
    "MBA in Information Systems",
    "B.Tech in Software Engineering",
]

COMPANIES = [
    "Google","Microsoft","Amazon","TCS","Infosys","Wipro","Accenture",
    "Cognizant","HCL","Tech Mahindra","IBM","Oracle","Deloitte","Capgemini",
    "Flipkart","Zomato","Swiggy","BYJU's","Paytm","PhonePe","Razorpay",
    "Freshworks","Zoho","MakeMyTrip","InMobi","Ola","Nykaa","Zepto",
]

FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Ananya","Diya",
    "Priya","Riya","Rohan","Rahul","Neha","Pooja","Kiran","Vikram",
    "Amit","Sneha","Kavya","Ishaan","Shruti","Meera","Dev","Raj","Simran",
]
LAST_NAMES = [
    "Sharma","Verma","Singh","Kumar","Gupta","Patel","Shah","Mehta",
    "Joshi","Rao","Nair","Pillai","Reddy","Iyer","Bhat","Malhotra",
    "Chopra","Kapoor","Saxena","Pandey","Mishra","Tiwari","Das","Banerjee",
]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_email(name):
    parts = name.lower().split()
    return f"{parts[0]}.{parts[1]}{random.randint(10,99)}@{random.choice(['gmail','yahoo','outlook'])}.com"


def random_phone():
    return f"+91 {random.randint(7000000000,9999999999)}"


def build_resume_text(name, email, phone, role, skills, edu, exp_years):
    """Construct a realistic resume-like text blob."""
    skill_sample = random.sample(skills, min(len(skills), random.randint(8, 16)))
    company_sample = random.sample(COMPANIES, min(exp_years, 3)) if exp_years > 0 else []

    lines = [
        f"Name: {name}",
        f"Email: {email}",
        f"Phone: {phone}",
        "",
        "OBJECTIVE",
        f"Motivated {role} with {exp_years} year(s) of experience seeking challenging opportunities.",
        "",
        "EDUCATION",
        f"{edu} | {random.randint(2015,2023)}",
        "",
        "SKILLS",
        ", ".join(skill_sample),
        "",
        "EXPERIENCE",
    ]
    for i, company in enumerate(company_sample):
        lines += [
            f"{role} – {company} ({2024 - exp_years + i} – {2024 - exp_years + i + 1})",
            f"  • Worked on {random.choice(skill_sample)} and {random.choice(skill_sample)} projects.",
            f"  • Collaborated with cross-functional teams to deliver high-quality solutions.",
        ]
    if not company_sample:
        lines.append("Fresher – No prior experience.")
    lines += [
        "",
        "CERTIFICATIONS",
        f"  • {role} Professional Certificate – Coursera ({random.randint(2021,2024)})",
        f"  • {random.choice(['AWS Certified','Google Cloud','Microsoft Azure','Oracle'])} Associate",
    ]
    return "\n".join(lines)


def compute_ats_score(skills_count, exp_years, edu):
    """Heuristic ATS score with controlled noise."""
    base = min(skills_count * 3.5, 55)
    exp_bonus = min(exp_years * 4, 25)
    edu_bonus = 10 if "M." in edu or "MBA" in edu else 5
    noise = random.gauss(0, 4)
    score = base + exp_bonus + edu_bonus + noise
    return round(float(np.clip(score, 20, 98)), 2)


def generate_dataset(n=400):
    records = []
    roles = list(ROLE_SKILLS.keys())

    for _ in range(n):
        role = random.choice(roles)
        skills_pool = ROLE_SKILLS[role]
        exp_years = random.randint(0, 10)
        edu = random.choice(EDUCATION_LEVELS)
        name = random_name()
        email = random_email(name)
        phone = random_phone()
        selected_skills = random.sample(skills_pool, random.randint(6, min(20, len(skills_pool))))
        resume_text = build_resume_text(name, email, phone, role, selected_skills, edu, exp_years)
        ats_score = compute_ats_score(len(selected_skills), exp_years, edu)

        records.append({
            "name": name,
            "email": email,
            "role": role,
            "skills": ", ".join(selected_skills),
            "education": edu,
            "experience_years": exp_years,
            "resume_text": resume_text,
            "ats_score": ats_score,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(400)
    df.to_csv("data/dataset.csv", index=False)
    print(f"✅ Dataset saved → data/dataset.csv  ({len(df)} rows)")
    print(df[["name","role","experience_years","ats_score"]].head(10))
