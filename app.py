"""
src/dashboard/app.py
JobGuard — Professional AI Fake Job Detection Dashboard v2.0

🎨 Design Philosophy:
   - Enterprise Security Dashboard aesthetic
   - Dark navy/teal base with gold accents
   - Glass-morphism cards with depth
   - Modern typography (Plus Jakarta Sans + Inter)
   - Smooth animations and micro-interactions
   - Professional color psychology for trust/warning

Run: streamlit run app.py   (start the API first: uvicorn main:app --reload)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import io
from pathlib import Path
from datetime import datetime
import time

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.explainability.knowledge_graph_realtime import RealTimeSkillKnowledgeGraph
from src.preprocessing.data_cleaning import TextCleaner
from src.explainability.rule_engine import analyze_job
from src.explainability.prompt_builder import build_prompt
from src.explainability.llm_analyzer import generate_ai_analysis, analyze_job_risk
from src.explainability.red_flags import detect_red_flags
from src.explainability.job_posting import JobPostingLite
from src import config as cfg
from src.api_client import JobGuardClient, ApiError
# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="JobGuard — Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "JobGuard v2.0 — AI-Powered Employment Fraud Detection",
        "Get Help": "https://github.com/patilrakshita2203-prog/JobGuard",
        "Report a bug": "https://github.com/patilrakshita2203-prog/JobGuard/issues",
    }
)

# ═══════════════════════════════════════════════════════════════
# PROFESSIONAL CSS STYLING
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    html, body {
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}
 :root {
    --primary-dark: #0f172a;
    --primary-blue: #1e293b;
    --accent-red: #ef4444;
    --accent-green: #10b981;
    --accent-cyan: #06b6d4;
    --accent-blue: #3b82f6;

    --card-bg: #111827;
    --card-bg-2: #1e293b;

    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;

    --border-color: #334155;
}
    
    html, body, [class*="css"], [class*="st"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
   .main {
    background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
    color: #f8fafc;
    }
    
    [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
    color: #f8fafc;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1629 0%, #1a2f5a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* HEADER BANNER */
    .header-banner {
        background: linear-gradient(135deg, #0a1629 0%, #1b3a6b 50%, #2d5a8c 100%);
        padding: 3rem 2.5rem;
        border-radius: 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid #fbbf24;
    }
    
    .header-banner::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(251, 191, 36, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    
    .header-banner h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 2;
        letter-spacing: -0.5px;
    }
    
    .header-banner p {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        position: relative;
        z-index: 2;
        font-weight: 300;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(251, 191, 36, 0.2);
        border: 1px solid #fbbf24;
        color: #fef3c7;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 1rem;
        position: relative;
        z-index: 2;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* VERDICT CARDS */
    .verdict-fake {
        background: linear-gradient(135deg, #e63946 0%, #d62828 100%);
        border: 1px solid rgba(230, 57, 70, 0.3);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 12px 32px rgba(230, 57, 70, 0.25);
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .verdict-fake:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(230, 57, 70, 0.35);
    }
    
    .verdict-genuine {
        background: linear-gradient(135deg, #06a77d 0%, #059669 100%);
        border: 1px solid rgba(6, 167, 125, 0.3);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 12px 32px rgba(6, 167, 125, 0.25);
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .verdict-genuine:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 48px rgba(6, 167, 125, 0.35);
    }
    
    /* METRIC CARDS */
    .metric-card {
        background: #111827;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 12px;
        padding: 1.75rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card:hover {
        border-color: #fbbf24;
        box-shadow: 0 12px 32px rgba(251, 191, 36, 0.15);
        transform: translateY(-4px);
    }
    
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* RISK BADGES */
    .risk-high {
        background: linear-gradient(135deg, rgba(230, 57, 70, 0.12) 0%, rgba(214, 40, 40, 0.08) 100%);
        border: 2px solid #e63946;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        font-weight: 700;
        color: #7f1d1d;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .risk-high::before {
        content: '🚨';
        font-size: 1.5rem;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12) 0%, rgba(217, 119, 6, 0.08) 100%);
        border: 2px solid #fbbf24;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        font-weight: 700;
        color: #78350f;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .risk-medium::before {
        content: '⚠️';
        font-size: 1.5rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(6, 167, 125, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 2px solid #06a77d;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        font-weight: 700;
        color: #065f46;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .risk-low::before {
        content: '✅';
        font-size: 1.5rem;
    }
    
    /* WORD HIGHLIGHTS */
    .word-fake {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        margin: 2px;
        display: inline-block;
        border: 1px solid #fca5a5;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(230, 57, 70, 0.1);
    }
    
    .word-fake:hover {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        transform: scale(1.08);
        box-shadow: 0 4px 8px rgba(230, 57, 70, 0.2);
    }
    
    .word-genuine {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px;
        display: inline-block;
        border: 1px solid #6ee7b7;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(6, 167, 125, 0.1);
    }
    
    .word-genuine:hover {
        background: linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%);
        transform: scale(1.08);
        box-shadow: 0 4px 8px rgba(6, 167, 125, 0.2);
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1rem;
        font-weight: 650;
        color: #cbd5e1;
        padding: 1rem 1.75rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6;
        border-bottom-color: #3b82f6;
    }
    
    /* INPUT FIELDS */
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #e63946;
        box-shadow: 0 0 0 4px rgba(230, 57, 70, 0.12);
    }
    
    /* BUTTONS */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border: none;
        padding: 0.85rem 1.75rem;
        font-weight: 700;
        border-radius: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(230, 57, 70, 0.35);
    }
    .stExpander:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 20px rgba(59,130,246,0.15);
}
    
    /* SIDEBAR */
    .sidebar-header {
        color: white;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid rgba(251, 191, 36, 0.3);
    }
    
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .sidebar-metric:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(251, 191, 36, 0.4);
    }
    
    .sidebar-metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #fbbf24;
        margin: 0.5rem 0;
    }
    
    .sidebar-metric-label {
        font-size: 0.75rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        font-weight: 600;
    }
    
    /* DATAFRAME */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
    }
    
    th {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        color: #f8fafc;
        font-weight: 700;
        padding: 1.2rem;
        text-align: left;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    td {
        padding: 1rem 1.2rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    tr:hover {
        background: #1e293b;
    }
    
    /* ANIMATIONS */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-header {
        animation: slideInDown 0.6s ease-out;
    }
    
    .animate-card {
        animation: fadeInUp 0.6s ease-out;
    }
            
    input::placeholder,
    textarea::placeholder {
    color: #94a3b8 !important;
}
    section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

div[data-baseweb="input"] input {
    color: #f8fafc !important;
}

textarea {
    color: #f8fafc !important;
}

label {
    color: #f8fafc !important;
}
    details summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.6rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
    padding: 0.8rem 1rem !important;
}

details summary p {
    margin: 0 !important;
}

details summary::marker {
    color: #3b82f6 !important;
}
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load ML model and vectorizer (cached)."""
    try:
        model = joblib.load(cfg.MODEL_PATH)
        vectorizer = joblib.load(cfg.VECTORIZER_PATH)
        return model, vectorizer, True
    except (FileNotFoundError, OSError):
        return None, None, False


@st.cache_resource
def load_explainer(_model, _vectorizer):
    """Load SHAP explainer (cached). Underscore args are not hashed by Streamlit."""
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        exp = SHAPExplainer(
            model=_model,
            vectorizer=_vectorizer,
            feature_names=list(_vectorizer.get_feature_names_out())
        )
        exp.build_explainer()
        return exp
    except Exception as e:
        return None


cleaner = TextCleaner()
model, vectorizer, models_loaded = load_models()
kg = RealTimeSkillKnowledgeGraph()

# ═══════════════════════════════════════════════════════════════
# BACKEND (FastAPI) CONNECTION
# ═══════════════════════════════════════════════════════════════
# The dashboard talks to the FastAPI backend (main.py) when it is reachable.
# If it is not, and local model files exist, it falls back to running the model
# inside this process (JOBGUARD_MODE=auto). Set JOBGUARD_MODE=api to require
# the backend, or JOBGUARD_MODE=local to never call it.

client = JobGuardClient(cfg.API_URL, cfg.API_KEY, timeout=30)


@st.cache_data(ttl=20, show_spinner=False)
def backend_status(url: str, mode: str) -> dict:
    """Health of the FastAPI backend (cached for 20 s so reruns stay fast)."""
    if mode == "local":
        return {"reachable": False, "reason": "Local mode (JOBGUARD_MODE=local)"}
    try:
        return {"reachable": True, **client.health()}
    except ApiError as exc:
        return {"reachable": False, "reason": str(exc)}


BACKEND = backend_status(cfg.API_URL, cfg.MODE)
USE_API = bool(BACKEND.get("reachable")) and bool(BACKEND.get("model_loaded"))

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    if prob >= 0.5:
        return "MEDIUM"
    if prob >= 0.3:
        return "LOW"
    return "SAFE"


def predict(text: str, fields: dict = None, explain: bool = True) -> dict:
    """Score one posting.

    Uses the FastAPI backend when it is reachable, otherwise the local model.
    `fields` (title, company_profile, description, ...) can carry the form
    inputs; without it the whole text is sent as the description.
    """
    if USE_API:
        payload = dict(fields or {})
        payload.setdefault("description", text)
        try:
            return client.predict(payload)
        except ApiError as exc:
            if exc.status_code is not None or cfg.MODE == "api" or not models_loaded:
                st.error(f"❌ {exc}")
                st.stop()
            st.session_state["api_fallback"] = str(exc)  # backend dropped: run locally

    if not models_loaded:
        if cfg.DEMO_MODE:  # explicit opt-in only: this returns RANDOM output
            import random
            prob = random.uniform(0.1, 0.95)
            return {
                "prediction": "FAKE" if prob > 0.5 else "GENUINE",
                "probability_fake": prob,
                "confidence": abs(prob - 0.5) * 2,
                "trust_score": int((1 - prob) * 100),
                "risk_level": _risk_level(prob),
                "top_fake_words": [], "top_genuine_words": [],
                "highlighted_html": text[:500], "demo_mode": True,
            }
        st.error("❌ No model available. Start the API or add the model files to `models/`.")
        st.stop()

    cleaned = cleaner.clean(text)
    X = vectorizer.transform([cleaned])
    prob = float(model.predict_proba(X)[0][1])
    prediction = "FAKE" if prob >= 0.5 else "GENUINE"

    top_fake, top_genuine = [], []
    if explain:
        explainer = load_explainer(model, vectorizer)
        if explainer is not None:
            try:
                shap_result = explainer.explain_single(text, cleaned)
                if prediction == "FAKE":
                    top_fake = shap_result.get("top_fake_words", [])[:10]
                else:
                    top_genuine = shap_result.get("top_real_words", [])[:10]
            except Exception:
                pass  # explanation is optional; the verdict still stands

    return {
        "prediction": prediction,
        "probability_fake": round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "trust_score": int((1 - prob) * 100),
        "risk_level": _risk_level(prob),
        "top_fake_words": top_fake,
        "top_genuine_words": top_genuine,
        "highlighted_html": text[:500],
        "demo_mode": False
    }


def ai_source_caption(source: str) -> str:
    """Tell the user WHY they are seeing the rule-based summary instead of Gemini."""
    if source in ("gemini", "cache"):
        return "Source: Gemini"
    if source == "no_key":
        return ("Source: rule-based summary. No Gemini key found: add GEMINI_API_KEY to your .env "
                "(or Streamlit secrets) and restart the app / API.")
    return ("Source: rule-based summary. The Gemini request failed or timed out "
            "(the reason is printed in the terminal running the app / API).")


def run_ai_analysis(job: dict, probability_fake: float, verify_company: bool = False) -> dict:
    """Rule-engine facts + Gemini (or rule-based) summary, via the API when possible.

    `job` keys: title, company, location, salary_range, description,
    requirements, benefits, experience.
    Returns {"analysis": str, "source": str, "facts": dict, "red_flags": list}.
    """
    if USE_API:
        try:
            data = client.analyze({
                "title": job.get("title", ""),
                "company_profile": job.get("company", ""),
                "description": job.get("description", ""),
                "requirements": job.get("requirements", ""),
                "benefits": job.get("benefits", ""),
                "salary_range": job.get("salary_range", ""),
                "location": job.get("location", ""),
                "experience": job.get("experience", ""),
                "verify_company": verify_company,
            })
            return {"analysis": data["analysis"], "source": data["analysis_source"],
                    "facts": data.get("facts", {}), "red_flags": data.get("red_flags", [])}
        except ApiError as exc:
            if exc.status_code is not None or cfg.MODE == "api":
                st.error(f"❌ {exc}")
                return {}
            st.session_state["api_fallback"] = str(exc)

    posting = JobPostingLite(**{k: (job.get(k) or "") for k in (
        "title", "company", "location", "salary_range", "description",
        "requirements", "benefits", "experience")})
    facts = analyze_job(posting, verify_company=verify_company)
    ai = analyze_job_risk(posting, facts, probability_fake=probability_fake)
    return {"analysis": ai["analysis"], "source": ai["source"],
            "facts": {k: v for k, v in facts.items() if k != "red_flags"},
            "red_flags": facts["red_flags"]}


def run_bulk(df: pd.DataFrame, raw_bytes: bytes, filename: str) -> dict:
    """Score every row of an uploaded CSV (API batch endpoint, else local loop)."""
    results_df = df.copy()
    probs = None

    if USE_API:
        try:
            out = client.predict_batch(raw_bytes, filename)
            if len(out) == len(df):
                # the API appends: prediction, probability_fake, trust_score, risk_level
                res = out.iloc[:, -4:]
                results_df["PREDICTION"] = res.iloc[:, 0].map({"REAL": "GENUINE"}).fillna(res.iloc[:, 0]).values
                probs = res.iloc[:, 1].astype(float).tolist()
                results_df["FRAUD_RISK_%"] = [round(p * 100, 1) for p in probs]
                results_df["TRUST_SCORE"] = res.iloc[:, 2].astype(int).values
        except ApiError as exc:
            if exc.status_code is not None or cfg.MODE == "api" or not models_loaded:
                st.error(f"❌ {exc}")
                return {}
            st.session_state["api_fallback"] = str(exc)

    if probs is None:  # local fallback (the original per-row loop)
        predictions = []
        progress_bar = st.progress(0)
        for i, (_, row) in enumerate(df.iterrows()):
            text_parts = []
            for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
                if col in df.columns:
                    text_parts.append(str(row.get(col, '') or ''))
            predictions.append(predict(' '.join(text_parts), explain=False))
            progress_bar.progress((i + 1) / len(df))
        results_df['PREDICTION'] = [r['prediction'] for r in predictions]
        results_df['FRAUD_RISK_%'] = [round(r['probability_fake'] * 100, 1) for r in predictions]
        results_df['TRUST_SCORE'] = [r['trust_score'] for r in predictions]
        probs = [r['probability_fake'] for r in predictions]

    # START TABLE FROM 1 INSTEAD OF 0
    results_df.index = range(1, len(results_df) + 1)
    n_fake = int((results_df['PREDICTION'] == 'FAKE').sum())
    return {"file": filename, "df": results_df, "n_fake": n_fake,
            "total": len(results_df), "avg_risk": float(np.mean(probs)) if probs else 0.0}


@st.cache_data(ttl=60, show_spinner=False)
def load_training_results(use_api: bool) -> dict:
    """Training metrics: from the API when connected, else from models/training_results.json."""
    if use_api:
        try:
            data = client.stats()
            if "models" in data:
                return data
        except ApiError:
            pass
    try:
        import json
        with open(cfg.TRAINING_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def top_scam_terms(n: int = 10):
    """Terms the model links most strongly to fake postings (from its coefficients)."""
    if not models_loaded or not hasattr(model, "coef_"):
        return []
    names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    idx = np.argsort(coefs)[::-1][:n]
    return [(names[i], float(coefs[i])) for i in idx]


@st.cache_data(ttl=10, show_spinner=False)
def read_history(limit: int = 20):
    """Recent predictions logged by the API (HTTP), or straight from the SQLite file."""
    if USE_API:
        try:
            return client.history(limit)
        except ApiError:
            return []
    try:
        import sqlite3
        conn = sqlite3.connect(cfg.DB_PATH)
        try:
            rows = conn.execute(
                "SELECT timestamp, prediction, probability, title FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        finally:
            conn.close()
        return [{"timestamp": r[0], "prediction": r[1], "probability_fake": r[2], "title": r[3]} for r in rows]
    except Exception:
        return []


def trust_meter_chart(trust_score: int):
    """Professional gauge chart for trust score."""
    color = "#e63946" if trust_score < 40 else ("#fbbf24" if trust_score < 70 else "#06a77d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 32, 'family': 'Plus Jakarta Sans, sans-serif'}, 'suffix': '/100'},
        title={'text': "TRUST SCORE", 'font': {'size': 20, 'family': 'Plus Jakarta Sans, sans-serif', 'color': '#111827'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#111827'},
            'bar': {'color': color, 'thickness': 0.25},
            'steps': [
                {'range': [0, 40], 'color': "#0a0101"},
                {'range': [40, 70], 'color': "#0B0901"},
                {'range': [70, 100], 'color': "#0b130e"},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': trust_score
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'family': 'Inter, sans-serif'}
    )
    return fig


def shap_bar_chart(result: dict):
    """Professional bar chart for SHAP word contributions."""
    fake_words = result.get("top_fake_words", [])[:8]
    genuine_words = result.get("top_genuine_words", [])[:8]

    if not fake_words and not genuine_words:
        return None

    words = [w for w, _ in fake_words] + [w for w, _ in genuine_words]
    scores = [s for _, s in fake_words] + [-s for _, s in genuine_words]
    colors = ["#e63946"] * len(fake_words) + ["#06a77d"] * len(genuine_words)

    fig = go.Figure(go.Bar(
        x=scores,
        y=words,
        orientation='h',
        marker_color=colors,
        text=[f"+{s:.3f}" if s > 0 else f"{s:.3f}" for s in scores],
        textposition='outside',
        marker={'line': {'width': 1.5, 'color': 'white'}}
    ))
    fig.update_layout(
        title={'text': "WORD CONTRIBUTION (SHAP VALUES)", 'font': {'size': 18, 'family': 'Plus Jakarta Sans, sans-serif', 'color': '#111827'}},
        xaxis_title="SHAP Score (→ FAKE  |  ← GENUINE)",
        height=380,
        margin=dict(l=20, r=80, t=60, b=20),
        xaxis=dict(zeroline=True, zerolinewidth=3, zerolinecolor="#070c14"),
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif', 'size': 12}
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-header">🛡️ JobGuard Pro</div>', unsafe_allow_html=True)

    if USE_API:
        st.success("✅ FastAPI backend connected")
        st.caption(
            f"{cfg.API_URL} · v{BACKEND.get('version', '?')} · "
            f"SHAP {'on' if BACKEND.get('explainer_loaded') else 'off'} · "
            f"Gemini {'on' if BACKEND.get('llm_configured') else 'off (rule-based fallback)'}"
        )
    elif models_loaded:
        st.warning("🟡 Local mode — backend not reachable")
        st.caption(f"{BACKEND.get('reason', '')}. Start it with `uvicorn main:app --reload`.")
    else:
        st.error("⚠️ Models not loaded")
        st.info("Add `models/best_model.joblib` and `models/tfidf_vectorizer.joblib`, "
                "or train them with `python -m src.ml.model_training`.")
    if st.session_state.get("api_fallback"):
        st.caption("⚠️ The backend dropped mid-session; using the local model.")
    if st.button("🔄 Re-check backend", key="recheck_backend"):
        backend_status.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 System Status")

    _best_f1_raw = load_training_results(USE_API).get("best_f1")
    _best_f1 = f"{_best_f1_raw:.2f}" if isinstance(_best_f1_raw, (int, float)) else "—"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{_best_f1}</div>
            <div class="sidebar-metric-label">Test F1-Score</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">17.8K</div>
            <div class="sidebar-metric-label">Dataset Size</div>
        </div>
        ''', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('''
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">866</div>
            <div class="sidebar-metric-label">Fake Jobs</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown('''
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">4.84%</div>
            <div class="sidebar-metric-label">Imbalance</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[📘 GitHub Repo](https://github.com/patilrakshita2203-prog/JobGuard)")
    st.markdown("[🔴 Report Cybercrime](https://cybercrime.gov.in)")
    st.markdown("[📞 NASSCOM Helpline](https://nasscom.in)")
    st.markdown("[📊 Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)")


# ═══════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner animate-header">
    <h1>🛡️ JobGuard Pro</h1>
    <p>Enterprise AI-Powered Fake Job Detection</p>
    <p style="font-size:0.95rem;opacity:0.8;margin-top:1rem;">
        NLP • TF-IDF + Logistic Regression • SHAP Explainability • FastAPI • PDF Reports
    </p>
    <div class="header-badge">v2.0 — Production Ready</div>
</div>
""", unsafe_allow_html=True)


# Stop early, with a clear message, when there is nothing to score with
if cfg.MODE == "api" and not USE_API:
    st.error(
        f"The dashboard is in API-only mode (JOBGUARD_MODE=api) but the FastAPI backend at "
        f"{cfg.API_URL} is not reachable. Start it with `uvicorn main:app --reload`, "
        f"or set JOBGUARD_MODE=auto to fall back to the local model."
    )
    st.stop()

if not USE_API and not models_loaded and not cfg.DEMO_MODE:
    st.error(
        "No model is available. Start the FastAPI backend (`uvicorn main:app --reload`) "
        "or add `models/best_model.joblib` and `models/tfidf_vectorizer.joblib` "
        "(train them with `python -m src.ml.model_training`)."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Analyzer",
    "📁 Bulk Analysis",
    "🌐 Live Scanner",
    "💰 Salary Checker",
    "📊 Analytics"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: SINGLE JOB ANALYZER
# ════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🔍 Analyze a Job Posting")
    st.markdown("Paste job details below and get an instant AI-powered fraud assessment with SHAP explanation.")

    col_input, col_output = st.columns([2, 1.5])

    with col_input:
        _demo = st.session_state.get('demo_data', {})
        with st.form("analyze_form", clear_on_submit=False):
            job_title = st.text_input("📋 Job Title", value=_demo.get("title", ""), placeholder="e.g., Software Engineer, Data Analyst")
            company = st.text_input("🏢 Company Name", value=_demo.get("company", ""), placeholder="e.g., Infosys, TCS")

            col_a, col_b = st.columns(2)
            with col_a:
                salary = st.text_input("💰 Salary/Stipend", value=_demo.get("salary", ""), placeholder="e.g., 6-9 LPA")
            with col_b:
                location = st.text_input("📍 Location", placeholder="e.g., Bangalore, Remote")

            description = st.text_area(
                "📝 Job Description *",
                height=200,
                value=_demo.get("description", ""),
                placeholder="Paste the complete job description here...",
            )

            requirements = st.text_area("Requirements (optional)", height=80, placeholder="Technical skills needed...")
            benefits = st.text_area("Benefits (optional)", height=60, placeholder="What you'll get...")

            col_c, col_d = st.columns(2)
            with col_c:
                has_logo = st.checkbox("✓ Company has official logo", value=False)
            with col_d:
                work_from_home = st.checkbox("✓ Work from home offered", value=False)

            col_demo1, col_demo2, col_analyze = st.columns([1, 1, 1.5])
            demo1 = col_demo1.form_submit_button("🚨 Scam Demo")
            demo2 = col_demo2.form_submit_button("✅ Genuine Demo")
            analyze_btn = col_analyze.form_submit_button("ANALYZE JOB", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 📊 Results")
        results_placeholder = st.empty()
        results_placeholder.info("📌 Fill job details and analyze...")

    # DEMO DATA
    FAKE_DEMO = {
        "title": "Work From Home — Earn ₹50,000/month (Urgent)",
        "company": "FastMoney Solutions",
        "salary": "50000-80000",
        "description": """URGENT HIRING! No experience needed. Registration fee of ₹500 required to start immediately. Investment required to unlock your account. 100% job guarantee! Direct joining. Limited seats. WhatsApp immediately. Join Telegram group. Security deposit ₹1000. Free laptop provided after payment. Earn unlimited!"""
    }

    GENUINE_DEMO = {
        "title": "Senior Python Developer",
        "company": "Infosys Technologies Ltd",
        "salary": "12-18 LPA",
        "description": """We seek an experienced Python developer for our backend team. Responsibilities: Design REST APIs, work with PostgreSQL, collaborate in Agile environment, write tested code, participate in code reviews. Requirements: 3-5 years experience, Python 3.x, SQL databases, Git knowledge, strong problem-solving. We offer: Competitive salary, health insurance, PF, learning budget, hybrid work. Interview process: Online coding test → Technical → HR"""
    }

    if demo1:
        st.session_state['demo_data'] = FAKE_DEMO
        st.rerun()
    elif demo2:
        st.session_state['demo_data'] = GENUINE_DEMO
        st.rerun()

    if 'demo_data' in st.session_state:
        demo = st.session_state['demo_data']
        st.success("✨ Demo data loaded! Click ANALYZE to see results.")

    if analyze_btn and description:
        combined = f"{job_title} {company} {description} {requirements} {benefits}"
        with st.spinner("Analyzing..."):
            result = predict(combined, fields={
                "title": job_title, "company_profile": company, "description": description,
                "requirements": requirements, "benefits": benefits,
                "salary_range": salary, "location": location,
                "telecommuting": int(work_from_home), "has_company_logo": int(has_logo),
            })
            # Knowledge Graph Analysis
            kg_result = kg.analyze_job_posting(
                job_title=job_title,
                job_description=description,
                claimed_salary=0
            )
        # keep everything in session_state: buttons inside the results (PDF,
        # AI analysis) trigger a rerun, which would otherwise wipe the results
        st.session_state['single'] = {
            "result": result,
            "kg_result": kg_result,
            "red_flags": result.get("red_flags") or detect_red_flags(combined, company),
            "job": {"title": job_title, "company": company, "location": location,
                    "salary_range": salary, "description": description,
                    "requirements": requirements, "benefits": benefits},
            "ai": None,
        }
    elif analyze_btn:
        st.warning("⚠️ Please enter at least the Job Description.")

    _single = st.session_state.get('single')
    if _single:
        result = _single["result"]
        kg_result = _single["kg_result"]

        with results_placeholder.container():
            # VERDICT
            if result["prediction"] == "FAKE":
                st.markdown('<div class="verdict-fake">⚠️ LIKELY SCAM</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-genuine">✅ APPEARS SAFE</div>', unsafe_allow_html=True)

            # TRUST METER
            st.plotly_chart(trust_meter_chart(result["trust_score"]), use_container_width=True)

            # METRICS
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{result['confidence']*100:.1f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                ''', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{result['probability_fake']*100:.1f}%</div>
                    <div class="metric-label">Fraud Risk</div>
                </div>
                ''', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{result['trust_score']}/100</div>
                    <div class="metric-label">Trust Score</div>
                </div>
                ''', unsafe_allow_html=True)

            # RISK LEVEL
            prob = result["probability_fake"]
            if prob >= 0.8:
                st.markdown('<div class="risk-high"><b>HIGH RISK</b> — Very likely a scam. Do NOT pay fees or share personal data.</div>', unsafe_allow_html=True)
            elif prob >= 0.5:
                st.markdown('<div class="risk-medium"><b>MEDIUM RISK</b> — Suspicious. Verify independently before proceeding.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low"><b>LOW RISK</b> — Appears legitimate. Verify official channels before applying.</div>', unsafe_allow_html=True)

            # KNOWLEDGE GRAPH ANALYSIS
            st.markdown("---")
            st.markdown("### 🧠 Skill Validation")

            if kg_result["role_detected"]:
                st.info(f"🎯 Detected Role: {kg_result['role_detected']}")
                skill_analysis = kg_result["skill_analysis"]
                colkg1, colkg2 = st.columns(2)

                with colkg1:
                    st.markdown("#### ✅ Expected Skills Found")
                    if skill_analysis["found_skills"]:
                        for skill in skill_analysis["found_skills"]:
                            st.markdown(f'<span class="word-genuine">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.warning("No expected skills detected.")

                with colkg2:
                    st.markdown("#### 🚨 Suspicious Skills")
                    if skill_analysis["mismatched_skills"]:
                        for skill in skill_analysis["mismatched_skills"]:
                            st.markdown(f'<span class="word-fake">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.success("No suspicious skill mismatch found.")

                st.metric("Skill Coverage", f"{skill_analysis['skill_coverage']*100:.1f}%")

                if skill_analysis["is_suspicious"]:
                    st.error("⚠️ Job skills do NOT match the detected role.")
                else:
                    st.success("✅ Skills appear consistent with the role.")
            else:
                st.warning("Role could not be detected.")

            # SHAP EXPLANATION
            if result.get("top_fake_words") or result.get("top_genuine_words"):
                st.markdown("---")
                st.markdown("### 🧠 AI Explanation — Key Indicators")

                chart = shap_bar_chart(result)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown("**🚨 Scam Indicators:**")
                    for word, score in result.get("top_fake_words", [])[:8]:
                        st.markdown(f'<span class="word-fake">{word}</span>', unsafe_allow_html=True)
                with exp_col2:
                    st.markdown("**✅ Trust Indicators:**")
                    for word, score in result.get("top_genuine_words", [])[:8]:
                        st.markdown(f'<span class="word-genuine">{word}</span>', unsafe_allow_html=True)

            # RED FLAGS (rule-based, English + Hindi keywords)
            st.markdown("---")
            st.markdown("### 🚩 Scam Red Flags")
            _flags = _single.get("red_flags") or []
            if _flags:
                for _f in _flags:
                    _icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(_f["severity"], "🟡")
                    st.markdown(f"{_icon} **{_f['label']}** — _“{_f['evidence']}”_")
            else:
                st.success("No rule-based scam red flags found in the text.")

            # AI RISK ANALYSIS (Gemini, with a rule-based fallback)
            st.markdown("### 🧠 AI Risk Analysis")
            _verify = st.checkbox("Also verify the company website online (slower)", key="verify_company_single")
            if st.button("Generate AI Risk Analysis", key="ai_single"):
                with st.spinner("Generating AI analysis..."):
                    _single["ai"] = run_ai_analysis(_single["job"], result["probability_fake"], verify_company=_verify)
            if _single.get("ai"):
                st.markdown(_single["ai"]["analysis"])
                st.caption(ai_source_caption(_single["ai"]["source"]))

            # PDF REPORT
            st.markdown("---")
            if st.button("📄 Generate PDF Report", use_container_width=False):
                try:
                    from src.reporting.pdf_reporter import generate_pdf_report
                    job_data = {
                        "title": job_title, "company": company,
                        "salary_range": salary, "location": location,
                        "description": description, "requirements": requirements
                    }
                    pdf_path = generate_pdf_report(job_data, result)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            f.read(),
                            file_name=Path(pdf_path).name,
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"PDF generation error: {e}")



# ════════════════════════════════════════════════════════════════
# TAB 2: BULK ANALYZER
# ════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 📁 Bulk Job Analysis")
    st.markdown("Upload a CSV file to analyze multiple job postings at once.")

    with st.expander("📋 CSV Format Guide"):
        st.markdown("""
        **Required column:** `description`
        
        **Optional columns:** `title`, `company_profile`, `requirements`, `benefits`, `salary_range`
        
        **Example:**
        ```
        title,company_profile,description
        Software Engineer,Tech Corp,"Python backend engineer needed..."
        Data Analyst,Analytics Inc,"We're hiring a data analyst..."
        ```
        """)

    uploaded_file = st.file_uploader("📤 Upload CSV File", type=['csv'])

    if uploaded_file:
        raw_bytes = uploaded_file.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            st.error(f"❌ Could not read this CSV: {e}")
            df = None

        if df is not None:
            st.info(f"📊 Loaded {len(df)} job postings")

            with st.expander("Preview", expanded=True):
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=500
                )

            if 'description' not in df.columns:
                st.error("❌ CSV must have 'description' column")

            elif st.button("🚀 Analyze All Jobs", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {len(df)} postings..."):
                    st.session_state['bulk'] = run_bulk(df, raw_bytes, uploaded_file.name)

    # results live in session_state so the download button does not wipe them
    bulk = st.session_state.get('bulk')
    if bulk and uploaded_file and bulk.get("file") == uploaded_file.name:
        results_df = bulk["df"]
        n_fake = bulk["n_fake"]

        st.success(f"✅ Analysis Complete — {n_fake} suspicious jobs found")

        st.dataframe(
            results_df,
            use_container_width=True,
            height=500
        )
        # SUMMARY STATS
        st.markdown("---")
        st.markdown("### 📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="metric-card"><div class="metric-value">{bulk["total"]}</div><div class="metric-label">Total Analyzed</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#e63946;">{n_fake}</div><div class="metric-label">Scams Found</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#06a77d;">{bulk["total"]-n_fake}</div><div class="metric-label">Safe Jobs</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><div class="metric-value">{bulk["avg_risk"]*100:.1f}%</div><div class="metric-label">Avg Risk</div></div>', unsafe_allow_html=True)

        # CHART
        verdict_data = pd.DataFrame({
            'Status': ['Safe', 'Scam'],
            'Count': [bulk["total"] - n_fake, n_fake]
        })
        fig = px.pie(
            verdict_data, values='Count', names='Status',
            color_discrete_map={'Scam': '#e63946', 'Safe': '#06a77d'},
            title='Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

        # DETAILED TABLE
        st.markdown("### 🔍 Detailed Results")
        display_cols = ['PREDICTION', 'FRAUD_RISK_%', 'TRUST_SCORE'] + [c for c in ['title', 'description'] if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True)

        # DOWNLOAD
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv_buffer.getvalue(),
            file_name=f"jobguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ════════════════════════════════════════════════════════════════
# TAB 3: LIVE SCANNER
# ════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🌐 Live Portal Scanner")
    st.markdown("Scrape and analyze live job postings from Naukri.com in real-time.")

    col1, col2, col3 = st.columns(3)
    keyword = col1.text_input("🔍 Keyword", value="fresher python", placeholder="Job role to search")
    platform = col2.selectbox("📱 Platform", ["Naukri.com", "Internshala"])
    max_jobs = col3.slider("📊 Max Jobs", 5, 25, 10)

    if not cfg.ENABLE_SCRAPING:
        st.info(
            "Live scanning is switched off on this deployment because it needs a local Chrome browser. "
            "Run JobGuard on your machine to use it, or set `JOBGUARD_ENABLE_SCRAPING=true`."
        )

    elif st.button("🔴 START SCAN", type="primary", use_container_width=True):

        with st.spinner(f"Scraping {platform}..."):

            try:

                if platform == "Naukri.com":
                    from src.scraper.job_scraper import NaukriScraper
                    scraper = NaukriScraper()

                else:
                    from src.scraper.internshala_scraper import InternshalaScraper
                    scraper = InternshalaScraper()

                jobs = scraper.search_jobs(
                    keyword=keyword,
                    max_jobs=max_jobs
                )

                live_results = []
                probs = []

                for job in jobs:

                    text = job.combined_text()
                    result = predict(text, explain=False)
                    probs.append(result['probability_fake'])

                    live_results.append({
                        "Title": job.title[:45],
                        "Company": job.company[:35],
                        "Location": job.location,
                        "Salary": job.salary_range or "—",
                        "Verdict": "🚨 SCAM" if result['prediction'] == "FAKE" else "✅ SAFE",
                        "Risk Score": f"{result['probability_fake']*100:.0f}%",
                        "Trust Score": f"{100 - (result['probability_fake']*100):.0f}%",
                    })

                st.session_state['scan'] = {
                    "jobs": jobs, "probs": probs, "platform": platform,
                    "df": pd.DataFrame(live_results),
                }
                st.session_state.pop('scan_ai', None)

            except Exception as e:
                st.error(
                    f"Scraping error: {e}\n\n"
                    "Live scanning needs Selenium and a local Chrome browser, and portals may block automated access."
                )

    scan = st.session_state.get('scan')
    if scan and scan["jobs"]:
        jobs = scan["jobs"]
        st.success(f"✅ Scraped {len(jobs)} jobs from {scan['platform']}")
        st.dataframe(scan["df"], width="stretch")

        st.markdown("---")
        st.subheader("🧠 AI Risk Analysis")
        selected_job = st.selectbox(
            "Select a job",
            range(len(jobs)),
            format_func=lambda i: f"{jobs[i].title} — {jobs[i].company}",
            key="scan_job_idx"
        )
        if st.button("🔍 Generate AI Risk Analysis", key="scan_ai_btn"):
            with st.spinner("Generating AI analysis..."):
                j = jobs[selected_job]   # the job picked in the dropdown
                ai = run_ai_analysis({
                    "title": j.title, "company": j.company, "location": j.location,
                    "salary_range": j.salary_range, "description": j.description,
                    "requirements": getattr(j, "requirements", ""),
                    "benefits": getattr(j, "benefits", ""),
                    "experience": getattr(j, "experience", ""),
                }, scan["probs"][selected_job], verify_company=True)
            st.session_state['scan_ai'] = (selected_job, ai)

        saved = st.session_state.get('scan_ai')
        if saved and saved[0] == selected_job and saved[1]:
            st.markdown(saved[1]["analysis"])
            st.caption(ai_source_caption(saved[1]["source"]))


# ════════════════════════════════════════════════════════════════
# TAB 4: SALARY CHECKER
# ════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 💰 Salary Anomaly Detector")
    st.markdown("Check if salary is realistic for the job role and experience level.")

    BENCHMARKS = {
        "Software Engineer": {"fresher": (350000, 900000), "exp": (800000, 2500000)},
        "Data Analyst": {"fresher": (300000, 700000), "exp": (700000, 1800000)},
        "Data Scientist": {"fresher": (500000, 1200000), "exp": (1200000, 3000000)},
        "Product Manager": {"fresher": (600000, 1500000), "exp": (1500000, 4000000)},
        "Marketing": {"fresher": (200000, 500000), "exp": (500000, 1200000)},
    }

    col_l, col_r = st.columns([1.5, 2])

    with col_l:
        role = st.selectbox("Job Role", list(BENCHMARKS.keys()))
        exp = st.radio("Experience", ["Fresher", "Experienced"], horizontal=True)
        sal_min = st.number_input("Claimed Min (₹)", value=600000, step=100000)
        sal_max = st.number_input("Claimed Max (₹)", value=900000, step=100000)

        if st.button("🔍 ANALYZE", use_container_width=True):
            exp_key = "fresher" if exp == "Fresher" else "exp"
            bench_min, bench_max = BENCHMARKS[role][exp_key]

            with col_r:
                st.markdown(f"### {role} Salary Check")

                # VISUALIZATION
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Market Range",
                    x=["Market", "Offered"],
                    y=[(bench_min + bench_max) / 2, (sal_min + sal_max) / 2],
                    error_y=dict(
                        type='data',
                        array=[(bench_max - bench_min) / 2, (sal_max - sal_min) / 2],
                        visible=True
                    ),
                    marker_color=['#06a77d', '#e63946']
                ))
                fig.update_layout(height=300, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**📊 Market Benchmark:** ₹{bench_min:,.0f} – ₹{bench_max:,.0f}")
                st.markdown(f"**💼 Claimed Salary:** ₹{sal_min:,.0f} – ₹{sal_max:,.0f}")

                if sal_max > bench_max * 2:
                    st.markdown('<div class="risk-high"><b>ANOMALY!</b> Salary is 2x+ above market. Classic scam tactic.</div>', unsafe_allow_html=True)
                elif sal_max > bench_max * 1.3:
                    st.markdown('<div class="risk-medium"><b>SUSPICIOUS</b> — Above market. Verify independently.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low"><b>REALISTIC</b> — Within normal market range.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 5: ANALYTICS
# ════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 📊 System Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🤖 Model Performance")
        results = load_training_results(USE_API)
        if results.get("models"):
            st.caption(f"Best model on the held-out test set: {results.get('best_model', '—')}")
            for m in results["models"]:
                with st.expander(f"📈 {m['model']}"):
                    mc1, mc2 = st.columns(2)
                    mc1.metric("F1-Score", f"{m.get('f1_score', 0):.3f}")
                    mc2.metric("AUC-ROC", f"{m.get('roc_auc', 0):.3f}")
                    mc3, mc4 = st.columns(2)
                    mc3.metric("Precision", f"{m.get('precision', 0):.3f}")
                    mc4.metric("Recall", f"{m.get('recall', 0):.3f}")
        else:
            st.info("Train models first to see performance.")

    with col2:
        st.markdown("#### 🔴 Top Scam-Indicating Terms")
        _terms = top_scam_terms(10)
        if _terms:
            fig = px.bar(
                x=[w for _, w in _terms][::-1],
                y=[t for t, _ in _terms][::-1],
                orientation='h',
                color=[w for _, w in _terms][::-1],
                color_continuous_scale='Reds',
                title="Strongest fake-job terms (model coefficients)"
            )
            fig.update_layout(height=350, showlegend=False, template='plotly_white',
                              coloraxis_showscale=False, xaxis_title="Coefficient", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Term weights are available when a linear model is loaded locally.")

    st.markdown("#### 🕒 Recent Predictions")
    _history = read_history(20)
    if _history:
        st.dataframe(pd.DataFrame(_history), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet. Analyze a job with the API running to see history here.")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
---
<div style="text-align:center;color:#6b7280;font-size:0.85rem;padding:2rem 0;">
    <p><b>🛡️ JobGuard Pro v2.0</b> — Enterprise AI Fraud Detection</p>
    <p>Built for RV College of Engineering | CSE 4th Semester</p>
    <p style="font-size:0.8rem;margin-top:0.5rem;">
        Dataset: EMSCAD (17,880 postings) | Model: TF-IDF + Logistic Regression | Explainability: SHAP
    </p>
    <p style="font-size:0.8rem;opacity:0.7;">
        ⚠️ This tool is for educational purposes. Always verify jobs through official channels.
    </p>
</div>
""", unsafe_allow_html=True)
