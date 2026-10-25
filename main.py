"""
JobGuard FastAPI backend.

Endpoints
    GET  /              service info
    GET  /health        liveness + model status
    POST /predict       score one job posting (+ SHAP words, red flags)
    POST /predict/batch score a CSV of postings, returns a CSV
    POST /analyze       rule-engine facts + red flags + Gemini (or rule-based) summary
    GET  /stats         training metrics (models/training_results.json)
    GET  /history       recent predictions logged by the API

Run:  uvicorn main:app --reload        (docs at http://127.0.0.1:8000/docs)
"""

import io
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Fix import paths (keep the original behaviour: run from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src import config as cfg  # noqa: E402
from src.preprocessing.data_cleaning import TextCleaner  # noqa: E402
from src.explainability.shap_explainer import SHAPExplainer  # noqa: E402
from src.explainability.job_posting import JobPostingLite  # noqa: E402
from src.explainability.red_flags import detect_red_flags  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "1.1.0"

model = None
vectorizer = None
explainer = None
cleaner = TextCleaner()


# ───────────────────────── startup / shutdown ─────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield


def load_models():
    """Load model + vectorizer (required) and the SHAP explainer (optional)."""
    global model, vectorizer, explainer

    try:
        model = joblib.load(cfg.MODEL_PATH)
        vectorizer = joblib.load(cfg.VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("Model files not found, /predict will return 503: %s", exc)
        model = vectorizer = explainer = None
        return

    try:
        explainer = SHAPExplainer(
            model=model,
            vectorizer=vectorizer,
            feature_names=list(vectorizer.get_feature_names_out()),
        )
        explainer.build_explainer()
    except Exception as exc:  # SHAP is a nice-to-have; predictions still work
        logger.warning("SHAP explainer unavailable, predictions will have no word scores: %s", exc)
        explainer = None


app = FastAPI(
    title="JobGuard API",
    version=API_VERSION,
    description="Fake job posting detection with explainable ML, red-flag rules and AI risk analysis.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── security helpers ─────────────────────────

def require_api_key(request: Request):
    """If JOBGUARD_API_KEY is set, every protected endpoint needs `X-API-Key`."""
    if cfg.API_KEY and request.headers.get("x-api-key") != cfg.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class _RateLimiter:
    """Tiny in-memory sliding-window limiter (per client IP, per bucket)."""

    def __init__(self):
        self._hits = defaultdict(deque)

    def check(self, request: Request, bucket: str, limit: int):
        if limit <= 0:
            return
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[(client, bucket)]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded, try again in a minute")
        window.append(now)


_limiter = _RateLimiter()


def limit_predict(request: Request):
    _limiter.check(request, "predict", cfg.RATE_LIMIT_PER_MIN)


def limit_batch(request: Request):
    _limiter.check(request, "batch", cfg.BATCH_RATE_LIMIT_PER_MIN)


# ───────────────────────── schemas ─────────────────────────

class JobInput(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str
    requirements: str = ""
    benefits: str = ""
    salary_range: Optional[str] = None
    location: Optional[str] = None
    telecommuting: int = 0
    has_company_logo: int = 0


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probability_fake: float
    trust_score: int
    top_fake_words: list
    top_genuine_words: list
    highlighted_html: str
    risk_level: str
    timestamp: str
    # Added in v1.1 (optional, so older clients keep working)
    red_flags: list = []
    explanation: dict = {}


class AnalyzeInput(JobInput):
    experience: str = ""
    verify_company: bool = False  # live web search is slow, so opt-in
    use_llm: bool = True


# ───────────────────────── core logic ─────────────────────────

def _risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    if prob >= 0.5:
        return "MEDIUM"
    if prob >= 0.3:
        return "LOW"
    return "SAFE"


def _raw_text(job: JobInput) -> str:
    return " ".join([job.title, job.company_profile, job.description, job.requirements, job.benefits])


def _dump(model_obj) -> dict:
    return model_obj.model_dump() if hasattr(model_obj, "model_dump") else model_obj.dict()


def _require_model():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _score(raw_text: str) -> dict:
    """Return prediction details, with SHAP words when the explainer is available."""
    cleaned = cleaner.clean(raw_text)

    if explainer is not None:
        try:
            result = explainer.explain_single(raw_text, cleaned)
            # shap_explainer names the second list `top_real_words`
            result["top_genuine_words"] = result.get("top_genuine_words") or result.get("top_real_words", [])
            return result
        except Exception as exc:  # e.g. a non-linear model the explainer cannot handle
            logger.warning("SHAP explanation failed, returning the plain prediction: %s", exc)

    prob = float(model.predict_proba(vectorizer.transform([cleaned]))[0][1])
    prediction = "FAKE" if prob >= 0.5 else "REAL"
    return {
        "prediction": prediction,
        "confidence": round(prob if prediction == "FAKE" else 1 - prob, 4),
        "probability_fake": round(prob, 4),
        "top_fake_words": [],
        "top_genuine_words": [],
    }


# ───────────────────────── endpoints ─────────────────────────

@app.get("/")
def root():
    return {"service": "JobGuard API", "version": API_VERSION, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    llm_configured = False
    try:
        from src.explainability.llm_analyzer import is_configured

        llm_configured = is_configured()
    except Exception:  # google-generativeai missing etc.
        pass

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "llm_configured": llm_configured,
        "auth_required": bool(cfg.API_KEY),
        "version": API_VERSION,
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    dependencies=[Depends(require_api_key), Depends(limit_predict)],
)
def predict(job: JobInput):
    # Plain `def` so FastAPI runs the CPU-heavy SHAP work in a worker thread.
    _require_model()

    try:
        raw_text = _raw_text(job)
        result = _score(raw_text)
        prob = result["probability_fake"]
        is_fake = result["prediction"] == "FAKE"
        trust = int((1 - prob) * 100)
        risk = _risk_level(prob)

        # show only the relevant side of the explanation
        top_fake_words = result.get("top_fake_words", [])[:10] if is_fake else []
        top_genuine_words = [] if is_fake else result.get("top_genuine_words", [])[:10]

        fake_words = [w[0] for w in top_fake_words[:5]]
        genuine_words = [w[0] for w in top_genuine_words[:5]]

        if is_fake:
            highlighted_html = f"""
🚨 Fake Job Detected

🔴 Top Fake Indicators:
{', '.join(fake_words) if fake_words else 'None'}

📊 Risk Score:
{round(prob * 100, 2)}% likelihood of being fake

⚠️ Recommendation:
Avoid applying. This job shows strong scam indicators.
"""
            recommendation = "Avoid applying. This job shows strong scam indicators."
        else:
            highlighted_html = f"""
✅ Genuine Job Detected

🟢 Trust Indicators:
{', '.join(genuine_words) if genuine_words else 'Low risk, no strong fake signals detected'}

📊 Confidence:
{round((1 - prob) * 100, 2)}% chance of being genuine

💡 Recommendation:
This job appears safe, but always verify company details.
"""
            recommendation = "This job appears safe, but always verify company details."

        red_flags = detect_red_flags(raw_text, job.company_profile)

        _log(_dump(job), result)

        return PredictionResult(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probability_fake=prob,
            trust_score=trust,
            top_fake_words=top_fake_words,
            top_genuine_words=top_genuine_words,
            highlighted_html=highlighted_html,
            risk_level=risk,
            timestamp=datetime.now().isoformat(),
            red_flags=red_flags,
            explanation={
                "verdict": "SCAM" if is_fake else "SAFE",
                "recommendation": recommendation,
                "top_fake_words": top_fake_words,
                "top_genuine_words": top_genuine_words,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post(
    "/predict/batch",
    dependencies=[Depends(require_api_key), Depends(limit_batch)],
)
async def predict_batch(file: UploadFile = File(...)):
    _require_model()

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    content = await file.read(cfg.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > cfg.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {cfg.MAX_UPLOAD_MB} MB)")

    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8-sig")))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if "description" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing description column")

    if len(df) > cfg.MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows (max {cfg.MAX_BATCH_ROWS})")

    for col in ["title", "company_profile", "requirements", "benefits"]:
        if col not in df.columns:
            df[col] = ""

    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    try:
        cleaned = [cleaner.clean(t) for t in texts]
        probs = model.predict_proba(vectorizer.transform(cleaned))[:, 1] if cleaned else []
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")

    results = [
        {
            "prediction": "FAKE" if p >= 0.5 else "REAL",
            "probability_fake": round(float(p), 4),
            "trust_score": int((1 - float(p)) * 100),
            "risk_level": _risk_level(float(p)),
        }
        for p in probs
    ]

    output_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    output = io.StringIO()
    output_df.to_csv(output, index=False)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


@app.post(
    "/analyze",
    dependencies=[Depends(require_api_key), Depends(limit_predict)],
)
def analyze(job: AnalyzeInput):
    """Rule-engine facts + red flags + AI (Gemini) or rule-based risk summary."""
    _require_model()

    try:
        from src.explainability.llm_analyzer import analyze_job_risk
        from src.explainability.rule_engine import analyze_job

        raw_text = _raw_text(job)
        prob = _score(raw_text)["probability_fake"]

        posting = JobPostingLite(
            title=job.title,
            company=job.company_profile,
            location=job.location or "",
            salary_range=job.salary_range or "",
            description=job.description,
            requirements=job.requirements,
            benefits=job.benefits,
            experience=job.experience,
        )
        facts = analyze_job(posting, verify_company=job.verify_company)
        ai = analyze_job_risk(posting, facts, probability_fake=prob, use_llm=job.use_llm)

        return {
            "probability_fake": prob,
            "risk_level": _risk_level(prob),
            "facts": {k: v for k, v in facts.items() if k != "red_flags"},
            "red_flags": facts["red_flags"],
            "analysis": ai["analysis"],
            "analysis_source": ai["source"],
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.get("/stats")
def stats():
    try:
        with open(cfg.TRAINING_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": "no stats"}


@app.get("/history", dependencies=[Depends(require_api_key)])
def history(limit: int = Query(20, ge=1, le=200)):
    """Most recent predictions logged by this API instance."""
    if not cfg.DB_PATH.exists():  # connecting would create an empty database file
        return []
    try:
        conn = sqlite3.connect(cfg.DB_PATH)
        try:
            rows = conn.execute(
                "SELECT timestamp, prediction, probability, title FROM predictions "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        rows = []

    return [
        {"timestamp": r[0], "prediction": r[1], "probability_fake": r[2], "title": r[3]}
        for r in rows
    ]


def _log(job_input: dict, result: dict):
    """Append one row to the SQLite history (best effort, never breaks a request)."""
    if not cfg.LOG_PREDICTIONS:
        return
    try:
        cfg.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cfg.DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    prediction TEXT,
                    probability REAL,
                    title TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO predictions (timestamp, prediction, probability, title) VALUES (?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    result["prediction"],
                    result["probability_fake"],
                    (job_input.get("title", "") or "")[:120],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        logger.debug("Could not log prediction: %s", exc)


if __name__ == "__main__":
    import uvicorn

    host = cfg.get_setting("JOBGUARD_HOST", "127.0.0.1")
    port = cfg.get_int("JOBGUARD_PORT", 8000)
    uvicorn.run("main:app", host=host, port=port, reload=cfg.get_bool("JOBGUARD_RELOAD", True))
