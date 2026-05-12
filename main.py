import logging
import json
from datetime import datetime
from typing import Optional
import joblib
import sys
import os

# Fix import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io

# Local imports
from data_cleaning import TextCleaner
from shap_explainer import SHAPExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JobGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None
explainer = None
cleaner = TextCleaner()


@app.on_event("startup")
async def load_models():
    global model, vectorizer, explainer
    try:
        model = joblib.load("models/best_model.joblib")
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

        explainer = SHAPExplainer(
            model=model,
            vectorizer=vectorizer,
            feature_names=list(vectorizer.get_feature_names_out())
        )
        explainer.build_explainer()

        logger.info("Models loaded successfully")

    except Exception as e:
        logger.warning(f"Model load error: {e}")


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(job: JobInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        raw_text = " ".join([
            job.title, job.company_profile, job.description,
            job.requirements, job.benefits
        ])

        cleaned = cleaner.clean(raw_text)

        result = explainer.explain_single(raw_text, cleaned)
        trust = explainer.get_trust_score(result)

        prob = result["probability_fake"]

        # Risk level
        if prob >= 0.8:
            risk = "HIGH"
        elif prob >= 0.5:
            risk = "MEDIUM"
        elif prob >= 0.3:
            risk = "LOW"
        else:
            risk = "SAFE"

        # FIX: show only relevant words
        if result["prediction"] == "FAKE":
            top_fake_words = result.get("top_fake_words", [])[:10]
            top_genuine_words = []
        else:
            top_fake_words = []
            top_genuine_words = result.get("top_genuine_words", [])[:10]

        fake_words = [w[0] for w in top_fake_words[:5]]
        genuine_words = [w[0] for w in top_genuine_words[:5]]

        # Clean explanation output
        if result["prediction"] == "FAKE":
            highlighted_html = f"""
🚨 Fake Job Detected

🔴 Top Fake Indicators:
{', '.join(fake_words) if fake_words else 'None'}

📊 Risk Score:
{round(prob * 100, 2)}% likelihood of being fake

⚠️ Recommendation:
Avoid applying. This job shows strong scam indicators.
"""
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

        _log(job.dict(), result)

        return PredictionResult(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probability_fake=result["probability_fake"],
            trust_score=trust,
            top_fake_words=top_fake_words,
            top_genuine_words=top_genuine_words,
            highlighted_html=highlighted_html,
            risk_level=risk,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    if "description" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing description column")

    for col in ["title", "company_profile", "requirements", "benefits"]:
        if col not in df.columns:
            df[col] = ""

    results = []

    for _, row in df.iterrows():
        raw = " ".join([
            str(row.get(c, "")) for c in
            ["title", "company_profile", "description", "requirements"]
        ])

        cleaned = cleaner.clean(raw)
        X = vectorizer.transform([cleaned])
        prob = float(model.predict_proba(X)[0][1])

        results.append({
            "prediction": "FAKE" if prob >= 0.5 else "REAL",
            "probability_fake": round(prob, 4),
            "trust_score": int((1 - prob) * 100),
            "risk_level": "HIGH" if prob >= 0.8 else ("MEDIUM" if prob >= 0.5 else "SAFE")
        })

    output_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    output = io.StringIO()
    output_df.to_csv(output, index=False)
    output.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )


@app.get("/stats")
def stats():
    try:
        with open("models/training_results.json") as f:
            return json.load(f)
    except:
        return {"error": "no stats"}


def _log(job_input, result):
    try:
        import sqlite3
        conn = sqlite3.connect("data/predictions.db")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prediction TEXT,
            probability REAL,
            title TEXT
        )
        """)

        conn.execute("""
        INSERT INTO predictions (timestamp, prediction, probability, title)
        VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            result["prediction"],
            result["probability_fake"],
            job_input.get("title", "")
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.debug(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)