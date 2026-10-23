"""
JobGuard - central configuration.

Every setting can come from (in this order):
  1. an environment variable (or a local .env file)
  2. Streamlit secrets (when running on Streamlit Community Cloud)
  3. the default below

Nothing in here is secret by default. Secrets such as GEMINI_API_KEY and
JOBGUARD_API_KEY must only ever be provided via environment variables / secrets.
"""

import os
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent


def get_setting(name: str, default=None):
    """Read a setting from env vars, then Streamlit secrets, then default."""
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:  # Streamlit secrets only exist when running inside Streamlit
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


def get_bool(name: str, default: bool = False) -> bool:
    value = get_setting(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_int(name: str, default: int) -> int:
    try:
        return int(get_setting(name, default))
    except (TypeError, ValueError):
        return default


# ── Paths (always absolute, so the app works from any working directory) ──
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"
DB_PATH = Path(get_setting("JOBGUARD_DB_PATH", str(BASE_DIR / "data" / "predictions.db")))

# ── API server ──
API_KEY = get_setting("JOBGUARD_API_KEY")  # if set, POST endpoints require X-API-Key
CORS_ORIGINS = [
    o.strip()
    for o in str(
        get_setting("JOBGUARD_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    ).split(",")
    if o.strip()
]
RATE_LIMIT_PER_MIN = get_int("JOBGUARD_RATE_LIMIT", 60)
BATCH_RATE_LIMIT_PER_MIN = get_int("JOBGUARD_BATCH_RATE_LIMIT", 10)
MAX_UPLOAD_MB = get_int("JOBGUARD_MAX_UPLOAD_MB", 5)
MAX_BATCH_ROWS = get_int("JOBGUARD_MAX_BATCH_ROWS", 2000)
LOG_PREDICTIONS = get_bool("JOBGUARD_LOG_PREDICTIONS", True)

# ── Dashboard ──
API_URL = str(get_setting("JOBGUARD_API_URL", "http://127.0.0.1:8000")).rstrip("/")
# auto  -> use the FastAPI backend when reachable, otherwise run the model locally
# api   -> always use the backend (show an error if it is down)
# local -> never call the backend
MODE = str(get_setting("JOBGUARD_MODE", "auto")).strip().lower()
ENABLE_SCRAPING = get_bool("JOBGUARD_ENABLE_SCRAPING", True)
DEMO_MODE = get_bool("JOBGUARD_DEMO_MODE", False)  # random output - never enable in production

# ── LLM ──
GEMINI_MODEL = str(get_setting("JOBGUARD_GEMINI_MODEL", "gemini-2.5-flash"))
LLM_TIMEOUT_SEC = get_int("JOBGUARD_LLM_TIMEOUT", 30)
