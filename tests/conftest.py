"""Shared pytest setup. Runs offline: no Gemini key, no web search, no Chrome."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set BEFORE src.config is imported anywhere
os.environ["JOBGUARD_MODE"] = "local"
os.environ["JOBGUARD_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "predictions.db")
os.environ["JOBGUARD_RATE_LIMIT"] = "1000"
os.environ["JOBGUARD_BATCH_RATE_LIMIT"] = "1000"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("JOBGUARD_API_KEY", None)

import pytest  # noqa: E402

MODELS_PRESENT = (ROOT / "models" / "best_model.joblib").exists() and (
    ROOT / "models" / "tfidf_vectorizer.joblib"
).exists()

needs_models = pytest.mark.skipif(
    not MODELS_PRESENT,
    reason="models/best_model.joblib and models/tfidf_vectorizer.joblib are not present",
)

FAKE_TEXT = (
    "URGENT HIRING! No experience needed. Registration fee of Rs 500 required. "
    "100% job guarantee! WhatsApp immediately. Join Telegram group. Security deposit Rs 1000. Free laptop."
)
GENUINE_TEXT = (
    "We seek an experienced Python developer for our backend team. Design REST APIs, work with "
    "PostgreSQL, Agile, code reviews. 3-5 years experience, Python, SQL, Git. Health insurance, PF, "
    "learning budget, hybrid work. Interview: online coding test, technical, HR."
)


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as client:
        yield client


@pytest.fixture()
def fresh_limiter():
    import main

    main._limiter._hits.clear()
    yield
    main._limiter._hits.clear()
