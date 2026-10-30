import io

import pytest

from src import config as cfg
import main
from conftest import FAKE_TEXT, GENUINE_TEXT, needs_models

pytestmark = needs_models


def test_health(api_client):
    data = api_client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "version" in data


def test_root_points_to_docs(api_client):
    assert api_client.get("/").json()["docs"] == "/docs"


def test_predict_returns_the_documented_fields(api_client):
    r = api_client.post("/predict", json={"title": "Work From Home", "description": FAKE_TEXT})
    assert r.status_code == 200
    data = r.json()
    for key in ["prediction", "confidence", "probability_fake", "trust_score", "top_fake_words",
                "top_genuine_words", "highlighted_html", "risk_level", "timestamp",
                "red_flags", "explanation"]:
        assert key in data
    assert 0 <= data["probability_fake"] <= 1
    assert 0 <= data["trust_score"] <= 100
    assert data["trust_score"] == int((1 - data["probability_fake"]) * 100)
    assert data["red_flags"], "scam text should produce rule-based red flags"


def test_genuine_job_gets_trust_words_and_no_flags(api_client):
    data = api_client.post("/predict", json={"title": "Python Developer", "description": GENUINE_TEXT}).json()
    assert data["prediction"] == "REAL"
    assert data["top_fake_words"] == []
    assert data["top_genuine_words"], "regression: genuine side of the explanation used to be empty"
    assert data["red_flags"] == []


def test_predict_validates_input(api_client):
    assert api_client.post("/predict", json={"title": "no description"}).status_code == 422


def test_predict_without_model_is_503_not_500(api_client, monkeypatch):
    monkeypatch.setattr(main, "model", None)
    assert api_client.post("/predict", json={"description": "x"}).status_code == 503


def test_batch_csv_roundtrip(api_client):
    csv = f'title,description\nPython Dev,"{GENUINE_TEXT}"\nData Entry,"{FAKE_TEXT}"\n'
    r = api_client.post("/predict/batch", files={"file": ("jobs.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    lines = r.text.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].endswith("prediction,probability_fake,trust_score,risk_level")


def test_batch_handles_missing_values(api_client):
    csv = "title,description\n,\n,some text here\n"
    r = api_client.post("/predict/batch", files={"file": ("jobs.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200
    assert "nan" not in r.text.lower()


@pytest.mark.parametrize("name,body,status", [
    ("jobs.txt", b"a", 400),
    ("jobs.csv", b"title,other\n1,2\n", 400),
    ("jobs.csv", b"\xff\xfe\x00bad", 400),
])
def test_batch_rejects_bad_files(api_client, name, body, status):
    r = api_client.post("/predict/batch", files={"file": (name, io.BytesIO(body), "text/csv")})
    assert r.status_code == status


def test_batch_size_limits(api_client, monkeypatch):
    big = io.BytesIO(b"description\n" + b"x" * 2048)
    monkeypatch.setattr(cfg, "MAX_UPLOAD_MB", 0)
    assert api_client.post("/predict/batch", files={"file": ("a.csv", big, "text/csv")}).status_code == 413
    monkeypatch.setattr(cfg, "MAX_UPLOAD_MB", 5)
    monkeypatch.setattr(cfg, "MAX_BATCH_ROWS", 1)
    rows = io.BytesIO(b"description\none\ntwo\n")
    assert api_client.post("/predict/batch", files={"file": ("a.csv", rows, "text/csv")}).status_code == 413


def test_analyze_falls_back_to_rules_without_gemini(api_client):
    r = api_client.post("/analyze", json={"title": "Data Entry", "company_profile": "FastMoney",
                                          "description": FAKE_TEXT})
    assert r.status_code == 200
    data = r.json()
    assert data["analysis_source"] == "no_key"
    assert "Recommendation" in data["analysis"]
    assert data["facts"]["official_website_found"] is None   # verify_company defaults to off
    assert data["red_flags"]


def test_history_lists_recent_predictions(api_client):
    api_client.post("/predict", json={"title": "History Probe Role", "description": GENUINE_TEXT})
    rows = api_client.get("/history?limit=5").json()
    assert rows and rows[0]["title"] == "History Probe Role"
    assert api_client.get("/history?limit=0").status_code == 422


def test_stats_endpoint(api_client):
    assert "best_model" in api_client.get("/stats").json()


def test_api_key_is_enforced_when_configured(api_client, monkeypatch):
    monkeypatch.setattr(cfg, "API_KEY", "s3cret")
    body = {"description": GENUINE_TEXT}
    assert api_client.post("/predict", json=body).status_code == 401
    assert api_client.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert api_client.post("/predict", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert api_client.get("/health").status_code == 200          # health stays public


def test_rate_limit_returns_429(api_client, monkeypatch, fresh_limiter):
    monkeypatch.setattr(cfg, "RATE_LIMIT_PER_MIN", 2)
    body = {"description": GENUINE_TEXT}
    codes = [api_client.post("/predict", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
