import pytest
import requests

from src import api_client
from src.api_client import ApiError, JobGuardClient


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_connection_failure_raises_api_error_without_status(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(api_client.requests, "request", boom)
    with pytest.raises(ApiError) as exc:
        JobGuardClient("http://x").health()
    assert exc.value.status_code is None


def test_error_status_is_exposed(monkeypatch):
    monkeypatch.setattr(api_client.requests, "request",
                        lambda *a, **k: FakeResponse(429, {"detail": "slow down"}))
    with pytest.raises(ApiError) as exc:
        JobGuardClient("http://x").predict({"description": "d"})
    assert exc.value.status_code == 429 and "slow down" in str(exc.value)


def test_real_is_renamed_to_genuine_and_key_is_sent(monkeypatch):
    seen = {}

    def fake(method, url, headers=None, timeout=None, **kw):
        seen.update(url=url, headers=headers)
        return FakeResponse(200, {"prediction": "REAL", "probability_fake": 0.1})

    monkeypatch.setattr(api_client.requests, "request", fake)
    out = JobGuardClient("http://api:8000/", api_key="secret").predict({"description": "d"})
    assert out["prediction"] == "GENUINE" and out["demo_mode"] is False
    assert seen["url"] == "http://api:8000/predict"
    assert seen["headers"] == {"X-API-Key": "secret"}
