"""
Small HTTP client the Streamlit dashboard uses to talk to the FastAPI backend.

Every call has a timeout and raises ApiError on any problem, so the dashboard
can show a friendly message or fall back to running the model locally.
"""

import io
from typing import Dict, Optional

import pandas as pd
import requests


class ApiError(Exception):
    """Backend unreachable, timed out, or returned an error status.

    status_code is None when the backend could not be reached at all.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class JobGuardClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"X-API-Key": api_key} if api_key else {}

    # ── internals ──
    def _request(self, method: str, path: str, timeout: Optional[float] = None, **kwargs):
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                timeout=timeout or self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach the JobGuard API at {self.base_url} ({exc.__class__.__name__})") from exc

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"API error {response.status_code}: {detail}", response.status_code)
        return response

    # ── endpoints ──
    def health(self, timeout: float = 2.0) -> Dict:
        return self._request("GET", "/health", timeout=timeout).json()

    def predict(self, payload: Dict) -> Dict:
        data = self._request("POST", "/predict", json=payload).json()
        # The API labels the safe class "REAL"; the dashboard calls it "GENUINE".
        if data.get("prediction") == "REAL":
            data["prediction"] = "GENUINE"
        data["demo_mode"] = False
        return data

    def analyze(self, payload: Dict) -> Dict:
        return self._request("POST", "/analyze", json=payload, timeout=90).json()

    def predict_batch(self, csv_bytes: bytes, filename: str = "jobs.csv") -> pd.DataFrame:
        response = self._request(
            "POST",
            "/predict/batch",
            files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
            timeout=120,
        )
        return pd.read_csv(io.StringIO(response.text))

    def stats(self) -> Dict:
        return self._request("GET", "/stats").json()

    def history(self, limit: int = 20):
        return self._request("GET", "/history", params={"limit": limit}).json()
