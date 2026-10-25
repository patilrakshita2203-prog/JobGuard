"""Headless smoke test of the Streamlit dashboard (local mode, no API, no network)."""

import pytest

from conftest import ROOT, needs_models

pytestmark = needs_models

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def _click(at, label_part):
    return [b for b in at.button if b.label and label_part in b.label][0].click().run()


def test_dashboard_loads_without_errors(app):
    assert not app.exception
    assert len(app.tabs) == 5


def test_demo_button_fills_the_form_and_analysis_flags_the_scam():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    _click(at, "Scam Demo")
    assert at.text_input[1].value == "FastMoney Solutions"
    _click(at, "ANALYZE JOB")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "LIKELY SCAM" in text
    assert "Scam Red Flags" in text


def test_results_survive_the_ai_analysis_button():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    _click(at, "Scam Demo")
    _click(at, "ANALYZE JOB")
    _click(at, "Generate AI Risk Analysis")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "LIKELY SCAM" in text, "results used to disappear after pressing a button inside them"
    assert "Recommendation" in text


def test_internshala_scan_and_ai_analysis_do_not_crash(monkeypatch):
    """Regression: Internshala jobs have no .requirements/.benefits/.experience attributes."""
    pytest.importorskip("selenium")
    import src.scraper.internshala_scraper as ish

    def fake_search(self, keyword="python", max_jobs=10):
        return [
            ish.JobPosting(title="Python Intern", company="Acme", location="Remote",
                           salary_range="₹ 10,000 /month", description="Work on APIs with Python and SQL."),
            ish.JobPosting(title="Data Entry", company="FastMoney", location="Remote",
                           salary_range="Not Disclosed", description="Registration fee Rs 500, WhatsApp to join."),
        ]

    monkeypatch.setattr(ish.InternshalaScraper, "search_jobs", fake_search)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    [s for s in at.selectbox if s.label and "Platform" in s.label][0].select("Internshala").run()
    _click(at, "START SCAN")
    assert not at.exception and not at.error
    [s for s in at.selectbox if s.key == "scan_job_idx"][0].select(1).run()
    _click(at, "Generate AI Risk Analysis")
    assert not at.exception
    assert at.session_state["scan_ai"][0] == 1
