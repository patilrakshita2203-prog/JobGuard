from types import SimpleNamespace

from src.explainability import llm_analyzer as llm


def setup_function():
    llm._CACHE.clear()


def test_without_api_key_returns_fallback_not_an_error(monkeypatch):
    monkeypatch.setattr(llm, "_get_model", lambda: None)
    text, source = llm.generate_ai_analysis_with_source("p", fallback="rule summary")
    assert (text, source) == ("rule summary", "fallback")
    assert "LLM Error" not in llm.generate_ai_analysis("p")


def test_success_is_returned_and_cached(monkeypatch):
    calls = []

    class FakeModel:
        def generate_content(self, prompt, request_options=None):
            calls.append(prompt)
            return SimpleNamespace(text="three bullets")

    monkeypatch.setattr(llm, "_get_model", lambda: FakeModel())
    assert llm.generate_ai_analysis_with_source("same prompt") == ("three bullets", "gemini")
    assert llm.generate_ai_analysis_with_source("same prompt") == ("three bullets", "cache")
    assert len(calls) == 1


def test_failures_retry_then_fall_back(monkeypatch):
    attempts = []

    class BrokenModel:
        def generate_content(self, prompt, request_options=None):
            attempts.append(1)
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(llm, "_get_model", lambda: BrokenModel())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    text, source = llm.generate_ai_analysis_with_source("p2", fallback="rules")
    assert (text, source) == ("rules", "fallback")
    assert len(attempts) == 3
    assert "quota" not in text


def test_analyze_job_risk_without_llm_uses_rules():
    job = SimpleNamespace(title="t", company="c", location="l", salary_range="", description="d",
                          requirements="", benefits="", experience="", source_platform="x")
    facts = {"salary_disclosed": False, "company_available": True, "official_website_found": None,
             "location_available": True, "description_length": 1, "experience_available": False,
             "red_flags": []}
    out = llm.analyze_job_risk(job, facts, probability_fake=0.1, use_llm=False)
    assert out["source"] == "fallback"
    assert "Recommendation" in out["analysis"]
