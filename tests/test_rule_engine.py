from types import SimpleNamespace

from src.explainability import rule_engine
from src.explainability.fallback_analyzer import build_fallback_analysis
from src.explainability.prompt_builder import build_prompt


def make_job(**kw):
    base = dict(
        title="Python Intern", company="Acme Labs", location="Bengaluru",
        salary_range="", description="Build APIs with Python.", requirements="", benefits="",
        experience="", source_platform="Test",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_facts_basic(monkeypatch):
    monkeypatch.setattr(rule_engine, "company_has_website", lambda name: True)
    facts = rule_engine.analyze_job(make_job(salary_range="6 LPA", experience="0-1 years"))
    assert facts["salary_disclosed"] is True
    assert facts["company_available"] is True
    assert facts["official_website_found"] is True
    assert facts["experience_available"] is True
    assert facts["location_available"] is True


def test_none_fields_do_not_crash(monkeypatch):
    monkeypatch.setattr(rule_engine, "company_has_website", lambda name: None)
    facts = rule_engine.analyze_job(make_job(salary_range=None, location=None, company=None))
    assert facts["salary_disclosed"] is False
    assert facts["company_available"] is False


def test_verify_company_false_skips_the_network(monkeypatch):
    def boom(name):
        raise AssertionError("network lookup must not run")

    monkeypatch.setattr(rule_engine, "company_has_website", boom)
    facts = rule_engine.analyze_job(make_job(), verify_company=False)
    assert facts["official_website_found"] is None


def test_unknown_website_is_not_reported_as_missing(monkeypatch):
    monkeypatch.setattr(rule_engine, "company_has_website", lambda name: None)
    job = make_job()
    facts = rule_engine.analyze_job(job)
    assert "Could not be checked" in build_prompt(job, facts)
    assert "could not be checked" in build_fallback_analysis(job, facts)


def test_red_flags_are_part_of_the_facts(monkeypatch):
    monkeypatch.setattr(rule_engine, "company_has_website", lambda name: None)
    facts = rule_engine.analyze_job(make_job(description="Registration fee Rs 500, WhatsApp to join"))
    assert {f["code"] for f in facts["red_flags"]} >= {"upfront_payment", "messaging_app_hiring"}
