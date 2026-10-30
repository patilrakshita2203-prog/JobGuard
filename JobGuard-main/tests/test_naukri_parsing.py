"""Salary / experience extraction for Naukri cards, including the real-world failure case."""

import pytest

from src.scraper.naukri_parsing import (
    NOT_DISCLOSED, extract_experience, extract_salary, match_salary,
)


def card(row3: str) -> str:
    return (
        '<div class="srp-jobtuple-wrapper"><div class="cust-job-tuple layout-wrapper">'
        '<h2><a class="title" href="#">Data Analyst</a></h2>'
        '<span class="comp-dtls-wrap"><a class="comp-name">Acme</a></span>'
        f'<div class="row3"><div class="job-details">{row3}</div></div>'
        "</div></div>"
    )


STANDARD = card(
    '<span class="exp-wrap"><span class="expwdth">0-2 Yrs</span></span>'
    '<span class="sal-wrap ver-line"><span class="ni-job-tuple-icon ni-job-tuple-icon-srp-rupee sal"></span>'
    '<span class="">3-6 Lacs PA</span></span>'
    '<span class="loc-wrap ver-line"><span class="locWdth">Bengaluru</span></span>'
)


def test_standard_card_lacs_pa():
    """Regression: 'Lacs PA' has no rupee sign / 'LPA' / 'Lakhs', so the old XPath missed it."""
    assert extract_salary(STANDARD) == "3-6 Lacs PA"
    assert extract_experience(STANDARD) == "0-2 Yrs"


def test_not_disclosed_is_reported_as_such():
    html = card('<span class="sal-wrap"><span>Not disclosed</span></span><span class="expwdth">1-3 Yrs</span>')
    assert extract_salary(html) == NOT_DISCLOSED
    assert extract_experience(html) == "1-3 Yrs"


def test_salary_in_title_attribute():
    html = card('<span class="sal-wrap"><span title="4.5-7 Lacs PA">4.5-7 Lacs…</span></span>')
    assert extract_salary(html) == "4.5-7 Lacs PA"


def test_renamed_classes_fall_back_to_text_search():
    html = card('<span class="x1">2-5 Yrs</span><span class="x2">10-15 LPA</span><span class="x3">Remote</span>')
    assert extract_salary(html) == "10-15 LPA"
    assert extract_experience(html) == "2-5 Yrs"


def test_plain_text_only_no_html():
    text = "Data Analyst\nAcme\n0-2 Yrs\n₹ 15,000 - 25,000 per month\nNoida"
    assert extract_salary("", text) == "₹ 15,000 - 25,000 per month"
    assert extract_experience("", text) == "0-2 Yrs"


def test_experience_is_never_mistaken_for_salary():
    html = card('<span class="x1">0-2 Yrs</span><span class="x3">Remote</span>')
    assert extract_salary(html) == NOT_DISCLOSED


def test_nothing_found_defaults_to_not_disclosed_and_empty_experience():
    assert extract_salary("<div>no details</div>", "Title\nCompany") == NOT_DISCLOSED
    assert extract_experience("<div>no details</div>", "Title\nCompany") == ""


@pytest.mark.parametrize("text,expected", [
    ("12 LPA", "12 LPA"),
    ("4.5 - 7 Lakhs", "4.5 - 7 Lakhs"),
    ("1-2 Cr", "1-2 Cr"),
    ("15,000-25,000 /month", "15,000-25,000 /month"),
    ("Hybrid - Noida, Pune", None),
    ("5-10 Yrs", None),
])
def test_match_salary_patterns(text, expected):
    assert match_salary(text) == expected


def test_scraper_uses_the_parser(monkeypatch):
    """Drive NaukriScraper.search_jobs with a fake browser (no Chrome, no network)."""
    selenium = pytest.importorskip("selenium")  # noqa: F841
    import src.scraper.job_scraper as js

    class FakeCard:
        text = "Data Analyst\nAcme\n0-2 Yrs\n3-6 Lacs PA\nBengaluru"

        def find_element(self, *a, **k):
            raise Exception("old selectors not needed for salary")

        def get_attribute(self, name):
            return STANDARD if name == "outerHTML" else ""

    class FakeDriver:
        def get(self, url): pass
        def find_elements(self, *a, **k): return [FakeCard()]
        def quit(self): pass

    class FakeManager:
        def install(self): return "chromedriver"

    monkeypatch.setattr(js.webdriver, "Chrome", lambda **k: FakeDriver())
    monkeypatch.setattr(js, "ChromeDriverManager", FakeManager)
    monkeypatch.setattr(js.time, "sleep", lambda s: None)

    jobs = js.NaukriScraper().search_jobs(keyword="data analyst", max_jobs=3)
    assert len(jobs) == 1
    assert jobs[0].salary_range == "3-6 Lacs PA"
    assert jobs[0].experience == "0-2 Yrs"
