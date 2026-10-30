"""
Pure-Python parsing of a Naukri job card (no Selenium needed, so it is unit-testable).

Naukri writes salary as "3-6 Lacs PA" (the rupee sign is an icon, not text) and
shows "Not disclosed" when the recruiter hides it. The class names on the card
change from time to time, so extraction works in two layers:

  1. look inside the usual salary / experience elements (and their title="" text)
  2. if the classes changed, search the card text with regular expressions
"""

import re
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

NOT_DISCLOSED = "Not Disclosed"

_NUM = r"\d{1,3}(?:[.,]\d{1,2})?"
_RANGE_SEP = r"\s*(?:-|–|—|to)\s*"
_PA = r"(?:\s*(?:p\.?\s?a\.?|per\s+annum|per\s+year|/\s*year|/\s*yr))?"

_SALARY_PATTERNS = [
    # 3-6 Lacs PA | 4.5 - 7 Lakhs | 10-15 LPA | 1-2 Cr
    re.compile(
        rf"(?:₹|rs\.?|inr)?\s*{_NUM}{_RANGE_SEP}{_NUM}\s*(?:lacs?|lakhs?|lpa|cr(?:ores?)?)\b{_PA}",
        re.I,
    ),
    # 6 Lacs PA | 12 LPA
    re.compile(rf"(?:₹|rs\.?|inr)?\s*{_NUM}\s*(?:lacs?|lakhs?|lpa)\b{_PA}", re.I),
    # ₹ 15,000 - 25,000 per month | ₹30,000
    re.compile(
        r"(?:₹|rs\.?|inr)\s*\d[\d,]*(?:" + _RANGE_SEP + r"(?:₹|rs\.?|inr)?\s*\d[\d,]*)?"
        r"(?:\s*(?:/|per|a)\s*(?:month|mo|year|yr|annum)|\s*p\.?\s?[am]\.?)?",
        re.I,
    ),
    # 15,000 - 25,000 /month  (no currency symbol)
    re.compile(
        r"\d{1,3}(?:,\d{2,3})+" + _RANGE_SEP + r"\d{1,3}(?:,\d{2,3})+"
        r"(?:\s*(?:/|per|a)\s*(?:month|mo|year|yr|annum)|\s*p\.?\s?[am]\.?)?",
        re.I,
    ),
]

_NOT_DISCLOSED_RE = re.compile(r"not\s+disclosed", re.I)

_EXPERIENCE_RE = re.compile(
    r"\b\d{1,2}\s*(?:-|–|—|to)\s*\d{1,2}\s*(?:yrs?|years?)\b|"
    r"\b\d{1,2}\+?\s*(?:yrs?|years?)\b|\bfreshers?\b",
    re.I,
)

_SALARY_SELECTORS = [
    ".sal-wrap", "span.sal", "li.salary", "[class*='sal-wrap']", "[class*='salary']", "[class*='sal ']",
]
_EXPERIENCE_SELECTORS = [".exp-wrap", ".expwdth", "span.exp", "[class*='exp-wrap']", "[class*='expwdth']"]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip(" |•·-–—,")


def match_salary(text: str) -> Optional[str]:
    """Return the salary phrase found in `text`, NOT_DISCLOSED, or None."""
    text = text or ""
    for pattern in _SALARY_PATTERNS:
        m = pattern.search(text)
        if m:
            return _clean(m.group(0))
    if _NOT_DISCLOSED_RE.search(text):
        return NOT_DISCLOSED
    return None


def match_experience(text: str) -> Optional[str]:
    m = _EXPERIENCE_RE.search(text or "")
    return _clean(m.group(0)) if m else None


def _candidate_texts(soup, selectors):
    for selector in selectors:
        try:
            elements = soup.select(selector)
        except Exception:  # an odd selector must never break scraping
            continue
        for el in elements:
            # title="" usually holds the full value while the visible text may be cut off with "…"
            for node in [el] + el.find_all(True):
                title = node.get("title")
                if title:
                    yield title
            yield el.get_text(" ", strip=True)


def extract_salary(card_html: str = "", card_text: str = "") -> str:
    """Best-effort salary for one card. Falls back to 'Not Disclosed'."""
    if card_html and BeautifulSoup is not None:
        soup = BeautifulSoup(card_html, "html.parser")
        for text in _candidate_texts(soup, _SALARY_SELECTORS):
            found = match_salary(text)
            if found:
                return found
        card_text = card_text or soup.get_text("\n", strip=True)

    # classes changed: search the visible card text line by line, skipping the
    # experience line so "0-2 Yrs" can never be mistaken for a salary
    for line in (card_text or "").splitlines():
        found = match_salary(line)
        if found and found != NOT_DISCLOSED:
            return found
    return NOT_DISCLOSED


def extract_experience(card_html: str = "", card_text: str = "") -> str:
    """Best-effort experience requirement ('' when not shown)."""
    if card_html and BeautifulSoup is not None:
        soup = BeautifulSoup(card_html, "html.parser")
        for text in _candidate_texts(soup, _EXPERIENCE_SELECTORS):
            found = match_experience(text)
            if found:
                return found
        card_text = card_text or soup.get_text("\n", strip=True)

    for line in (card_text or "").splitlines():
        found = match_experience(line)
        if found:
            return found
    return ""
