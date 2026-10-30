"""
Company website check.

Returns a tri-state value:
    True  -> a search result whose domain matches the company name was found
    False -> search worked, but nothing matched the company name
    None  -> could not be checked (empty name, search error, rate limit, no results)

None is deliberately NOT treated as False: a flaky search must never be
reported to the user (or the LLM) as "this company has no website".
"""

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Sites that list companies but are not the company's own website
_AGGREGATOR_DOMAINS = (
    "linkedin.", "naukri.", "indeed.", "glassdoor.", "internshala.", "facebook.",
    "instagram.", "twitter.", "x.com", "youtube.", "wikipedia.", "ambitionbox.",
    "justdial.", "crunchbase.", "quora.", "reddit.", "monster.", "foundit.",
    "shine.", "timesjobs.", "apna.", "zaubacorp.", "tofler.", "bloomberg.",
)

_GENERIC_WORDS = {
    "pvt", "private", "ltd", "limited", "llp", "inc", "llc", "corp", "corporation",
    "company", "co", "technologies", "technology", "tech", "solutions", "services",
    "systems", "software", "india", "global", "group", "the", "and", "of",
    "consulting", "enterprises", "industries", "international",
}

_CACHE: dict = {}
_CACHE_TTL_SEC = 6 * 60 * 60


def _tokens(company_name: str):
    words = re.findall(r"[a-z0-9]+", company_name.lower())
    significant = [w for w in words if w not in _GENERIC_WORDS and len(w) >= 3]
    return significant or [w for w in words if len(w) >= 3]


def _domain_label(url: str) -> str:
    host = urlparse(url if "//" in url else f"//{url}").netloc.lower()
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_matches_company(url: str, company_name: str) -> bool:
    host = _domain_label(url)
    if not host or any(host.startswith(d) or f".{d}" in f".{host}" for d in _AGGREGATOR_DOMAINS):
        return False
    squashed_host = re.sub(r"[^a-z0-9]", "", host)
    squashed_name = re.sub(r"[^a-z0-9]", "", " ".join(_tokens(company_name)))
    if squashed_name and squashed_name in squashed_host:
        return True
    return any(token in squashed_host for token in _tokens(company_name))


def _search(query: str):
    try:
        from ddgs import DDGS  # new package name
    except ImportError:
        from duckduckgo_search import DDGS  # older package name
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=5))


def company_has_website(company_name: str) -> Optional[bool]:
    name = (company_name or "").strip()
    if not name:
        return None

    key = name.lower()
    cached = _CACHE.get(key)
    if cached and time.time() - cached[1] < _CACHE_TTL_SEC:
        return cached[0]

    try:
        results = _search(f"{name} official website")
    except Exception as exc:  # rate limit, network, missing package ...
        logger.warning("Company website check failed for %r: %s", name, exc)
        return None

    if not results:
        return None

    found = any(
        _domain_matches_company(r.get("href", "") or r.get("url", ""), name) for r in results
    )
    _CACHE[key] = (found, time.time())
    return found
