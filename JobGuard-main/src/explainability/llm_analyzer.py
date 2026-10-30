"""
Gemini-powered risk analysis with safe fallbacks.

- The API key comes from GEMINI_API_KEY (env / .env / Streamlit secrets) and is
  read lazily, so importing this module never crashes when no key is set.
- Calls have a timeout, retry twice with back-off, and results are cached.
- If Gemini is unavailable the caller gets a rule-based summary instead of a
  raw "LLM Error" string.
"""

import hashlib
import logging
import time
from typing import Dict, Optional

from src.config import GEMINI_MODEL, LLM_TIMEOUT_SEC, get_setting
from src.explainability.fallback_analyzer import build_fallback_analysis

logger = logging.getLogger(__name__)

_MODEL = None
_CACHE: Dict[str, str] = {}
_MAX_CACHE = 256


def _get_model():
    """Create the Gemini client on first use. Returns None if not configured."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    api_key = get_setting("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        _MODEL = genai.GenerativeModel(GEMINI_MODEL)
        return _MODEL
    except Exception as exc:
        logger.warning("Could not initialise Gemini: %s", exc)
        return None


def is_configured() -> bool:
    return bool(get_setting("GEMINI_API_KEY"))


def generate_ai_analysis(prompt: str, fallback: Optional[str] = None) -> str:
    """Return Gemini's analysis for `prompt`.

    Same name/signature as before (the new `fallback` argument is optional).
    On any failure it returns `fallback` (or a short friendly message) instead
    of raising or exposing the provider's raw error text.
    """
    text, _ = generate_ai_analysis_with_source(prompt, fallback)
    return text


def generate_ai_analysis_with_source(prompt: str, fallback: Optional[str] = None):
    """Like generate_ai_analysis but also returns the source: 'gemini' | 'cache' | 'no_key' | 'fallback'."""
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if key in _CACHE:
        return _CACHE[key], "cache"

    default_msg = fallback or (
        "AI analysis is unavailable right now. Please rely on the model score "
        "and the rule-based red flags above."
    )

    model = _get_model()
    if model is None:
        # "no_key": GEMINI_API_KEY is missing; "fallback": key present but client could not start
        return default_msg, ("fallback" if is_configured() else "no_key")

    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt, request_options={"timeout": LLM_TIMEOUT_SEC}
            )
            text = (response.text or "").strip()
            if text:
                if len(_CACHE) >= _MAX_CACHE:
                    _CACHE.pop(next(iter(_CACHE)))
                _CACHE[key] = text
                return text, "gemini"
            last_error = "empty response"
        except Exception as exc:  # network, quota, safety block ...
            last_error = exc
            logger.warning("Gemini attempt %d failed: %s", attempt + 1, exc)
        time.sleep(1.5 * (attempt + 1))

    logger.error("Gemini failed after retries: %s", last_error)
    return default_msg, "fallback"


def analyze_job_risk(job, facts: Dict, probability_fake: Optional[float] = None, use_llm: bool = True):
    """High-level helper: build the prompt, ask Gemini, fall back to rules.

    Returns {"analysis": str, "source": "gemini"|"cache"|"no_key"|"fallback"}.
    """
    from src.explainability.prompt_builder import build_prompt

    fallback = build_fallback_analysis(job, facts, facts.get("red_flags"), probability_fake)
    if not use_llm:
        return {"analysis": fallback, "source": "fallback"}

    text, source = generate_ai_analysis_with_source(build_prompt(job, facts), fallback)
    return {"analysis": text, "source": source}
