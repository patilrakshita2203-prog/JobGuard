"""Deterministic risk summary used when the Gemini call is unavailable."""

from typing import Dict, List, Optional


def _yn(value: Optional[bool], yes: str, no: str, unknown: str) -> str:
    if value is None:
        return unknown
    return yes if value else no


def build_fallback_analysis(
    job, facts: Dict, red_flags: Optional[List[Dict]] = None, probability_fake: Optional[float] = None
) -> str:
    """Return 3 bullet points + 1 recommendation built only from verified facts."""
    red_flags = red_flags or []
    bullets = []

    if facts.get("website_check_requested") is False:
        website_line = "The company website was not looked up, so the employer is unverified."
    else:
        website_line = _yn(
            facts.get("official_website_found"),
            "An official-looking website matching the company name was found online.",
            "No website matching the company name was found online, so the employer could not be verified.",
            "The company website could not be checked right now (search unavailable), so the employer is unverified.",
        )
    bullets.append(website_line)

    transparency = []
    transparency.append("salary is disclosed" if facts.get("salary_disclosed") else "salary is not disclosed")
    transparency.append("location is given" if facts.get("location_available") else "location is missing")
    transparency.append(
        "experience is stated" if facts.get("experience_available") else "experience is not stated"
    )
    bullets.append("Posting transparency: " + ", ".join(transparency) + ".")

    if red_flags:
        top = "; ".join(f["label"] for f in red_flags[:3])
        bullets.append(f"{len(red_flags)} scam red flag(s) detected: {top}.")
    elif probability_fake is not None:
        bullets.append(
            f"The ML model rates the fraud risk at {probability_fake * 100:.0f}% and no rule-based red flags were found."
        )
    else:
        bullets.append("No rule-based scam red flags were found in the text.")

    high = any(f["severity"] == "high" for f in red_flags)
    if high or (probability_fake is not None and probability_fake >= 0.8):
        rec = "Do not pay any fee or share documents or bank details; verify the employer through its official website before replying."
    elif red_flags or (probability_fake is not None and probability_fake >= 0.5):
        rec = "Verify the recruiter and company independently before sharing personal information."
    else:
        rec = "Looks reasonable, but always confirm the opening on the company's official careers page."

    return "\n".join(f"- {b}" for b in bullets) + f"\n\n**Recommendation:** {rec}"
