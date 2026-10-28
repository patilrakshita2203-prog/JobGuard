from src.explainability.company_verifier import company_has_website
from src.explainability.red_flags import detect_red_flags


def _s(value) -> str:
    return (value or "").strip()


def analyze_job(job, verify_company: bool = True):
    """Extract objective facts about a job posting.

    `official_website_found` is True / False / None (None = could not be checked).
    Set verify_company=False to skip the (slow, network) website lookup.
    """
    facts = {}

    # Salary
    facts["salary_disclosed"] = (
        _s(job.salary_range).lower() != "not disclosed" and _s(job.salary_range) != ""
    )

    # Company
    facts["company_available"] = _s(job.company) != ""
    facts["website_check_requested"] = bool(verify_company)
    facts["official_website_found"] = (
        company_has_website(job.company) if verify_company else None
    )

    # Location
    facts["location_available"] = _s(job.location) != ""

    # Description
    facts["description_length"] = len(_s(job.description))

    # Experience
    facts["experience_available"] = _s(getattr(job, "experience", "")) != ""

    # Scam red flags (English + Hindi keyword rules)
    text = " ".join(
        _s(getattr(job, attr, ""))
        for attr in ("title", "company", "description", "requirements", "benefits")
    )
    facts["red_flags"] = detect_red_flags(text, _s(job.company))

    return facts
