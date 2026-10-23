"""
Rule-based scam red-flag detector (English + Hindi keywords).

This runs next to the ML model, not instead of it: it never changes the model
probability. It explains *why* a posting looks suspicious in plain words and
catches patterns people recognise instantly (fees, WhatsApp-only hiring, ...).
"""

import re
from typing import Dict, List

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# (code, severity, label, regex)
_RULES = [
    (
        "upfront_payment",
        "high",
        "Asks you to pay money (registration / training / security fee)",
        r"registration\s+(fee|charges?)|training\s+(fee|charges?)|security\s+deposit|"
        r"refundable\s+(deposit|amount)|processing\s+fee|kit\s+charges?|advance\s+payment|"
        r"pay\s+(rs\.?|inr|₹)\s*\d+|"
        r"पंजीकरण\s*शुल्क|रजिस्ट्रेशन\s*फ़?ीस|सिक्योरिटी\s*डिपॉजिट|शुल्क\s*जमा|पैसे\s*जमा",
    ),
    (
        "guaranteed_outcome",
        "high",
        "Promises a guaranteed job or guaranteed income",
        r"100\s*%\s*(job\s*)?(guarantee|placement)|guaranteed\s+(job|placement|income|salary)|"
        r"job\s+guarantee|नौकरी\s*की\s*गारंटी|100\s*%\s*गारंटी",
    ),
    (
        "messaging_app_hiring",
        "high",
        "Hiring or contact happens only on WhatsApp / Telegram",
        r"whats\s?app|telegram|व्हाट्सएप|व्हाट्सऐप|टेलीग्राम",
    ),
    (
        "sensitive_data_request",
        "high",
        "Asks for sensitive documents or banking details up front",
        r"aadh?aar|pan\s*card|bank\s+(account|details)|\botp\b|cvv|upi\s*pin|आधार|बैंक\s*(खाता|डिटेल)",
    ),
    (
        "investment_or_crypto",
        "high",
        "Mentions investment, crypto or forex as part of the 'job'",
        r"investment\s+required|invest\s+(rs\.?|inr|₹)|bitcoin|crypto(currency)?|forex|"
        r"wire\s+transfer|western\s+union|lottery|prize\s+money",
    ),
    (
        "no_interview",
        "medium",
        "No interview or 'direct joining'",
        r"no\s+interview|direct\s+joining|without\s+interview|बिना\s*इंटरव्यू|डायरेक्ट\s*जॉइनिंग",
    ),
    (
        "urgency_pressure",
        "medium",
        "Pressure tactics (urgent hiring, limited seats)",
        r"urgent(ly)?\s+(hiring|requirement|vacancy)|limited\s+seats?|apply\s+immediately|"
        r"join\s+immediately|hurry|last\s+date\s+today|तुरंत\s*(जॉइन|ज्वाइन)|सीमित\s*सीटें",
    ),
    (
        "unrealistic_earnings",
        "medium",
        "Very high or 'unlimited' earnings for easy work",
        r"unlimited\s+(earning|income)|earn\s+(up\s+to\s+)?(rs\.?|inr|₹)?\s*\d{2,3}[,]?\d{3}\s*"
        r"(/|per\s+)?\s*(month|day|week)|earn\s+(money\s+)?(fast|easily|daily)|easy\s+money|"
        r"daily\s+payout",
    ),
    (
        "no_experience_high_pay",
        "low",
        "'No experience needed' used as a selling point",
        r"no\s+experience\s+(needed|required|necessary)|बिना\s*अनुभव",
    ),
    (
        "free_gift",
        "low",
        "Promises free laptop / phone / gifts",
        r"free\s+(laptop|mobile|phone|gift)",
    ),
]

_COMPILED = [(c, s, l, re.compile(rx, re.IGNORECASE)) for c, s, l, rx in _RULES]

_FREE_EMAIL = re.compile(
    r"[\w.+-]+@(gmail|yahoo|ymail|outlook|hotmail|rediffmail|protonmail|live)\.[a-z.]+",
    re.IGNORECASE,
)


def detect_red_flags(text: str, company: str = "") -> List[Dict]:
    """Return a list of {code, severity, label, evidence} dicts, worst first."""
    text = text or ""
    flags: List[Dict] = []

    for code, severity, label, pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            flags.append(
                {
                    "code": code,
                    "severity": severity,
                    "label": label,
                    "evidence": match.group(0).strip()[:60],
                }
            )

    email = _FREE_EMAIL.search(text)
    if email:
        flags.append(
            {
                "code": "free_email_contact",
                "severity": "medium",
                "label": "Recruiter contact uses a free email address (Gmail/Yahoo/...)",
                "evidence": email.group(0)[:60],
            }
        )

    flags.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], f["code"]))
    return flags


def red_flag_score(flags: List[Dict]) -> int:
    """0-100 helper score: 30 per high, 15 per medium, 5 per low flag (capped)."""
    weights = {"high": 30, "medium": 15, "low": 5}
    return min(100, sum(weights[f["severity"]] for f in flags))
