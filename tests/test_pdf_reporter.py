from src.reporting.pdf_reporter import generate_pdf_report

RESULT = {"prediction": "FAKE", "confidence": 0.9, "probability_fake": 0.9, "trust_score": 10, "risk_level": "HIGH"}


def test_pdf_survives_markup_characters_and_rupee(tmp_path, monkeypatch):
    """Regression: '&', '<' and unclosed tags in a description used to crash ReportLab."""
    monkeypatch.chdir(tmp_path)
    job = {
        "title": "R&D Engineer <Lead>",
        "company": "A&B <b>Corp",
        "location": "Pune",
        "salary_range": "₹ 15,000 - 25,000 per month",
        "description": 'Join our R&D <team & "partners"> now!\nSecond line with <b>unclosed tag and ₹50,000',
    }
    path = generate_pdf_report(job, RESULT)
    data = (tmp_path / path).read_bytes() if not path.startswith("/") else open(path, "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 1000


def test_pdf_handles_missing_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = generate_pdf_report({"title": None, "description": None}, {"prediction": "REAL"})
    assert path.endswith(".pdf")
