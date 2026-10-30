import pytest

pytest.importorskip("selenium")


def test_internshala_posting_has_the_fields_the_dashboard_reads():
    """Regression: the dashboard's AI-analysis button read .requirements/.benefits/.experience."""
    from src.scraper.internshala_scraper import JobPosting

    job = JobPosting(title="Python Intern", company="Acme", description="card text")
    for field in ("requirements", "benefits", "experience", "salary_range", "location"):
        assert getattr(job, field) == "" or getattr(job, field) is not None
    assert "card text" in job.combined_text()
