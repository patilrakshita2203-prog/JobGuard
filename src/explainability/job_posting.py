"""Lightweight job container used by the API and the rule engine.

The scrapers have their own JobPosting dataclass (which imports Selenium at
module level). This one has the same field names but no heavy imports, so the
FastAPI backend can use the rule engine without installing a browser driver.
"""

from dataclasses import dataclass, asdict


@dataclass
class JobPostingLite:
    title: str = ""
    company: str = ""
    location: str = ""
    salary_range: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    experience: str = ""
    source_platform: str = "Manual input"

    def to_dict(self):
        return asdict(self)

    def combined_text(self):
        return " ".join(
            [self.title, self.company, self.description, self.requirements, self.benefits]
        )
