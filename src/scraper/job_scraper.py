# Job scraper for live job postings

import time
import logging
import random
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# Rotate user agents to avoid blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/118.0 Safari/537.36"
]

# Delay between requests
MIN_DELAY = 2
MAX_DELAY = 5


# Store job posting details
@dataclass
class JobPosting:
    title: str = ""
    company: str = ""
    location: str = ""
    salary_range: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    experience: str = ""
    employment_type: str = ""
    source_url: str = ""
    source_platform: str = ""
    posted_date: str = ""
    has_company_logo: int = 0
    telecommuting: int = 0

    # Convert object to dictionary
    def to_dict(self):
        return asdict(self)

    # Combine text for ML prediction
    def combined_text(self):
        return " ".join([
            self.title,
            self.company,
            self.description,
            self.requirements,
            self.benefits
        ])


# Base scraper with session handling
class BaseScraper:
    def __init__(self):
        self.session = requests.Session()

    # Create request headers
    def get_headers(self):
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9"
        }

    # Fetch page with retry logic
    def fetch_page(self, url, retries=3):
        for attempt in range(retries):
            try:
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

                response = self.session.get(
                    url,
                    headers=self.get_headers(),
                    timeout=15
                )

                if response.status_code == 200:
                    return BeautifulSoup(response.content, "html.parser")

                elif response.status_code == 429:
                    logger.warning("Rate limited")
                    time.sleep(10)

                elif response.status_code == 403:
                    logger.warning("Access blocked")
                    return None

            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                time.sleep(3)

        return None


# Naukri.com scraper
class NaukriScraper(BaseScraper):
    BASE_URL = "https://www.naukri.com"
    SEARCH_URL = "https://www.naukri.com/{keyword}-jobs"

    # Search jobs from Naukri
    def search_jobs(self, keyword="fresher", max_jobs=10):
        logger.info(f"Searching jobs for: {keyword}")

        url = self.SEARCH_URL.format(
            keyword=keyword.replace(" ", "-")
        )

        soup = self.fetch_page(url)

        if not soup:
            return self.get_mock_jobs(max_jobs)

        job_cards = (
            soup.find_all("article", class_="jobTuple")
            or soup.find_all("div", class_="cust-job-tuple")
        )

        jobs = []

        for card in job_cards[:max_jobs]:
            job = self.parse_job_card(card)
            if job:
                jobs.append(job)

        if not jobs:
            return self.get_mock_jobs(max_jobs)

        return jobs

    # Parse single job card
    def parse_job_card(self, card):
        try:
            job = JobPosting(source_platform="Naukri.com")

            title_tag = card.find("a", class_="title") or card.find("h2")
            job.title = title_tag.get_text(strip=True) if title_tag else "Unknown"

            company_tag = (
                card.find("a", class_="comp-name")
                or card.find("a", class_="companyInfo")
            )
            job.company = company_tag.get_text(strip=True) if company_tag else ""

            location_tag = (
                card.find("li", class_="loc")
                or card.find("span", class_="loc")
            )
            job.location = location_tag.get_text(strip=True) if location_tag else ""

            salary_tag = (
                card.find("li", class_="salary")
                or card.find("span", class_="salary")
            )
            job.salary_range = salary_tag.get_text(strip=True) if salary_tag else ""

            exp_tag = (
                card.find("li", class_="exp")
                or card.find("span", class_="exp")
            )
            job.experience = exp_tag.get_text(strip=True) if exp_tag else ""

            return job if job.title != "Unknown" else None

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    # Return mock jobs for testing
    def get_mock_jobs(self, n):
        return [
            JobPosting(
                title="Work From Home - Earn ₹50,000/month",
                company="FastMoney Pvt Ltd",
                location="Remote",
                salary_range="50000-80000",
                description="Registration fee required",
                requirements="Anyone can apply",
                source_platform="Mock Data",
                has_company_logo=0
            ),
            JobPosting(
                title="Python Backend Developer",
                company="Infosys",
                location="Bangalore",
                salary_range="600000-900000",
                description="Python developer needed",
                requirements="Python, Django, SQL",
                benefits="Health insurance",
                source_platform="Mock Data",
                has_company_logo=1
            )
        ][:n]


# Scrape jobs from all platforms
def scrape_live_jobs(keyword="fresher", max_jobs=5):
    scraper = NaukriScraper()
    return scraper.search_jobs(
        keyword=keyword,
        max_jobs=max_jobs
    )


# Run scraper directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    jobs = scrape_live_jobs(
        keyword="software engineer",
        max_jobs=5
    )

    for job in jobs:
        print(f"Title: {job.title}")
        print(f"Company: {job.company}")
        print(f"Location: {job.location}")
        print(f"Salary: {job.salary_range}")
        print("-" * 40)