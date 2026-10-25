# Job scraper for live job postings

import os
import time
import logging
import random
import requests
from dataclasses import dataclass, asdict
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

try:
    from src.scraper.naukri_parsing import extract_salary, extract_experience, NOT_DISCLOSED
except ImportError:  # running this file directly: python src/scraper/job_scraper.py
    from naukri_parsing import extract_salary, extract_experience, NOT_DISCLOSED

logger = logging.getLogger(__name__)

# Rotate user agents to avoid blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/118.0 Safari/537.36"
]

# Set JOBGUARD_SCRAPER_DEBUG=true to print what was extracted and save the first card's HTML
DEBUG_SCRAPER = os.getenv("JOBGUARD_SCRAPER_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

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

# Naukri.com scraper
class NaukriScraper(BaseScraper):

    def search_jobs(self, keyword="software engineer", max_jobs=10):

        options = Options()

        #options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        url = f"https://www.naukri.com/{keyword.replace(' ', '-')}-jobs?k={keyword.replace(' ', '%20')}"

        driver.get(url)
        print("Opened URL:", url)

        time.sleep(5)

        cards = driver.find_elements(
            
            By.XPATH,
            "//div[contains(@class,'cust-job-tuple')]"
            
        )
        print("Cards found:", len(cards))

        jobs = []

        for card in cards[:max_jobs]:

            try:

                title = card.text.strip().split("\n")[0]

                company = ""
                location = ""
                salary = ""

                try:
                    company = card.find_element(By.CLASS_NAME, "comp-name").text
                except:
                    pass

                try:
                    location = card.find_element(By.CLASS_NAME, "locWdth").text
                except:
                    pass

                full_text = card.text.strip()

                # Salary + experience: Naukri shows "3-6 Lacs PA" (the old check only
                # looked for the rupee sign / "LPA" / "Lakhs", so it never matched)
                try:
                    card_html = card.get_attribute("outerHTML") or ""
                except Exception:
                    card_html = ""
                salary = extract_salary(card_html, full_text) or NOT_DISCLOSED
                experience = extract_experience(card_html, full_text)

                if DEBUG_SCRAPER and not jobs:  # dump the first card so selectors can be fixed fast
                    try:
                        with open("debug_naukri_card.html", "w", encoding="utf-8") as f:
                            f.write(card_html)
                    except OSError:
                        pass
                if DEBUG_SCRAPER:
                    print(f"[debug] {title[:40]!r} salary={salary!r} experience={experience!r}")

                job = JobPosting(
                    title=title,
                    company=company,
                    location=location,
                    salary_range=salary,
                    experience=experience,
                    description=full_text,
                    source_platform="Naukri"
                )

                jobs.append(job)

            except Exception as e:
                print("Error:", e)

        driver.quit()

        return jobs

    # Parse single job card
    
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