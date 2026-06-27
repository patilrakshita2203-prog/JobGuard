import time
from dataclasses import dataclass, asdict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@dataclass
class JobPosting:
    title: str = ""
    company: str = ""
    location: str = ""
    salary_range: str = ""
    description: str = ""
    source_platform: str = ""

    def to_dict(self):
        return asdict(self)

    def combined_text(self):
        return " ".join([
            self.title,
            self.company,
            self.description
        ])


class InternshalaScraper:

    def search_jobs(self, keyword="python", max_jobs=10):

        options = Options()

        # Uncomment to run headless
        # options.add_argument("--headless")

        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        driver = webdriver.Chrome(options=options)

        jobs = []

        try:

            url = f"https://internshala.com/jobs/{keyword.replace(' ', '-')}-jobs"

            print(f"Navigating to: {url}")
            driver.get(url)

            # Wait for page to load
            print("Waiting for page to load...")
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.TAG_NAME, "body")
                )
            )

            # Wait extra time for JavaScript to render
            time.sleep(5)

            # Scroll down to load more jobs
            print("Scrolling to load more jobs...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            # Try multiple selectors to find job cards
            cards = []
            
            # Selector 1: Main job card container
            try:
                cards = driver.find_elements(By.XPATH, "//div[@class='individual_internship']")
                print(f"Found {len(cards)} cards with selector 1")
            except:
                pass

            # Selector 2: Fallback - any div with data attributes
            if not cards:
                try:
                    cards = driver.find_elements(By.XPATH, "//div[contains(@class,'individual')]")
                    print(f"Found {len(cards)} cards with selector 2")
                except:
                    pass

            # Selector 3: Link-based approach
            if not cards:
                try:
                    job_links = driver.find_elements(By.XPATH, "//a[contains(@href,'/jobs/')]")
                    print(f"Found {len(job_links)} job links")
                except:
                    pass

            print(f"TOTAL CARDS FOUND: {len(cards)}")

            if not cards:
                print("No job cards found. Trying to get page source...")
                # Print a small part of HTML for debugging
                page_source = driver.page_source
                if "internship" in page_source.lower():
                    print("✓ Page contains 'internship' text")
                else:
                    print("✗ Page doesn't contain 'internship' text")
                return []

            for idx, card in enumerate(cards[:max_jobs]):

                try:
                    # Get the entire text content first
                    card_text = card.text
                    print(f"\n--- Card {idx} ---")
                    print(card_text[:200])  # Print first 200 chars for debugging

                    # TITLE - Try multiple selectors
                    title = "N/A"

                    # Try h3 with job-internship-name class
                    try:
                        title = card.find_element(By.XPATH, ".//h3[@class='job-internship-name']").text.strip()
                    except:
                        pass

                    # Try any h3
                    if title == "N/A":
                        try:
                            title = card.find_element(By.XPATH, ".//h3").text.strip()
                        except:
                            pass

                    # Try span with job title
                    if title == "N/A":
                        try:
                            title = card.find_element(By.XPATH, ".//span[@class='job-title']").text.strip()
                        except:
                            pass

                    # Try first link text as fallback
                    if title == "N/A":
                        try:
                            title = card.find_element(By.XPATH, ".//a").text.strip()
                            if len(title) > 100:  # If too long, reset
                                title = "N/A"
                        except:
                            pass

                    # COMPANY - Try multiple selectors
                    company = "N/A"

                    # Try p with company-name class
                    try:
                        company = card.find_element(By.XPATH, ".//p[@class='company-name']").text.strip()
                    except:
                        pass

                    # Try any company link
                    if company == "N/A":
                        try:
                            company = card.find_element(By.XPATH, ".//a[@class='company_link']").text.strip()
                        except:
                            pass

                    # Try any p tag
                    if company == "N/A":
                        try:
                            p_tags = card.find_elements(By.XPATH, ".//p")
                            if p_tags:
                                company = p_tags[0].text.strip()
                        except:
                            pass

                    # LOCATION - Try multiple selectors
                    location = "N/A"

                    # Try span with location_link class
                    try:
                        location_spans = card.find_elements(By.XPATH, ".//span[@class='location_link']")
                        if location_spans:
                            locations = [loc.text.strip() for loc in location_spans if loc.text.strip()]
                            location = ", ".join(locations) if locations else "N/A"
                    except:
                        pass

                    # Try links with location
                    if location == "N/A":
                        try:
                            location_links = card.find_elements(By.XPATH, ".//a[contains(@href,'location')]")
                            if location_links:
                                locations = [loc.text.strip() for loc in location_links if loc.text.strip()]
                                location = ", ".join(locations) if locations else "N/A"
                        except:
                            pass

                    # Try to find "Remote" text
                    if location == "N/A":
                        try:
                            if "Remote" in card_text or "remote" in card_text:
                                location = "Remote"
                        except:
                            pass

                    # SALARY - Try multiple selectors
                    salary = "Not Disclosed"

                    # Try span with stipend class
                    try:
                        salary = card.find_element(By.XPATH, ".//span[@class='stipend']").text.strip()
                    except:
                        pass

                    # Try any element with ₹ symbol
                    if salary == "Not Disclosed":
                        try:
                            salary_elem = card.find_element(By.XPATH, ".//*[contains(text(),'₹')]")
                            salary = salary_elem.text.strip()
                        except:
                            pass

                    # Try to extract from card text
                    if salary == "Not Disclosed" and "₹" in card_text:
                        for line in card_text.split('\n'):
                            if "₹" in line:
                                salary = line.strip()
                                break

                    # DESCRIPTION
                    description = f"{title} at {company}" if title != "N/A" else "Job posting"

                    # Create job object
                    job = JobPosting(
                        title=title,
                        company=company,
                        location=location,
                        salary_range=salary,
                        description=description,
                        source_platform="Internshala"
                    )

                    # Only add if we got at least a title
                    if title != "N/A" and len(title) > 3:
                        jobs.append(job)
                        print(f"✓ Added: {title} | {company} | {location} | {salary}")
                    else:
                        print(f"✗ Skipped: Title too short or N/A")

                except Exception as e:
                    print(f"CARD ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"SCRAPER ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("\nClosing browser...")
            driver.quit()

        print(f"\n✅ TOTAL JOBS SCRAPED: {len(jobs)}")
        return jobs


# Test the scraper
if __name__ == "__main__":
    scraper = InternshalaScraper()
    jobs = scraper.search_jobs(keyword="python", max_jobs=10)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(jobs)} jobs scraped")
    print(f"{'='*60}\n")
    
    for i, job in enumerate(jobs, 1):
        print(f"{i}. {job.title}")
        print(f"   Company: {job.company}")
        print(f"   Location: {job.location}")
        print(f"   Salary: {job.salary_range}")
        print()