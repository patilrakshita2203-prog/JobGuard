import sys
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Root path access
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from predictor import JobPredictor
from shap_explainer import SHAPExplainer

# Load prediction model
predictor = JobPredictor.load(
    model_path="models/best_model.joblib",
    vectorizer_path="models/tfidf_vectorizer.joblib"
)

# Load SHAP explainer
explainer = SHAPExplainer.load(
    model_path="models/best_model.joblib",
    vectorizer_path="models/tfidf_vectorizer.joblib"
)

# Chrome setup
options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

# Open Naukri jobs page
url = "https://www.naukri.com/software-engineer-jobs?k=software%20engineer"
driver.get(url)

time.sleep(10)

# Find job cards
cards = driver.find_elements(
    By.XPATH,
    "//div[contains(@class,'cust-job-tuple')]"
)

print(f"\nCards found: {len(cards)}")
print("\nReal Time Jobs + Fake Job Detection:\n")

count = 0

for card in cards[:10]:
    try:
        # Full scraped text
        job_text = card.text.strip().lower()

        # Remove noisy/common words
        remove_words = [
            "ago",
            "day",
            "days",
            "month",
            "months",
            "year",
            "years",
            "posted",
            "today",
            "apply",
            "save",
            "reviews",
            "review",
            "few hours",
            "few",
            "hours",
            "full",
            "minimum",
            "type",
            "click",
            "job",
            "description",
            "alignment",
            "clear",
            "internship"
        ]

        for word in remove_words:
            job_text = job_text.replace(word, "")

        # Remove symbols and numbers
        job_text = re.sub(r"[^a-zA-Z\s]", " ", job_text)

        # Clean extra spaces
        job_text = " ".join(job_text.split())

        # Fake keyword checking
        fake_keywords = [
            "registration fee",
            "payment required",
            "urgent hiring",
            "whatsapp",
            "telegram",
            "limited seats",
            "investment required",
            "earn money fast",
            "work from home",
            "guaranteed income",
            "no experience needed",
            "direct joining",
            "advance payment"
        ]

        for keyword in fake_keywords:
            if keyword in job_text:
                print(f"⚠ Suspicious Keyword Found: {keyword}")

        # First line as title
        title = card.text.strip().split("\n")[0]

        if not title:
            continue

        print("=" * 60)
        print(f"Job Title: {title}")

        # Prediction
        result = predictor.predict(job_text)

        print(f"Prediction   : {result['prediction']}")
        print(f"Confidence   : {result['confidence']}%")

        # SHAP explanation
        shap_result = explainer.explain_single(
            job_text,
            job_text
        )

        print(
            f"Trust Score  : "
            f"{explainer.get_trust_score(shap_result)}"
        )

        print("Top Important Words:")

        if shap_result["top_fake_words"]:
            for word, score in shap_result["top_fake_words"][:3]:
                print(f"⚠ {word} ({round(score, 3)})")

        elif shap_result["top_real_words"]:
            for word, score in shap_result["top_real_words"][:3]:
                print(f"✅ {word} ({round(score, 3)})")

        print("-" * 60)

        count += 1

    except Exception as e:
        print("Error:", e)

print(f"\nTotal Jobs Processed: {count}")

input("\nPress Enter to close browser...")
driver.quit()