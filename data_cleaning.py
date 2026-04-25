"""
JobGuard - Text Preprocessing Pipeline
Handles cleaning, feature engineering, and preprocessing
for fake job detection using NLP + ML.
"""

import re
import string
import logging
import unicodedata
from typing import Tuple

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# -------------------------------
# NLTK Downloads
# -------------------------------
REQUIRED_NLTK_PACKAGES = [
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "averaged_perceptron_tagger"
]

for pkg in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# Scam Indicators
# -------------------------------
SCAM_INDICATORS = [
    "urgent",
    "immediate",
    "registration fee",
    "investment required",
    "work from home earn",
    "no experience needed",
    "guaranteed income",
    "unlimited earning",
    "part time earn",
    "easy money",
    "click here",
    "apply now limited",
    "100% job guarantee",
    "free laptop",
    "government approved",
    "direct joining",
    "no interview",
    "whatsapp to apply",
    "telegram group",
    "training fee",
    "security deposit",
    "advance payment",
    "wire transfer",
    "western union",
    "bitcoin",
    "lottery",
    "prize money"
]


# -------------------------------
# Text Cleaner Class
# -------------------------------
class TextCleaner:
    """
    Cleans raw job posting text and extracts useful fraud-related features.
    """

    def __init__(self, language: str = "english"):
        self.stop_words = set(stopwords.words(language))

        # Keep negation words (important for meaning)
        negation_words = {
            "no", "not", "never", "none",
            "neither", "nor", "cannot",
            "can't", "won't"
        }
        self.stop_words -= negation_words

        self.lemmatizer = WordNetLemmatizer()

        self.scam_pattern = re.compile(
            "|".join(re.escape(word) for word in SCAM_INDICATORS),
            re.IGNORECASE
        )

    def remove_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&nbsp;", " ", text)
        return text

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode(
            "ascii", "ignore"
        ).decode("ascii")

    def remove_urls_and_emails(self, text: str) -> str:
        text = re.sub(r"http[s]?://\S+", " URL ", text)
        text = re.sub(r"www\.\S+", " URL ", text)
        text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)
        return text

    def remove_special_characters(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_and_clean(self, text: str) -> str:
        tokens = word_tokenize(text.lower())

        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
            and token not in string.punctuation
            and len(token) > 2
            and not token.isdigit()
        ]

        return " ".join(cleaned_tokens)

    def count_scam_indicators(self, text: str) -> int:
        return len(self.scam_pattern.findall(text))

    def extract_features(self, text: str) -> dict:
        words = text.split()

        return {
            "scam_indicator_count": self.count_scam_indicators(text),
            "has_url": int(bool(re.search(r"http|www", text, re.I))),
            "has_email": int(bool(re.search(r"\S+@\S+\.\S+", text))),
            "has_whatsapp": int(bool(re.search(r"whatsapp|telegram", text, re.I))),
            "has_fee_mention": int(bool(re.search(r"fee|deposit|pay|invest", text, re.I))),
            "has_urgent": int(bool(re.search(r"urgent|immediate|asap", text, re.I))),
            "exclamation_count": text.count("!"),
            "all_caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0
        }

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.remove_html(text)
        text = self.normalize_unicode(text)
        text = self.remove_urls_and_emails(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        text = self.tokenize_and_clean(text)

        return text


# -------------------------------
# Data Preprocessor Class
# -------------------------------
class DataPreprocessor:
    """
    Full preprocessing pipeline for fake job posting dataset.
    """

    TEXT_COLUMNS = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    def __init__(self):
        self.cleaner = TextCleaner()

    def combine_text_fields(self, df: pd.DataFrame) -> pd.Series:
        return df[self.TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing {len(df)} records...")

        df = df.copy()

        # Combine text
        df["combined_text"] = self.combine_text_fields(df)
        df["raw_text"] = df["combined_text"].copy()

        # Clean text
        df["cleaned_text"] = df["combined_text"].apply(
            self.cleaner.clean
        )

        # Feature engineering
        feature_df = df["raw_text"].apply(
            self.cleaner.extract_features
        ).apply(pd.Series)

        df = pd.concat([df, feature_df], axis=1)

        # Additional binary features
        df["has_salary_range"] = (
            ~df.get("salary_range", pd.Series(index=df.index)).isna()
        ).astype(int)

        df["has_company_logo"] = df.get(
            "has_company_logo",
            pd.Series(0, index=df.index)
        ).fillna(0).astype(int)

        df["telecommuting"] = df.get(
            "telecommuting",
            pd.Series(0, index=df.index)
        ).fillna(0).astype(int)

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")

        return df

    def load_and_preprocess(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = pd.read_csv(filepath)

        logger.info(
            f"Loaded dataset: {len(df)} rows | Fake jobs: {df['fraudulent'].sum()}"
        )

        df = self.preprocess_dataframe(df)

        # Save processed file
        df.to_csv("data/processed_jobs.csv", index=False)
        logger.info("Processed dataset saved to data/processed_jobs.csv")

        X = df["cleaned_text"]
        y = df["fraudulent"]

        return df, X, y


# -------------------------------
# Main Run
# -------------------------------
if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    df, X, y = preprocessor.load_and_preprocess(
        "data/fake_job_postings.csv"
    )

    print("\nDataset Ready Successfully")
    print(f"Shape: {df.shape}")
    print(f"Fake Jobs Count: {y.sum()}")
    print("\nSample Cleaned Text:\n")
    print(X.iloc[0][:300])
