import logging
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from data_cleaning import DataPreprocessor
from vectorizer import TFIDFExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = "data/fake_job_postings.csv"
RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    logger.info("=" * 50)
    logger.info(f"F1 Score  : {metrics['f1_score']}")
    logger.info(f"Precision : {metrics['precision']}")
    logger.info(f"Recall    : {metrics['recall']}")
    logger.info(f"ROC AUC   : {metrics['roc_auc']}")
    logger.info("=" * 50)

    return metrics


def get_bert_embeddings(texts):
    logger.info("Loading BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Generating BERT embeddings...")
    embeddings = model.encode(
        texts.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info(f"BERT embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    logger.info("Starting Hybrid Model Training")
    logger.info(f"Time: {datetime.now().isoformat()}")

    logger.info("[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df, X_text, y = preprocessor.load_and_preprocess(DATA_PATH)

    logger.info("[2/6] TF-IDF feature extraction...")
    tfidf = TFIDFExtractor(max_features=10000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X_text)
    tfidf.save("models/tfidf_vectorizer.joblib")

    logger.info("[3/6] BERT embeddings generation...")
    X_bert = get_bert_embeddings(df["raw_text"])

    logger.info("[4/6] Combining TF-IDF + BERT...")
    X_combined = hstack([
        X_tfidf,
        csr_matrix(X_bert)
    ])
    logger.info(f"Hybrid feature shape: {X_combined.shape}")

    logger.info("[5/6] Train test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("[6/6] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    joblib.dump(model, "models/best_model_hybrid.joblib")
    logger.info("Saved: models/best_model_hybrid.joblib")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "Hybrid TF-IDF + BERT + Logistic Regression",
        "metrics": metrics,
    }

    with open("models/hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
