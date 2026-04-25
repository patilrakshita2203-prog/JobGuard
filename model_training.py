"""Model training pipeline for JobGuard using LR, RF and XGBoost."""

import json
import joblib
import logging
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from data_cleaning import DataPreprocessor
from smote_balancer import SMOTEBalancer
from vectorizer import TFIDFExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = "data/fake_job_postings.csv"
RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance using classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
        ),
    }

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"{'=' * 50}")

    return metrics


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression baseline model."""
    logger.info("Training Logistic Regression...")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest intermediate model."""
    logger.info("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost final model."""
    logger.info("Training XGBoost...")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    return model


def main():
    """Run complete training pipeline and save best model."""
    logger.info("Starting JobGuard Model Training")
    logger.info(f"Time: {datetime.now().isoformat()}")

    logger.info("\n[1/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df, X_text, y = preprocessor.load_and_preprocess(DATA_PATH)

    logger.info("\n[2/5] Extracting TF-IDF features...")
    tfidf = TFIDFExtractor(max_features=10000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X_text)
    tfidf.save("models/tfidf_vectorizer.joblib")

    logger.info("\n[3/5] Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("\n[4/5] Applying SMOTE...")
    balancer = SMOTEBalancer(sampling_strategy=0.3, k_neighbors=5)
    X_train_balanced, y_train_balanced = balancer.fit_resample(
        X_train,
        y_train,
    )

    logger.info("\n[5/5] Training models...")

    models_to_train = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("XGBoost", train_xgboost),
    ]

    all_metrics = []
    best_model = None
    best_model_name = ""
    best_f1 = 0

    for model_name, train_function in models_to_train:
        model = train_function(X_train_balanced, y_train_balanced)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics.append(metrics)

        file_name = model_name.lower().replace(" ", "_")
        joblib.dump(model, f"models/{file_name}.joblib")

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model
            best_model_name = model_name

    joblib.dump(best_model, "models/best_model.joblib")

    results = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "best_f1": best_f1,
        "models": all_metrics,
    }

    with open("models/training_results.json", "w") as file:
        json.dump(results, file, indent=2)

    logger.info(f"Best Model: {best_model_name} | F1: {best_f1:.4f}")
    logger.info("Training completed successfully")

    return results


if __name__ == "__main__":
    results = main()

    print("\nFinal Results")
    for item in results["models"]:
        print(
            f"{item['model']:25s} | "
            f"F1: {item['f1_score']:.4f} | "
            f"AUC: {item['roc_auc']:.4f}"
        )
