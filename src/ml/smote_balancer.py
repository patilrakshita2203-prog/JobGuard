import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse, hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SMOTEBalancer:
    def __init__(
        self,
        sampling_strategy: float = 0.5,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

        self.smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )

    def fit_resample(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"Before SMOTE | Total: {len(y)} | "
            f"Fake: {y.sum()} | Genuine: {(y == 0).sum()}"
        )

        if issparse(X):
            logger.info("Converting sparse matrix to dense for SMOTE...")
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        X_resampled, y_resampled = self.smote.fit_resample(X_dense, y)

        logger.info(
            f"After SMOTE | Total: {len(y_resampled)} | "
            f"Fake: {y_resampled.sum()} | Genuine: {(y_resampled == 0).sum()}"
        )

        logger.info(
            f"Imbalance ratio improved: {y.mean():.3f} -> "
            f"{y_resampled.mean():.3f}"
        )

        return X_resampled, y_resampled

    def get_class_distribution(self, y: np.ndarray) -> dict:
        total = len(y)
        fake = int(y.sum())
        genuine = int(total - fake)

        if fake == 0:
            imbalance_ratio = 0
        else:
            imbalance_ratio = round(genuine / fake, 2)

        return {
            "total": total,
            "fake": fake,
            "genuine": genuine,
            "fake_pct": round((fake / total) * 100, 2),
            "genuine_pct": round((genuine / total) * 100, 2),
            "imbalance_ratio": imbalance_ratio,
        }


class FeatureEngineer:
    NUMERIC_FEATURES = [
        "scam_indicator_count",
        "has_url",
        "has_email",
        "has_whatsapp",
        "has_fee_mention",
        "has_urgent",
        "exclamation_count",
        "all_caps_ratio",
        "word_count",
        "avg_word_length",
        "has_salary_range",
        "has_company_logo",
        "telecommuting",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, tfidf_matrix, df: pd.DataFrame):
        available_features = [
            feature for feature in self.NUMERIC_FEATURES if feature in df.columns
        ]

        numeric_data = df[available_features].fillna(0)
        numeric_scaled = self.scaler.fit_transform(numeric_data)

        self._fitted = True

        combined_matrix = hstack([
            tfidf_matrix,
            csr_matrix(numeric_scaled),
        ])

        logger.info(
            f"Feature matrix shape: {combined_matrix.shape} | "
            f"TF-IDF: {tfidf_matrix.shape[1]} + "
            f"Numeric: {numeric_scaled.shape[1]}"
        )

        return combined_matrix

    def transform(self, tfidf_matrix, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        available_features = [
            feature for feature in self.NUMERIC_FEATURES if feature in df.columns
        ]

        numeric_data = df[available_features].fillna(0)
        numeric_scaled = self.scaler.transform(numeric_data)

        combined_matrix = hstack([
            tfidf_matrix,
            csr_matrix(numeric_scaled),
        ])

        return combined_matrix


if __name__ == "__main__":
    print("SMOTE Balancer and Feature Engineer module ready.")
