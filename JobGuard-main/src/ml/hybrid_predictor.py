"""Hybrid (TF-IDF + BERT) predictor.

Uses the model that was actually trained on TF-IDF + BERT features
(best_model_hybrid.joblib). The old version loaded best_model.joblib, which only
knows the 10,000 TF-IDF columns, so appending BERT columns raised a shape error.

TF-IDF is fed the CLEANED text and BERT the RAW text, exactly as in
hybrid_model_training.py.
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class JobPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.bert_model = None
        self.cleaner = None

    def load(self):
        from sentence_transformers import SentenceTransformer  # heavy, so import lazily

        from src.preprocessing.data_cleaning import TextCleaner

        self.model = joblib.load(MODELS_DIR / "best_model_hybrid.joblib")
        hybrid_vectorizer = MODELS_DIR / "tfidf_vectorizer_hybrid.joblib"
        self.vectorizer = joblib.load(
            hybrid_vectorizer if hybrid_vectorizer.exists() else MODELS_DIR / "tfidf_vectorizer.joblib"
        )
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cleaner = TextCleaner()
        return self

    def _features(self, texts):
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Load model first using load()")

        cleaned = [self.cleaner.clean(t) for t in texts]
        tfidf_features = self.vectorizer.transform(cleaned)
        bert_embeddings = self.bert_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return hstack([tfidf_features, csr_matrix(bert_embeddings)])

    @staticmethod
    def _label(pred, prob):
        return {
            "prediction": "Fake Job" if pred == 1 else "Real Job",
            "probability": round(float(np.max(prob)) * 100, 2),
        }

    def predict(self, text):
        features = self._features([text])
        return self._label(self.model.predict(features)[0], self.model.predict_proba(features)[0])

    def predict_batch(self, texts):
        features = self._features(texts)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        return [self._label(p, pr) for p, pr in zip(predictions, probabilities)]
