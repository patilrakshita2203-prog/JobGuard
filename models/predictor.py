import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix


class JobPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.bert_model = None

    def load(self):
        self.model = joblib.load("models/best_model.joblib")
        self.vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self

    def predict(self, text):
        if not self.model or not self.vectorizer:
            raise RuntimeError("Load model first using load()")

        tfidf_features = self.vectorizer.transform([text])

        bert_embedding = self.bert_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        combined_features = hstack([
            tfidf_features,
            csr_matrix(bert_embedding)
        ])

        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]

        return {
            "prediction": "Fake Job" if prediction == 1 else "Real Job",
            "probability": round(float(np.max(probability)) * 100, 2)
        }

    def predict_batch(self, texts):
        results = []

        tfidf_features = self.vectorizer.transform(texts)

        bert_embeddings = self.bert_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        combined_features = hstack([
            tfidf_features,
            csr_matrix(bert_embeddings)
        ])

        predictions = self.model.predict(combined_features)
        probabilities = self.model.predict_proba(combined_features)

        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": "Fake Job" if pred == 1 else "Real Job",
                "probability": round(float(np.max(prob)) * 100, 2)
            })

        return results