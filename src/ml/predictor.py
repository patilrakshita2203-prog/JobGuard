import joblib

class JobPredictor:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    @classmethod
    def load(cls, model_path, vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return cls(model, vectorizer)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        return {
            "prediction": "FAKE" if pred == 1 else "REAL",
            "confidence": round(max(prob) * 100, 2)
        }