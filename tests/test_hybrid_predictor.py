import pytest


def test_hybrid_predictor_matches_the_hybrid_model_shape(monkeypatch):
    """Regression: the predictor appended BERT columns to a TF-IDF-only model (shape error)."""
    import numpy as np
    import joblib
    from conftest import ROOT

    hybrid = ROOT / "models" / "best_model_hybrid.joblib"
    vec = ROOT / "models" / "tfidf_vectorizer.joblib"
    if not (hybrid.exists() and vec.exists()):
        pytest.skip("hybrid model files not present")

    from src.ml import hybrid_predictor as module

    class FakeBert:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384))

    from src.preprocessing.data_cleaning import TextCleaner

    p = module.JobPredictor()
    p.model, p.vectorizer = joblib.load(hybrid), joblib.load(vec)
    p.bert_model, p.cleaner = FakeBert(), TextCleaner()
    out = p.predict("urgent hiring registration fee whatsapp")
    assert out["prediction"] in ("Fake Job", "Real Job")
    assert len(p.predict_batch(["a job", "another job"])) == 2
