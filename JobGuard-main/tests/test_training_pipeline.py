"""End-to-end run of model_training.main() on a tiny synthetic dataset.

Needs the training-only packages (xgboost, imbalanced-learn); skipped where they are
not installed (e.g. the default CI job).
"""

import random

import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("imblearn")


def make_dataset(n=400, n_fake=40):
    rng = random.Random(0)
    fake_words = "urgent registration fee whatsapp earn money guaranteed income deposit telegram".split()
    real_words = "software engineer python backend team experience interview insurance agile review".split()
    rows = []
    for i in range(n):
        fake = i < n_fake
        pool = fake_words if fake else real_words
        text = " ".join(rng.choices(pool, k=14) + rng.choices(real_words + fake_words, k=3))
        rows.append({
            "title": f"{rng.choice(pool)} role {i}", "company_profile": "", "description": text,
            "requirements": rng.choice(pool), "benefits": rng.choice(pool), "fraudulent": int(fake),
        })
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)


def test_training_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    make_dataset().to_csv(tmp_path / "data" / "fake_job_postings.csv", index=False)

    import src.ml.model_training as mt

    fit_sizes = []
    original = mt.TFIDFExtractor.fit_transform

    def spy(self, texts):
        fit_sizes.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(mt.TFIDFExtractor, "fit_transform", spy)
    results = mt.main()

    # TF-IDF must be fitted on the 80% training split only (no test-set leakage)
    assert fit_sizes == [320]

    by_name = {m["model"]: m for m in results["models"]}
    assert set(by_name) == {"Logistic Regression", "Random Forest", "XGBoost"}

    # Regression: XGBoost used to predict every posting as fake (AUC < 0.5) because it was
    # trained on dense arrays but scored on sparse ones.
    xgb = by_name["XGBoost"]
    assert xgb["roc_auc"] > 0.9 and xgb["f1_score"] > 0.5
    assert (tmp_path / "models" / "best_model.joblib").exists()
