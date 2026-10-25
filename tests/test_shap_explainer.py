import joblib
import pytest

from conftest import FAKE_TEXT, GENUINE_TEXT, ROOT, needs_models
from src.preprocessing.data_cleaning import TextCleaner
from src.explainability.shap_explainer import SHAPExplainer

pytestmark = needs_models


@pytest.fixture(scope="module")
def explainer():
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
    return SHAPExplainer(
        model=model, vectorizer=vectorizer, feature_names=list(vectorizer.get_feature_names_out())
    ).build_explainer()


@pytest.mark.parametrize("text", [FAKE_TEXT, GENUINE_TEXT])
def test_explanations_only_use_words_that_are_in_the_posting(explainer, text):
    """Regression: the old 4-sentence baseline listed absent words ('earn', 'money') as trust indicators."""
    cleaned = TextCleaner().clean(text)
    result = explainer.explain_single(text, cleaned)
    words = [w for w, _ in result["top_fake_words"] + result["top_real_words"]]
    assert words, "expected some explanation words"
    for word in words:
        assert word in cleaned, f"{word!r} is not in the posting"


def test_prediction_is_unchanged_by_the_baseline(explainer):
    cleaned = TextCleaner().clean(GENUINE_TEXT)
    result = explainer.explain_single(GENUINE_TEXT, cleaned)
    assert result["prediction"] == "REAL"
    assert result["probability_fake"] < 0.5
