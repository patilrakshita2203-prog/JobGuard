import logging
import shap
import joblib
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SHAPExplainer:
    # SHAP explanation for fake job detection

    def __init__(self, model=None, vectorizer=None, feature_names=None):
        self.model = model
        self.vectorizer = vectorizer
        self.feature_names = feature_names
        self._explainer = None

    def build_explainer(self, X_background=None):
        # Build SHAP explainer
        print("Building SHAP Explainer...")

        if X_background is None:
            X_background = self.vectorizer.transform([
                "software engineer python backend developer remote job",
                "urgent hiring work from home registration fee required",
                "data analyst sql power bi fresher opportunity",
                "earn money fast no experience whatsapp immediately"
            ])

        self._explainer = shap.Explainer(
            self.model,
            X_background
        )

        print("SHAP Explainer Ready")
        return self

    @classmethod
    def load(cls, model_path, vectorizer_path):
        # Load saved model + vectorizer
        instance = cls()

        instance.model = joblib.load(model_path)
        instance.vectorizer = joblib.load(vectorizer_path)
        instance.feature_names = list(
            instance.vectorizer.get_feature_names_out()
        )

        instance.build_explainer()
        return instance

    def explain_single(self, text, cleaned_text):
        # Explain one job posting
        if self._explainer is None:
            raise RuntimeError("Call build_explainer() first")

        X = self.vectorizer.transform([cleaned_text])

        prob = self.model.predict_proba(X)[0][1]
        prediction = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob if prediction == "FAKE" else 1 - prob

        shap_values = self._explainer(X)

        shap_arr = shap_values.values[0]

        feature_shap = {}

        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if i < len(shap_arr) and abs(shap_arr[i]) > 0.001:
                    feature_shap[name] = float(shap_arr[i])

        sorted_features = sorted(
            feature_shap.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_fake = [(w, s) for w, s in sorted_features if s > 0][:10]
        top_real = [(w, abs(s)) for w, s in sorted_features if s < 0][:10]

        highlighted = self._highlight_text(
            text,
            top_fake,
            top_real
        )

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "probability_fake": round(float(prob), 4),
            "top_fake_words": top_fake,
            "top_real_words": top_real,
            "highlighted_html": highlighted,
            "feature_shap": dict(sorted_features[:20])
        }

    def _highlight_text(self, text, fake_words, real_words):
        # Highlight important words
        fake_set = {w.lower() for w, _ in fake_words}
        real_set = {w.lower() for w, _ in real_words}

        words = text.split()
        html_parts = []

        for word in words:
            clean_word = word.lower().strip(".,!?;:()")

            if clean_word in fake_set:
                html_parts.append(
                    f'<span style="background-color:#ffe0e0;'
                    f'color:#cc0000;font-weight:bold;'
                    f'border-radius:3px;padding:1px 3px;">'
                    f'🚨 {word}</span>'
                )

            elif clean_word in real_set:
                html_parts.append(
                    f'<span style="background-color:#e0ffe0;'
                    f'color:#006600;border-radius:3px;'
                    f'padding:1px 3px;">'
                    f'✅ {word}</span>'
                )

            else:
                html_parts.append(word)

        return " ".join(html_parts)

    def plot_summary(self, X_sample, save_path="reports/shap_summary.png"):
        # Save SHAP summary plot
        Path(save_path).parent.mkdir(exist_ok=True)

        shap_values = self._explainer(X_sample)

        plt.figure(figsize=(12, 8))

        shap.summary_plot(
            shap_values.values,
            X_sample,
            feature_names=(
                self.feature_names[:20]
                if self.feature_names
                else None
            ),
            max_display=20,
            show=False
        )

        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Summary plot saved: {save_path}")

    def get_trust_score(self, result):
        # Convert to trust score
        prob_genuine = 1 - result["probability_fake"]
        return int(prob_genuine * 100)