"""Feature extraction using TF-IDF and BERT embeddings."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class TFIDFExtractor:
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )
        self._fitted = False

    def fit_transform(self, texts):
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        logger.info(
            f"TF-IDF matrix: {matrix.shape} | Vocabulary: {len(self.vectorizer.vocabulary_)}"
        )
        return matrix

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: str = "models/tfidf_vectorizer.joblib"):
        joblib.dump(self.vectorizer, path)
        logger.info(f"TF-IDF vectorizer saved to {path}")

    @classmethod
    def load(cls, path: str = "models/tfidf_vectorizer.joblib"):
        instance = cls()
        instance.vectorizer = joblib.load(path)
        instance._fitted = True
        return instance


class BERTEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers using: pip install sentence-transformers"
                )
        return self._model

    def encode(self, texts, show_progress: bool = True):
        model = self._load_model()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"BERT embeddings shape: {embeddings.shape}")
        return embeddings

    def save_embeddings(
        self,
        embeddings,
        path: str = "models/bert_embeddings.npy",
    ):
        np.save(path, embeddings)
        logger.info(f"Embeddings saved to {path}")

    def load_embeddings(self, path: str = "models/bert_embeddings.npy"):
        return np.load(path)


class HybridVectorizer:
    def __init__(self, use_bert: bool = False):
        self.tfidf = TFIDFExtractor()
        self.use_bert = use_bert
        self.bert = BERTEmbedder() if use_bert else None

    def fit_transform(self, texts, raw_texts=None):
        tfidf_matrix = self.tfidf.fit_transform(texts)

        if self.use_bert:
            from scipy.sparse import csr_matrix, hstack

            bert_input = raw_texts if raw_texts is not None else texts
            bert_matrix = self.bert.encode(bert_input)
            combined_matrix = hstack([
                tfidf_matrix,
                csr_matrix(bert_matrix),
            ])

            logger.info(f"Hybrid matrix shape: {combined_matrix.shape}")
            return combined_matrix

        return tfidf_matrix

    def transform(self, texts, raw_texts=None):
        tfidf_matrix = self.tfidf.transform(texts)

        if self.use_bert:
            from scipy.sparse import csr_matrix, hstack

            bert_input = raw_texts if raw_texts is not None else texts
            bert_matrix = self.bert.encode(
                bert_input,
                show_progress=False,
            )

            return hstack([
                tfidf_matrix,
                csr_matrix(bert_matrix),
            ])

        return tfidf_matrix


if __name__ == "__main__":
    print("Vectorizer module ready.")
