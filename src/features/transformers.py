"""Transformadores scikit-learn usados nos pipelines (precisam ser importáveis para o joblib)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from src.config import LABELS


class VaderFeatures(BaseEstimator, TransformerMixin):
    """Pontuações do VADER (léxico pré-treinado): neg, neu, pos, compound."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        rows = [sia.polarity_scores(str(t)) for t in X]
        return np.array([[r["neg"], r["neu"], r["pos"], r["compound"]] for r in rows])


class VaderClassifier(BaseEstimator, ClassifierMixin):
    """Classificador zero-shot: limiares sobre o compound do VADER (Hutto & Gilbert, 2014)."""

    def __init__(self, neg_threshold: float = -0.05, pos_threshold: float = 0.05):
        self.neg_threshold = neg_threshold
        self.pos_threshold = pos_threshold

    def fit(self, X, y=None):
        self.classes_ = np.array(LABELS)
        return self

    def _compound(self, X):
        return VaderFeatures().transform(X)[:, 3]

    def predict(self, X):
        c = self._compound(X)
        out = np.where(c <= self.neg_threshold, "NEGATIVO",
                       np.where(c >= self.pos_threshold, "POSITIVO", "NEUTRO"))
        return out

    def predict_proba(self, X):
        # pseudo-probabilidade a partir do compound (para ROC/PR); não calibrada
        c = self._compound(X)
        neg = np.clip(-c, 0, 1)
        pos = np.clip(c, 0, 1)
        neu = 1 - np.abs(c)
        p = np.vstack([neg, neu, pos]).T + 1e-6
        return p / p.sum(axis=1, keepdims=True)


class SpacyVectorizer(BaseEstimator, TransformerMixin):
    """Média dos vetores de palavra pré-treinados do spaCy (en_core_web_md, 300 dim)."""

    def __init__(self, model: str = "en_core_web_md"):
        self.model = model

    def _nlp(self):
        if not hasattr(self, "_nlp_cache"):
            import spacy

            self._nlp_cache = spacy.load(self.model, exclude=["parser", "ner", "lemmatizer",
                                                              "attribute_ruler", "tagger", "senter"])
        return self._nlp_cache

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = self._nlp()
        return np.vstack([nlp.make_doc(str(t)).vector for t in X])

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_nlp_cache", None)
        return state
