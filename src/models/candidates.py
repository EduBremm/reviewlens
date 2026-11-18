"""Definição dos modelos candidatos e de suas grades de hiperparâmetros.

Todo pré-processamento fica DENTRO do Pipeline, então o TF-IDF/escalonamento é
ajustado só nas dobras de treino da validação cruzada (sem vazamento).
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.features.text import STOPWORDS, clean_text
from src.features.transformers import SpacyVectorizer, VaderClassifier, VaderFeatures


def tfidf(**kw) -> TfidfVectorizer:
    base = dict(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                stop_words=list(STOPWORDS), token_pattern=r"(?u)\b\w\w+\b")
    base.update(kw)
    return TfidfVectorizer(**base)


def build_candidates(seed: int) -> dict:
    """nome -> (descrição, pipeline, grade). Todos recebem o texto ORIGINAL do review."""
    return {
        "dummy": (
            "Baseline: sempre a classe majoritária",
            Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
            {},
        ),
        "vader": (
            "Pré-treinado zero-shot: léxico VADER (limiares ±0,05)",
            Pipeline([("clf", VaderClassifier())]),
            {},  # VADER usa pontuação/maiúsculas -> recebe o texto original
        ),
        "naive_bayes": (
            "Complement Naive Bayes + TF-IDF",
            Pipeline([("tfidf", tfidf()), ("clf", ComplementNB())]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.3, 1.0]},
        ),
        "svm": (
            "SVM linear + TF-IDF (calibrado p/ probabilidade)",
            Pipeline([("tfidf", tfidf()),
                      ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed),
                                                     cv=3, method="sigmoid"))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__estimator__C": [0.05, 0.1, 0.5, 1.0]},
        ),
        "logreg": (
            "Regressão Logística + TF-IDF",
            Pipeline([("tfidf", tfidf()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.5, 1.0, 2.0, 5.0]},
        ),
        "spacy_vectors_lr": (
            "Pré-treinado: vetores de palavra spaCy (300d) + Regressão Logística",
            Pipeline([("vec", SpacyVectorizer()), ("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]),
            {"clf__C": [0.01, 0.1, 1.0]},
        ),
        "hybrid_tfidf_vader": (
            "Híbrido: TF-IDF + atributos VADER (pré-treinado) + Regressão Logística",
            Pipeline([("feats", FeatureUnion([("tfidf", tfidf()), ("vader", VaderFeatures())])),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]),
            {"feats__tfidf__ngram_range": [(1, 2)], "clf__C": [0.5, 1.0, 2.0]},
        ),
    }
