"""Funções puras de texto (testadas em tests/test_features.py)."""
from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negações e intensificadores que a lista padrão do scikit-learn remove,
# mas que invertem/alteram o sentimento ("not good", "never works").
KEEP_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "cannot", "against",
    "very", "too", "much", "more", "most", "less", "least", "only", "but", "again",
    "off", "down", "up", "without", "cant", "dont", "doesnt", "didnt", "wont", "isnt",
}
STOPWORDS = frozenset(set(ENGLISH_STOP_WORDS) - KEEP_WORDS)

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NUM = re.compile(r"\d+")
_NON_ALPHA = re.compile(r"[^a-z' ]+")
_SPACES = re.compile(r"\s+")
_CONTRACTIONS = {
    "can't": "cant", "won't": "wont", "n't": "nt", "i'm": "im", "it's": "its",
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt", "isn't": "isnt",
}


def clean_text(text: str | None) -> str:
    """Normaliza um review: minúsculas, remove URL/e-mail/números/emoji, junta contrações."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    s = s.lower().replace("’", "'")
    s = _URL.sub(" ", s)
    s = _EMAIL.sub(" ", s)
    for k, v in _CONTRACTIONS.items():
        s = s.replace(k, v)
    s = _NUM.sub(" ", s)
    s = _NON_ALPHA.sub(" ", s).replace("'", "")
    return _SPACES.sub(" ", s).strip()


def score_to_label(score: int, label_map: dict | None = None) -> str:
    label_map = label_map or {1: "NEGATIVO", 2: "NEGATIVO", 3: "NEUTRO", 4: "POSITIVO", 5: "POSITIVO"}
    score = int(score)
    if score not in label_map:
        raise ValueError(f"score fora de 1..5: {score}")
    return label_map[score]


def major_version(version: str | None) -> str | None:
    """'5.4.0.6' -> '5.4' (agrupa builds da mesma versão)."""
    if version is None or (isinstance(version, float)) or str(version).strip() == "":
        return None
    parts = re.findall(r"\d+", str(version))
    if not parts:
        return None
    return ".".join(parts[:2]) if len(parts) > 1 else parts[0]
