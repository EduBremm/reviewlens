import numpy as np
import pytest

from src.features.text import STOPWORDS, clean_text, major_version, score_to_label
from src.features.transformers import VaderClassifier


@pytest.mark.parametrize("raw,expected", [
    ("I can't LOGIN!!! see https://x.com", "i cant login see"),
    ("Version 2.0 is bad :(", "version is bad"),
    ("  Muito    bom  ", "muito bom"),
    (None, ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_negations_are_kept():
    assert {"not", "no", "never"}.isdisjoint(STOPWORDS)


@pytest.mark.parametrize("score,label", [(1, "NEGATIVO"), (2, "NEGATIVO"), (3, "NEUTRO"), (4, "POSITIVO"), (5, "POSITIVO")])
def test_score_to_label(score, label):
    assert score_to_label(score) == label


def test_score_to_label_rejects_out_of_range():
    with pytest.raises(ValueError):
        score_to_label(0)


def test_major_version():
    assert major_version("5.4.0.6") == "5.4"
    assert major_version(None) is None
    assert major_version("v12") == "12"


def test_vader_baseline_polarity():
    clf = VaderClassifier().fit([])
    pred = clf.predict(np.array(["I love this app, amazing!", "Terrible, it crashes and I hate it"]))
    assert list(pred) == ["POSITIVO", "NEGATIVO"]
    p = clf.predict_proba(np.array(["good"]))
    assert np.isclose(p.sum(), 1)
