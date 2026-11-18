import numpy as np

from src.models.candidates import build_candidates


def test_candidates_cover_theme_algorithms():
    c = build_candidates(42)
    assert {"naive_bayes", "svm", "logreg", "vader", "spacy_vectors_lr", "dummy"} <= c.keys()
    for name, (desc, pipe, grid) in c.items():
        assert desc and hasattr(pipe, "fit") and isinstance(grid, dict)


def test_tfidf_candidate_fits_on_raw_text():
    _, pipe, _ = build_candidates(42)["naive_bayes"]
    X = np.array(["love it!", "great app", "hate it, crashes", "awful bug", "it is ok", "fine I guess"] * 3)
    y = np.array(["POSITIVO", "POSITIVO", "NEGATIVO", "NEGATIVO", "NEUTRO", "NEUTRO"] * 3)
    pipe.fit(X, y)
    assert pipe.predict(np.array(["I love this app"]))[0] == "POSITIVO"
    assert np.allclose(pipe.predict_proba(X).sum(axis=1), 1)
