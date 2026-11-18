import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from src.config import ROOT
from tests.conftest import requires


@requires("models/model.joblib")
def test_model_meets_minimum_on_test():
    m = json.load(open(ROOT / "reports/metrics.json"))
    assert m["metricas"]["f1_macro"] >= 0.60
    assert m["metricas"]["roc_auc_ovr"] >= 0.80


@requires("models/model.joblib")
def test_model_beats_baselines_in_cv():
    cv = json.load(open(ROOT / "reports/cv_results.json"))["resultados"]
    best = max(cv, key=lambda k: cv[k]["f1_macro"])
    assert cv[best]["f1_macro"] > cv["vader"]["f1_macro"] > cv["dummy"]["f1_macro"]


@requires("models/model.joblib")
def test_model_is_deterministic():
    model = joblib.load(ROOT / "models/model.joblib")
    tr = pd.read_parquet(ROOT / "data/processed/train.parquet").sample(1500, random_state=0)
    a = clone(model).fit(tr.content.to_numpy(), tr.label.to_numpy())
    b = clone(model).fit(tr.content.to_numpy(), tr.label.to_numpy())
    x = np.array(["love it", "it crashes", "it is ok i guess"])
    assert np.allclose(a.predict_proba(x), b.predict_proba(x))


@requires("models/model.joblib")
def test_sanity_obvious_examples():
    model = joblib.load(ROOT / "models/model.joblib")
    pred = model.predict(np.array(["Best app ever, love it, very useful",
                                   "Worst app, crashes all the time, waste of money, uninstalling"]))
    assert list(pred) == ["POSITIVO", "NEGATIVO"]


@requires("models/model.joblib")
def test_metrics_report_matches_model():
    model = joblib.load(ROOT / "models/model.joblib")
    te = pd.read_parquet(ROOT / "data/processed/test.parquet")
    f1 = f1_score(te.label, model.predict(te.content.to_numpy()), average="macro")
    m = json.load(open(ROOT / "reports/metrics.json"))
    assert abs(f1 - m["metricas"]["f1_macro"]) < 1e-3
