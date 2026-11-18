import pandas as pd

from src.config import ROOT, params
from src.data.download import sha256_of
from src.data.validate import check_schema, quality_report
from src.features.build import prepare
from tests.conftest import requires


def _toy():
    return pd.DataFrame({
        "reviewId": ["a", "b", "c", "d", "e", "f"],
        "userName": ["x"] * 6, "userImage": ["u"] * 6,
        "content": ["Great app!", "great app!", "Crashes all the time", "ok", "Crashes all the time", "meh it is fine"],
        "score": [5, 5, 1, 3, 4, 3], "thumbsUpCount": [0] * 6,
        "reviewCreatedVersion": ["1.2.3", None, "1.2.4", "1.3", "1.3", None],
        "at": ["2020-01-01 10:00:00"] * 6, "replyContent": [None] * 6, "repliedAt": [None] * 6,
        "sortOrder": ["newest"] * 6, "appId": ["com.x.todo"] * 6,
    })


def test_schema_ok_on_toy():
    assert check_schema(_toy()) == []


def test_schema_detects_bad_score():
    df = _toy()
    df.loc[0, "score"] = 9
    assert any("score" in e for e in check_schema(df))


def test_prepare_removes_pii_short_duplicates_and_ambiguous():
    out, log = prepare(_toy(), params()["features"])
    assert "userName" not in out.columns and "replyContent" not in out.columns
    assert "ok" not in out["content"].tolist()               # < 3 caracteres
    assert out["content"].str.lower().eq("great app!").sum() == 1   # duplicado mesma nota
    assert "Crashes all the time" not in out["content"].tolist()    # mesmo texto, notas 1 e 4 -> ambíguo
    assert set(out["label"]) <= {"NEGATIVO", "NEUTRO", "POSITIVO"}


def test_quality_report_counts():
    rep = quality_report(_toy())
    assert rep["linhas"] == 6 and rep["texto_duplicado"] == 2


@requires("data/raw/reviews.csv")
def test_raw_hash_matches_params():
    assert sha256_of(ROOT / params()["data"]["raw_path"]) == params()["data"]["sha256"]


@requires("data/processed/train.parquet")
def test_split_is_stratified_and_disjoint():
    tr = pd.read_parquet(ROOT / "data/processed/train.parquet")
    te = pd.read_parquet(ROOT / "data/processed/test.parquet")
    assert not set(tr.reviewId) & set(te.reviewId)
    d = (tr.label.value_counts(normalize=True) - te.label.value_counts(normalize=True)).abs().max()
    assert d < 0.01
