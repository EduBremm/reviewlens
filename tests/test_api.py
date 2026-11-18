import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires

pytestmark = requires("models/model.joblib")


@pytest.fixture(scope="module")
def client(tmp_db):
    from src.api import main

    main.DB_PATH = tmp_db
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(client):
    j = client.get("/model/info").json()
    assert {"modelo", "versao", "classes"} <= j.keys()


def test_predict_negative_with_alert_and_group(client):
    r = client.post("/predict", json={"text": "Terrible update, the app crashes and sync stopped working", "app": "Todoist"})
    assert r.status_code == 200
    j = r.json()
    assert j["sentimento"] == "NEGATIVO" and j["alerta_negativo"] is True
    assert abs(sum(j["probabilidades"].values()) - 1) < 1e-3
    assert j["grupo_reclamacao"] is not None and j["model_version"]


def test_predict_positive(client):
    j = client.post("/predict", json={"text": "Love this app, it keeps me organized every day!"}).json()
    assert j["sentimento"] == "POSITIVO" and j["alerta_negativo"] is False


def test_predict_invalid_payload(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_batch_and_history(client):
    r = client.post("/predict/batch", json={"reviews": [{"text": "great"}, {"text": "awful, crashes"}]})
    assert r.status_code == 200 and r.json()["n"] == 2
    hist = client.get("/predictions?limit=5").json()
    assert len(hist) >= 2 and "model_version" in hist[0]


@requires("data/reviewlens.db")
def test_insights_endpoints(client):
    s = client.get("/insights/summary").json()
    assert s["n"] > 1000 and 0 < s["neg"] < 1
    assert len(client.get("/insights/timeline").json()) > 6
    v = client.get("/insights/versions?min_reviews=20").json()
    assert v and v[0]["pct_negativo"] >= v[-1]["pct_negativo"]
    assert len(client.get("/insights/complaints").json()) >= 6
    assert client.get("/insights/terms?label=NEGATIVO&k=10").status_code == 200
    assert client.get("/insights/terms?label=XYZ").status_code == 422
    assert client.get("/insights/summary?app=Todoist").json()["n"] > 0
