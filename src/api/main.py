"""ReviewLens API — classificação de sentimento e indicadores de satisfação.

Rodar: uvicorn src.api.main:app --reload   (Swagger em http://localhost:8000/docs)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query

from src.analysis import insights
from src.api.schemas import BatchIn, BatchOut, PredictionOut, ReviewIn
from src.config import DB_PATH, MODEL_PATH, MODELS_DIR
from src.db import connect, insert_prediction, rows


class ModelService:
    """Carrega o modelo versionado + modelo de grupos e expõe predição explicada."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.card = json.load(open(MODELS_DIR / "model_card.json", encoding="utf-8"))
        self.classes = list(self.model.classes_)
        self.threshold = float(self.card.get("limiar_alerta_negativo", 0.5))
        cpath = MODELS_DIR / "clusters.joblib"
        self.clusters = joblib.load(cpath) if cpath.exists() else None
        self._vec = self.model.named_steps.get("tfidf")
        self._clf = self.model.named_steps.get("clf")

    def explain(self, text: str, label: str, k: int = 5) -> list[str]:
        """Termos do texto com maior contribuição (tf-idf × coeficiente) para a classe prevista."""
        if self._vec is None or not hasattr(self._clf, "coef_"):
            return []
        x = self._vec.transform([text])
        coef = self._clf.coef_[self.classes.index(label)]
        contrib = x.multiply(coef).tocsr()
        idx, vals = contrib.indices, contrib.data
        names = self._vec.get_feature_names_out()
        order = np.argsort(vals)[::-1]
        return [names[idx[i]] for i in order[:k] if vals[i] > 0]

    def predict(self, texts: list[str]) -> list[dict]:
        proba = self.model.predict_proba(np.array(texts, dtype=object))
        out = []
        for t, pr in zip(texts, proba):
            label = self.classes[int(pr.argmax())]
            probs = {c: round(float(p), 4) for c, p in zip(self.classes, pr)}
            alerta = probs["NEGATIVO"] >= self.threshold
            grupo = cid = None
            if self.clusters is not None and (label == "NEGATIVO" or alerta):
                from src.features.text import clean_text

                cid = int(self.clusters["pipeline"].predict([clean_text(t)])[0])
                grupo = self.clusters["nomes"].get(cid)
            out.append({"sentimento": label, "probabilidades": probs, "confianca": round(float(pr.max()), 4),
                        "alerta_negativo": bool(alerta), "grupo_reclamacao": grupo, "cluster_id": cid,
                        "termos_influentes": self.explain(t, label), "model_version": self.card["versao"]})
        return out


@lru_cache(maxsize=1)
def service() -> ModelService:
    return ModelService()


@asynccontextmanager
async def lifespan(_app):
    service()  # carrega o modelo uma vez na subida
    yield


app = FastAPI(title="ReviewLens API", version="1.0.0", lifespan=lifespan,
              description="Mineração de sentimentos em reviews de aplicativos (tema 9).")


def _save_and_build(con, req: ReviewIn, r: dict) -> PredictionOut:
    pid = insert_prediction(con, text=req.text, app=req.app, label=r["sentimento"], probs=r["probabilidades"],
                            alerta=r["alerta_negativo"], cluster_id=r["cluster_id"], model_version=r["model_version"])
    return PredictionOut(id=pid, **{k: v for k, v in r.items() if k != "cluster_id"})


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", "modelo_carregado": s.model is not None, "banco": str(DB_PATH)}


@app.get("/model/info")
def model_info():
    return service().card


@app.post("/predict", response_model=PredictionOut)
def predict(req: ReviewIn):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="texto vazio")
    r = service().predict([req.text])[0]
    with connect(DB_PATH) as con:
        return _save_and_build(con, req, r)


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(req: BatchIn):
    res = service().predict([x.text for x in req.reviews])
    with connect(DB_PATH) as con:
        out = [_save_and_build(con, x, r) for x, r in zip(req.reviews, res)]
    dist: dict[str, int] = {}
    for o in out:
        dist[o.sentimento] = dist.get(o.sentimento, 0) + 1
    return BatchOut(n=len(out), distribuicao=dist, resultados=out)


@app.get("/predictions")
def predictions(limit: int = Query(50, ge=1, le=1000)):
    with connect(DB_PATH) as con:
        return rows(con, "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/apps")
def list_apps():
    with connect(DB_PATH) as con:
        return insights.apps(con)


@app.get("/insights/summary")
def ins_summary(app_name: str | None = Query(None, alias="app")):
    with connect(DB_PATH) as con:
        return insights.summary(con, app_name)


@app.get("/insights/timeline")
def ins_timeline(app_name: str | None = Query(None, alias="app"), since: str = "2019-01"):
    with connect(DB_PATH) as con:
        return insights.timeline(con, app_name, since)


@app.get("/insights/versions")
def ins_versions(app_name: str | None = Query(None, alias="app"), min_reviews: int = Query(20, ge=1),
                 limit: int = Query(15, ge=1, le=200)):
    with connect(DB_PATH) as con:
        return insights.versions(con, app_name, min_reviews, limit)


@app.get("/insights/complaints")
def ins_complaints(app_name: str | None = Query(None, alias="app")):
    with connect(DB_PATH) as con:
        return insights.complaints(con, app_name)


@app.get("/insights/terms")
def ins_terms(app_name: str | None = Query(None, alias="app"), label: str = "NEGATIVO", k: int = Query(25, le=100)):
    if label not in ("NEGATIVO", "NEUTRO", "POSITIVO"):
        raise HTTPException(status_code=422, detail="label inválido")
    with connect(DB_PATH) as con:
        return insights.terms(con, app_name, label, k)
