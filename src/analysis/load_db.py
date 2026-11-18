"""Etapa 7 — Publicação: aplica o modelo registrado a todas as reviews e carrega o banco do produto.

Uso: python -m src.analysis.load_db
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config import DATA_PROCESSED, DB_PATH, MODEL_PATH, REPORTS_DIR
from src.db import connect, to_json


def main() -> None:
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PROCESSED / "reviews_clean.parquet")
    proba = model.predict_proba(df["content"].to_numpy())
    classes = list(model.classes_)
    df["label_modelo"] = [classes[i] for i in proba.argmax(axis=1)]
    df["p_negativo"] = proba[:, classes.index("NEGATIVO")]
    cl = pd.read_parquet(DATA_PROCESSED / "neg_clusters.parquet")
    df = df.merge(cl, on="reviewId", how="left")
    clusters = json.load(open(REPORTS_DIR / "clusters.json", encoding="utf-8"))["clusters"]

    with connect(DB_PATH) as con:
        con.execute("DELETE FROM reviews")
        con.execute("DELETE FROM complaint_clusters")
        con.executemany(
            "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r.reviewId, r.app_name, r.content, int(r.score), r.label, r.label_modelo, float(r.p_negativo),
              None if pd.isna(r.reviewCreatedVersion) else r.reviewCreatedVersion, r.version_major,
              r.at.isoformat(), r.month, int(r.thumbsUpCount), None if pd.isna(r.cluster) else int(r.cluster))
             for r in df.itertuples()],
        )
        con.executemany(
            "INSERT INTO complaint_clusters VALUES (?,?,?,?,?)",
            [(c["cluster_id"], c["nome"], to_json(c["top_termos"]), c["n"], to_json(c["exemplos"])) for c in clusters],
        )
        n = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"[load_db] {n} reviews e {len(clusters)} grupos gravados em {DB_PATH}")


if __name__ == "__main__":
    main()
