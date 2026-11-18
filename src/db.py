"""Persistência em SQLite (arquivo único; troca por PostgreSQL mudando só a conexão)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id      TEXT PRIMARY KEY,
    app            TEXT NOT NULL,
    content        TEXT NOT NULL,
    score          INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    label_estrelas TEXT NOT NULL,
    label_modelo   TEXT NOT NULL,
    p_negativo     REAL NOT NULL,
    version        TEXT,
    version_major  TEXT,
    at             TEXT NOT NULL,
    month          TEXT NOT NULL,
    thumbs_up      INTEGER DEFAULT 0,
    cluster_id     INTEGER
);
CREATE INDEX IF NOT EXISTS ix_reviews_app_month ON reviews(app, month);
CREATE INDEX IF NOT EXISTS ix_reviews_version ON reviews(app, version_major);

CREATE TABLE IF NOT EXISTS complaint_clusters (
    cluster_id INTEGER PRIMARY KEY,
    nome       TEXT NOT NULL,
    top_termos TEXT NOT NULL,   -- JSON
    n          INTEGER NOT NULL,
    exemplos   TEXT NOT NULL    -- JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    app           TEXT,
    text          TEXT NOT NULL,
    label         TEXT NOT NULL,
    p_negativo    REAL NOT NULL,
    p_neutro      REAL NOT NULL,
    p_positivo    REAL NOT NULL,
    alerta        INTEGER NOT NULL,
    cluster_id    INTEGER,
    model_version TEXT NOT NULL
);
"""


@contextmanager
def connect(path: Path | str | None = None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def insert_prediction(con, *, text: str, app: str | None, label: str, probs: dict, alerta: bool,
                      cluster_id: int | None, model_version: str) -> int:
    cur = con.execute(
        "INSERT INTO predictions (created_at, app, text, label, p_negativo, p_neutro, p_positivo, alerta,"
        " cluster_id, model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), app, text, label, probs["NEGATIVO"],
         probs["NEUTRO"], probs["POSITIVO"], int(alerta), cluster_id, model_version),
    )
    return int(cur.lastrowid)


def rows(con, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def to_json(o) -> str:
    return json.dumps(o, ensure_ascii=False)
