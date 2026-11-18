"""Consultas de indicadores usadas pela API e pelo dashboard (uma única fonte da verdade)."""
from __future__ import annotations

import json
from collections import Counter

from src.db import rows
from src.features.text import clean_text

_STOP = set("""a about after again all also am an and any app apps are as at be because been but by can cant
could did didnt do does doesnt dont even every for from get got had has have i if im in into is it its ive
just like me more my no not now of on one only or other out please really so some still than that the
their them then there they this time to too up us use using very was way we were what when which while
will with would you your""".split())


def _where(app: str | None, col: str = "label_modelo"):
    return ("WHERE app = ?", (app,)) if app else ("", ())


def summary(con, app: str | None = None) -> dict:
    w, p = _where(app)
    r = rows(con, f"""SELECT COUNT(*) n,
        AVG(label_modelo='NEGATIVO') neg, AVG(label_modelo='NEUTRO') neu, AVG(label_modelo='POSITIVO') pos,
        AVG(score) nota_media, AVG(label_modelo = label_estrelas) concordancia_estrelas,
        MIN(at) inicio, MAX(at) fim FROM reviews {w}""", p)[0]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def timeline(con, app: str | None = None, since: str = "2019-01") -> list[dict]:
    w, p = _where(app)
    w = (w + " AND " if w else "WHERE ") + "month >= ?"
    return rows(con, f"""SELECT month, COUNT(*) n,
        ROUND(AVG(label_modelo='NEGATIVO'),4) neg, ROUND(AVG(label_modelo='NEUTRO'),4) neu,
        ROUND(AVG(label_modelo='POSITIVO'),4) pos, ROUND(AVG(score),3) nota_media
        FROM reviews {w} GROUP BY month ORDER BY month""", p + (since,))


def versions(con, app: str | None = None, min_reviews: int = 20, limit: int = 15) -> list[dict]:
    w, p = _where(app)
    w = (w + " AND " if w else "WHERE ") + "version_major IS NOT NULL"
    return rows(con, f"""SELECT app, version_major versao, COUNT(*) n,
        ROUND(AVG(label_modelo='NEGATIVO'),4) pct_negativo, ROUND(AVG(score),3) nota_media,
        MIN(substr(at,1,10)) primeira_review
        FROM reviews {w} GROUP BY app, version_major HAVING COUNT(*) >= ?
        ORDER BY pct_negativo DESC, n DESC LIMIT ?""", p + (min_reviews, limit))


def complaints(con, app: str | None = None) -> list[dict]:
    w, p = _where(app)
    w = (w + " AND " if w else "WHERE ") + "r.cluster_id IS NOT NULL"
    w = w.replace("app =", "r.app =")
    out = rows(con, f"""SELECT c.cluster_id, c.nome, c.top_termos, COUNT(r.review_id) n
        FROM reviews r JOIN complaint_clusters c ON c.cluster_id = r.cluster_id {w}
        GROUP BY c.cluster_id ORDER BY n DESC""", p)
    total = sum(o["n"] for o in out) or 1
    for o in out:
        o["top_termos"] = json.loads(o["top_termos"])
        o["pct"] = round(o["n"] / total, 4)
    return out


def terms(con, app: str | None = None, label: str = "NEGATIVO", k: int = 25) -> list[dict]:
    w, p = _where(app)
    w = (w + " AND " if w else "WHERE ") + "label_modelo = ?"
    cnt: Counter = Counter()
    for (txt,) in con.execute(f"SELECT content FROM reviews {w}", p + (label,)):
        toks = [t for t in clean_text(txt).split() if t not in _STOP and len(t) > 2]
        cnt.update(toks)
        cnt.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return [{"termo": t, "freq": f} for t, f in cnt.most_common(k)]


def apps(con) -> list[dict]:
    return rows(con, """SELECT app, COUNT(*) n, ROUND(AVG(label_modelo='NEGATIVO'),4) pct_negativo,
        ROUND(AVG(score),3) nota_media FROM reviews GROUP BY app ORDER BY pct_negativo DESC""")
