"""Etapa 2 — Validação: esquema + relatório de qualidade (reports/quality.json).

Falha (exit 1) se o esquema estiver quebrado. Problemas de qualidade "esperados"
(duplicados, textos vazios) são apenas contados aqui e tratados em src.features.build.

Uso: python -m src.data.validate
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, ROOT, ensure_dirs, params

REQUIRED = {
    "reviewId": "object", "content": "object", "score": "int64",
    "reviewCreatedVersion": "object", "at": "object", "appId": "object",
}
PII_COLUMNS = ["userName", "userImage"]


def check_schema(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in REQUIRED:
        if col not in df.columns:
            errors.append(f"coluna ausente: {col}")
    if "score" in df.columns and not df["score"].dropna().between(1, 5).all():
        errors.append("score fora do intervalo 1..5")
    if "at" in df.columns and pd.to_datetime(df["at"], errors="coerce").isna().any():
        errors.append("datas inválidas em 'at'")
    return errors


def quality_report(df: pd.DataFrame) -> dict:
    content = df["content"].fillna("")
    length = content.str.len()
    q1, q3 = length.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    at = pd.to_datetime(df["at"])
    dup_text_conflict = (
        df.assign(c=content.str.strip().str.lower())
        .groupby("c")["score"].nunique().gt(1).sum()
    )
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "apps": int(df["appId"].nunique()),
        "periodo": [str(at.min().date()), str(at.max().date())],
        "ausentes_pct": {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns},
        "reviewId_duplicado": int(df["reviewId"].duplicated().sum()),
        "texto_duplicado": int(content.str.strip().str.lower().duplicated().sum()),
        "textos_duplicados_com_notas_diferentes": int(dup_text_conflict),
        "texto_vazio_ou_curto_lt3": int((content.str.strip().str.len() < 3).sum()),
        "comprimento_texto": {k: round(float(v), 1) for k, v in length.describe().items()},
        "outliers_comprimento_iqr": {"limite_superior": float(upper), "qtd": int((length > upper).sum()),
                                      "pct": round(float((length > upper).mean() * 100), 2)},
        "distribuicao_score": {int(k): int(v) for k, v in df["score"].value_counts().sort_index().items()},
        "colunas_pii": [c for c in PII_COLUMNS if c in df.columns],
        "colunas_com_vazamento": ["replyContent", "repliedAt"],
        "reviews_por_ano": {int(k): int(v) for k, v in at.dt.year.value_counts().sort_index().items()},
        "versao_ausente_pct": round(float(df["reviewCreatedVersion"].isna().mean() * 100), 2),
        "nao_ascii_pct": round(float(content.map(lambda s: any(ord(ch) > 127 for ch in s)).mean() * 100), 2),
        "media_palavras": round(float(content.str.split().map(len).mean()), 1),
        "media_palavras_por_score": {int(k): round(float(v), 1) for k, v in
                                     content.str.split().map(len).groupby(df["score"]).mean().items()},
        "sha256": params()["data"]["sha256"],
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(ROOT / params()["data"]["raw_path"])
    errors = check_schema(df)
    if errors:
        sys.exit("[validate] esquema inválido: " + "; ".join(errors))
    rep = quality_report(df)
    with open(REPORTS_DIR / "quality.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(f"[validate] ok — {rep['linhas']} linhas, {rep['texto_duplicado']} textos duplicados, "
          f"{rep['texto_vazio_ou_curto_lt3']} vazios/curtos")


if __name__ == "__main__":
    main()
