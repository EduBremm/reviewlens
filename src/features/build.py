"""Etapa 3 — Preparação: limpeza, deduplicação, rótulo, anonimização e divisão treino/teste.

Saídas:
  data/processed/reviews_clean.parquet  (todas as reviews tratadas — usadas pelo dashboard)
  data/processed/train.parquet / test.parquet  (80/20 estratificado, seed do params.yaml)

Uso: python -m src.features.build
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, REPORTS_DIR, ROOT, ensure_dirs, params
from src.features.text import app_display_name, clean_text, major_version, score_to_label

KEEP = ["reviewId", "appId", "content", "score", "thumbsUpCount", "reviewCreatedVersion", "at"]


def prepare(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    log = {"entrada": len(df)}
    df = df[KEEP].copy()  # remove PII (userName, userImage) e colunas pós-review (replyContent, repliedAt)
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df = df[df["content"].str.len() >= cfg["min_chars"]]
    log["apos_remover_curtos"] = len(df)
    df = df.drop_duplicates("reviewId")
    # mesmo texto + mesma nota = cópia; mesmo texto com notas diferentes = rótulo ambíguo -> remove todos
    df["_key"] = df["content"].str.lower()
    df = df.drop_duplicates(["_key", "score"])
    ambiguous = df.groupby("_key")["score"].transform("nunique") > 1
    log["removidos_rotulo_ambiguo"] = int(ambiguous.sum())
    df = df[~ambiguous].drop(columns="_key")
    log["apos_deduplicar"] = len(df)
    df["text_clean"] = df["content"].map(clean_text)
    df = df[df["text_clean"].str.len() >= cfg["min_chars"]]
    log["apos_limpeza"] = len(df)
    label_map = {int(k): v for k, v in cfg["label_map"].items()}
    df["label"] = df["score"].map(lambda s: score_to_label(s, label_map))
    df["at"] = pd.to_datetime(df["at"])
    df["month"] = df["at"].dt.to_period("M").astype(str)
    df["version_major"] = df["reviewCreatedVersion"].map(major_version)
    df["n_words"] = df["content"].str.split().map(len)
    df["app_name"] = df["appId"].map(app_display_name)
    log["saida"] = len(df)
    return df.reset_index(drop=True), log


def main() -> None:
    ensure_dirs()
    p = params()
    raw = pd.read_csv(ROOT / p["data"]["raw_path"])
    df, log = prepare(raw, p["features"])
    train, test = train_test_split(
        df, test_size=p["features"]["test_size"], stratify=df["label"], random_state=p["seed"]
    )
    df.to_parquet(DATA_PROCESSED / "reviews_clean.parquet", index=False)
    train.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "test.parquet", index=False)
    log.update({
        "treino": len(train), "teste": len(test),
        "dist_label_treino": train["label"].value_counts(normalize=True).round(4).to_dict(),
        "dist_label_teste": test["label"].value_counts(normalize=True).round(4).to_dict(),
    })
    with open(REPORTS_DIR / "preparation.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)
    print(f"[build] {log}")


if __name__ == "__main__":
    main()
