"""Configuração central: caminhos do projeto e leitura de params.yaml."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DB_PATH = Path(os.getenv("REVIEWLENS_DB", ROOT / "data" / "reviewlens.db"))
MODEL_PATH = Path(os.getenv("REVIEWLENS_MODEL", MODELS_DIR / "model.joblib"))

LABELS = ["NEGATIVO", "NEUTRO", "POSITIVO"]


@lru_cache(maxsize=1)
def params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
