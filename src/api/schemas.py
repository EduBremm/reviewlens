"""Contratos (Pydantic) da API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["The app crashes every time I open it"])
    app: str | None = Field(None, max_length=100, examples=["todoist"])


class BatchIn(BaseModel):
    reviews: list[ReviewIn] = Field(..., min_length=1, max_length=500)


class Probabilities(BaseModel):
    NEGATIVO: float
    NEUTRO: float
    POSITIVO: float


class PredictionOut(BaseModel):
    id: int
    sentimento: str
    probabilidades: Probabilities
    confianca: float
    alerta_negativo: bool
    grupo_reclamacao: str | None = None
    termos_influentes: list[str]
    model_version: str


class BatchOut(BaseModel):
    n: int
    distribuicao: dict[str, int]
    resultados: list[PredictionOut]
