"""Etapa 4 — Treino: busca de hiperparâmetros + validação cruzada estratificada de todos os candidatos.

- GridSearchCV com StratifiedKFold(5, shuffle, seed) só no conjunto de TREINO.
- Métrica de seleção: F1-macro (as três classes importam, inclusive NEUTRO).
- Cada candidato vira um run no MLflow; o vencedor é salvo em models/model.joblib
  e registrado no MLflow Model Registry.
- Ablação: efeito da lista de stopwords (com/sem negações) na Regressão Logística.

Uso: python -m src.models.train
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from src.config import DATA_PROCESSED, LABELS, MODEL_PATH, MODELS_DIR, REPORTS_DIR, ROOT, ensure_dirs, params
from src.features.text import STOPWORDS
from src.models.candidates import build_candidates

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc_ovr": "roc_auc_ovr",
    "recall_neg": make_scorer(recall_score, labels=["NEGATIVO"], average="macro", zero_division=0),
    "recall_neu": make_scorer(recall_score, labels=["NEUTRO"], average="macro", zero_division=0),
    "recall_pos": make_scorer(recall_score, labels=["POSITIVO"], average="macro", zero_division=0),
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sem-git"


def _mlflow():
    try:
        import mlflow

        cfg = params()["mlflow"]
        uri = os.getenv("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
        if "://" not in uri:
            uri = f"sqlite:///{ROOT / uri / 'mlflow.db'}"
            (ROOT / cfg["tracking_uri"]).mkdir(exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg["experiment"])
        return mlflow
    except Exception as exc:  # MLflow é opcional para rodar o pipeline
        print(f"[train] MLflow indisponível ({exc}); seguindo sem rastreamento")
        return None


def main() -> None:
    ensure_dirs()
    p = params()
    seed, tcfg = p["seed"], p["train"]
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    X, y = train["content"].to_numpy(), train["label"].to_numpy()
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    mlflow = _mlflow()
    sha = git_sha()

    results, fitted = {}, {}
    for name in tcfg["models"]:
        desc, pipe, grid = candidates[name]
        t0 = time.time()
        gs = GridSearchCV(pipe, grid or {}, scoring=SCORING, refit="f1_macro", cv=cv,
                          n_jobs=tcfg["n_jobs"], error_score="raise")
        gs.fit(X, y)
        elapsed = time.time() - t0
        i = gs.best_index_
        cvr = gs.cv_results_
        row = {"descricao": desc, "melhores_params": {k: str(v) for k, v in gs.best_params_.items()},
               "combinacoes_testadas": int(len(cvr["params"])), "tempo_busca_s": round(elapsed, 1),
               "tempo_fit_medio_s": round(float(cvr["mean_fit_time"][i]), 2)}
        for m in SCORING:
            row[m] = round(float(cvr[f"mean_test_{m}"][i]), 4)
            row[f"{m}_std"] = round(float(cvr[f"std_test_{m}"][i]), 4)
        row["f1_macro_por_dobra"] = [round(float(cvr[f"split{k}_test_f1_macro"][i]), 4)
                                     for k in range(tcfg["cv_folds"])]
        results[name], fitted[name] = row, gs.best_estimator_
        print(f"[train] {name:20s} F1-macro={row['f1_macro']:.4f}±{row['f1_macro_std']:.4f} "
              f"AUC={row['roc_auc_ovr']:.4f} ({elapsed:.0f}s)")
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.set_tags({"git_sha": sha, "descricao": desc, "fase": "cv"})
                mlflow.log_params({"modelo": name, "seed": seed, "cv_folds": tcfg["cv_folds"],
                                   **row["melhores_params"]})
                mlflow.log_metrics({f"cv_{m}": row[m] for m in SCORING}
                                   | {f"cv_{m}_std": row[f"{m}_std"] for m in SCORING})

    best = max((n for n in results if n not in ("dummy",)), key=lambda n: results[n]["f1_macro"])
    model = fitted[best]
    joblib.dump(model, MODEL_PATH)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    card = {"modelo": best, "descricao": results[best]["descricao"], "versao": f"{version}-{sha}",
            "git_sha": sha, "seed": seed, "classes": LABELS, "n_treino": int(len(train)),
            "params": results[best]["melhores_params"],
            "cv": {m: results[best][m] for m in SCORING}, "treinado_em": datetime.now(timezone.utc).isoformat()}

    # Limiar de alerta para NEGATIVO escolhido com predições out-of-fold (só treino, sem tocar no teste)
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=tcfg["n_jobs"])
    neg_idx = list(model.classes_).index("NEGATIVO")
    is_neg = (y == "NEGATIVO")
    curve = []
    for thr in np.round(np.arange(0.10, 0.91, 0.05), 2):
        flag = oof[:, neg_idx] >= thr
        tp = int((flag & is_neg).sum())
        curve.append({"limiar": float(thr), "recall_neg": tp / is_neg.sum(),
                      "precisao_neg": tp / max(flag.sum(), 1), "alertas_pct": float(flag.mean())})
    ok = [c for c in curve if c["recall_neg"] >= 0.90]
    alert_thr = max(ok, key=lambda c: c["limiar"])["limiar"] if ok else 0.5
    card["limiar_alerta_negativo"] = alert_thr
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, ensure_ascii=False, indent=2)

    # Ablação de stopwords (Regressão Logística com os melhores hiperparâmetros)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    ablation = {}
    base_lr = fitted["logreg"]
    for label, sw in [("sem_remocao_de_stopwords", None),
                      ("stopwords_padrao_sklearn (remove 'not', 'no', 'never')", list(ENGLISH_STOP_WORDS)),
                      ("stopwords_customizada (mantém negações) — adotada", list(STOPWORDS))]:
        est = clone(base_lr).set_params(tfidf__stop_words=sw)
        r = cross_validate(est, X, y, cv=cv, scoring={"f1_macro": "f1_macro",
                                                         "recall_neg": SCORING["recall_neg"]},
                           n_jobs=tcfg["n_jobs"])
        ablation[label] = {"f1_macro": round(float(r["test_f1_macro"].mean()), 4),
                           "f1_macro_std": round(float(r["test_f1_macro"].std()), 4),
                           "recall_neg": round(float(r["test_recall_neg"].mean()), 4)}
    print(f"[train] ablação stopwords: {ablation}")

    with open(REPORTS_DIR / "cv_results.json", "w", encoding="utf-8") as fh:
        json.dump({"melhor": best, "resultados": results, "ablacao_stopwords": ablation,
                   "curva_limiar_neg_oof": curve}, fh, ensure_ascii=False, indent=2)

    if mlflow:
        with mlflow.start_run(run_name=f"{best}-final"):
            mlflow.set_tags({"git_sha": sha, "fase": "final"})
            mlflow.log_params({"modelo": best, **results[best]["melhores_params"],
                               "limiar_alerta_negativo": alert_thr})
            mlflow.log_metrics({f"cv_{m}": results[best][m] for m in SCORING})
            mlflow.log_artifact(str(MODELS_DIR / "model_card.json"))
            try:
                import mlflow.sklearn

                mlflow.sklearn.log_model(model, name="model", registered_model_name="reviewlens-sentimento",
                                         input_example=np.array(["great app"]))
            except Exception as exc:
                print(f"[train] registro no Model Registry falhou: {exc}")
    print(f"[train] melhor modelo: {best} -> {MODEL_PATH} (limiar alerta NEG={alert_thr})")


if __name__ == "__main__":
    main()
