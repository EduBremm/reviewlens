"""Etapa 5 — Avaliação: o conjunto de TESTE é usado uma única vez, aqui, com o modelo final.

Saídas: reports/metrics.json, reports/errors_sample.csv e figuras eval_*.png

Uso: python -m src.models.evaluate
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import DATA_PROCESSED, FIGURES_DIR, LABELS, MODEL_PATH, MODELS_DIR, REPORTS_DIR, ensure_dirs, params
from src.features.transformers import VaderClassifier
from src.viz import INK2, SENT_COLORS, plt, save, setup

# Metas definidas ANTES de olhar o teste (seção 1.2 do relatório)
TARGETS = {"f1_macro": 0.60, "roc_auc_ovr": 0.80, "ganho_f1_vs_vader_pp": 10.0,
           "recall_neg_alerta": 0.90, "precisao_neg_alerta": 0.60}


def core_metrics(y, pred, proba, classes) -> dict:
    yb = np.vstack([(y == c) for c in classes]).T.astype(int)
    rep = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(yb, proba, average="macro"),
        "pr_auc_macro": average_precision_score(yb, proba, average="macro"),
        "log_loss": log_loss(y, proba, labels=list(classes)),
        "brier_medio": float(np.mean([brier_score_loss(yb[:, i], proba[:, i]) for i in range(len(classes))])),
        "por_classe": {l: {k: rep[l][k] for k in ("precision", "recall", "f1-score", "support")} for l in LABELS},
        "pr_auc_por_classe": {c: average_precision_score(yb[:, i], proba[:, i]) for i, c in enumerate(classes)},
    }


def r4(o):
    if isinstance(o, dict):
        return {k: r4(v) for k, v in o.items()}
    if isinstance(o, (float, np.floating)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main() -> None:
    ensure_dirs()
    setup()
    p = params()
    model = joblib.load(MODEL_PATH)
    card = json.load(open(MODELS_DIR / "model_card.json", encoding="utf-8"))
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    X, y = test["content"].to_numpy(), test["label"].to_numpy()
    classes = list(model.classes_)

    proba = model.predict_proba(X)
    pred = model.predict(X)
    m = core_metrics(y, pred, proba, classes)

    # baseline de referência no mesmo teste (não houve seleção com ele)
    vader = VaderClassifier().fit(X)
    mv = core_metrics(y, vader.predict(X), vader.predict_proba(X), LABELS)

    # alerta de review negativo com o limiar escolhido no treino (out-of-fold)
    thr = card["limiar_alerta_negativo"]
    ni = classes.index("NEGATIVO")
    flag = proba[:, ni] >= thr
    is_neg = y == "NEGATIVO"
    alerta = {"limiar": thr, "recall_neg": (flag & is_neg).sum() / is_neg.sum(),
              "precisao_neg": (flag & is_neg).sum() / max(flag.sum(), 1), "alertas_pct": flag.mean(),
              "negativos_perdidos": int((~flag & is_neg).sum()), "negativos_total": int(is_neg.sum())}

    cm = confusion_matrix(y, pred, labels=LABELS)

    # ---------- figuras ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title, fmt in [(axes[0], cm, "Contagens", "d"),
                                (axes[1], cm / cm.sum(axis=1, keepdims=True), "Normalizada por linha (recall)", ".2f")]:
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.grid(False)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > mat.max() * 0.6 else "#0b0b0b")
    save(fig, FIGURES_DIR / "eval_confusao.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, c in enumerate(classes):
        yt = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(yt, proba[:, i])
        axes[0].plot(fpr, tpr, color=SENT_COLORS[c], label=f"{c} (AUC {roc_auc_score(yt, proba[:, i]):.3f})")
        pr, rc, _ = precision_recall_curve(yt, proba[:, i])
        axes[1].plot(rc, pr, color=SENT_COLORS[c], label=f"{c} (AP {average_precision_score(yt, proba[:, i]):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
    axes[0].set(xlabel="taxa de falso positivo", ylabel="taxa de verdadeiro positivo", title="Curva ROC (um-contra-todos)")
    axes[1].set(xlabel="recall", ylabel="precisão", title="Curva Precisão-Recall")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    save(fig, FIGURES_DIR / "eval_roc_pr.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for i, c in enumerate(classes):
        fp, mp = calibration_curve((y == c).astype(int), proba[:, i], n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", markersize=4, color=SENT_COLORS[c], label=c)
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1, label="perfeitamente calibrado")
    ax.set(xlabel="probabilidade prevista", ylabel="frequência observada", title="Curva de calibração (teste)")
    ax.legend()
    save(fig, FIGURES_DIR / "eval_calibracao.png")

    # importância global: coeficientes da regressão logística (modelo linear interpretável)
    interp = {}
    clf = model.named_steps.get("clf")
    vec = model.named_steps.get("tfidf")
    if vec is not None and hasattr(clf, "coef_"):
        names = vec.get_feature_names_out()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
        for ax, (i, c) in zip(axes, enumerate(classes)):
            coef = clf.coef_[i]
            top = np.argsort(coef)[::-1][:12]
            interp[c] = [(names[j], round(float(coef[j]), 3)) for j in top]
            ax.barh(names[top][::-1], coef[top][::-1], color=SENT_COLORS[c], height=0.7)
            ax.set_title(f"Termos que mais empurram p/ {c}")
            ax.grid(axis="y", visible=False)
        fig.tight_layout()
        save(fig, FIGURES_DIR / "eval_importancia.png")

    # ---------- análise de erros ----------
    test = test.assign(pred=pred, p_neg=proba[:, ni], p_neu=proba[:, classes.index("NEUTRO")],
                       p_pos=proba[:, classes.index("POSITIVO")], acerto=(pred == y))
    test["faixa_tamanho"] = pd.cut(test["n_words"], [0, 5, 15, 40, 100, 10_000],
                                   labels=["1-5", "6-15", "16-40", "41-100", ">100"])
    by_len = test.groupby("faixa_tamanho", observed=True).agg(n=("acerto", "size"), acuracia=("acerto", "mean"))
    by_app = test.groupby("app_name").agg(n=("acerto", "size"), acuracia=("acerto", "mean"),
                                          f1=("acerto", lambda s: f1_score(test.loc[s.index, "label"],
                                                                           test.loc[s.index, "pred"],
                                                                           average="macro"))).sort_values("f1")
    by_score = test.groupby("score").agg(n=("acerto", "size"), acuracia=("acerto", "mean"))
    # erros "grosseiros": polaridade invertida
    flip = test[((test.label == "NEGATIVO") & (test.pred == "POSITIVO")) |
                ((test.label == "POSITIVO") & (test.pred == "NEGATIVO"))]
    errs = test[~test.acerto].copy()
    errs["confianca"] = errs[["p_neg", "p_neu", "p_pos"]].max(axis=1)
    errs.sort_values("confianca", ascending=False)[["app_name", "score", "label", "pred", "confianca", "content"]] \
        .head(40).to_csv(REPORTS_DIR / "errors_sample.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(by_len.index.astype(str), by_len["acuracia"] * 100, color="#2a78d6", width=0.6)
    axes[0].set(title="Acurácia por tamanho do review", xlabel="palavras", ylabel="%")
    axes[1].bar(by_score.index.astype(str), by_score["acuracia"] * 100,
                color=[SENT_COLORS["NEGATIVO"]] * 2 + [SENT_COLORS["NEUTRO"]] + [SENT_COLORS["POSITIVO"]] * 2, width=0.6)
    axes[1].set(title="Acurácia por nota original", xlabel="estrelas", ylabel="%")
    save(fig, FIGURES_DIR / "eval_erros.png")

    # ---------- robustez temporal (análise extra; treina no passado, testa no futuro) ----------
    full = pd.concat([train, test.drop(columns=[c for c in test.columns if c not in train.columns])])
    cut = pd.Timestamp("2020-08-01")
    past, fut = full[full["at"] < cut], full[full["at"] >= cut]
    tm = clone(model).fit(past["content"].to_numpy(), past["label"].to_numpy())
    tp = tm.predict(fut["content"].to_numpy())
    temporal = {"corte": str(cut.date()), "n_treino": len(past), "n_teste": len(fut),
                "f1_macro": f1_score(fut["label"], tp, average="macro"),
                "accuracy": accuracy_score(fut["label"], tp)}

    resumo = {
        "modelo": card["modelo"], "versao": card["versao"], "particao": "teste (20%, estratificado)",
        "seed": p["seed"], "n_teste": len(test), "metricas": m, "vader_mesmo_teste": mv,
        "ganho_f1_vs_vader_pp": (m["f1_macro"] - mv["f1_macro"]) * 100,
        "alerta_negativo": alerta, "matriz_confusao": {"labels": LABELS, "valores": cm.tolist()},
        "metas": TARGETS,
        "atingiu": {
            "f1_macro": m["f1_macro"] >= TARGETS["f1_macro"],
            "roc_auc_ovr": m["roc_auc_ovr"] >= TARGETS["roc_auc_ovr"],
            "ganho_f1_vs_vader_pp": (m["f1_macro"] - mv["f1_macro"]) * 100 >= TARGETS["ganho_f1_vs_vader_pp"],
            "recall_neg_alerta": bool(alerta["recall_neg"] >= TARGETS["recall_neg_alerta"]),
            "precisao_neg_alerta": bool(alerta["precisao_neg"] >= TARGETS["precisao_neg_alerta"]),
        },
        "erros": {"acuracia_por_tamanho": by_len.reset_index().astype({"faixa_tamanho": str}).to_dict("records"),
                  "acuracia_por_score": by_score.reset_index().to_dict("records"),
                  "por_app": by_app.reset_index().to_dict("records"),
                  "polaridade_invertida": len(flip), "polaridade_invertida_pct": len(flip) / len(test)},
        "interpretacao_top_termos": interp,
        "robustez_temporal": temporal,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(r4(resumo), fh, ensure_ascii=False, indent=2, default=str)
    print(f"[evaluate] teste: F1-macro={m['f1_macro']:.4f} AUC={m['roc_auc_ovr']:.4f} "
          f"acc={m['accuracy']:.4f} | VADER F1={mv['f1_macro']:.4f} | alerta recall={alerta['recall_neg']:.3f} "
          f"prec={alerta['precisao_neg']:.3f} | temporal F1={temporal['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
