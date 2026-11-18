"""Análise exploratória: figuras em reports/figures/eda_*.png e achados em reports/eda.json.

Uso: python -m src.analysis.eda
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import DATA_PROCESSED, FIGURES_DIR, LABELS, REPORTS_DIR, ensure_dirs
from src.viz import INK2, SENT_COLORS, SEQ_BLUE, plt, save, setup


def main() -> None:
    ensure_dirs()
    setup()
    df = pd.read_parquet(DATA_PROCESSED / "reviews_clean.parquet")
    out = {}

    # 1. distribuição do alvo e das estrelas
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sc = df["score"].value_counts().sort_index()
    cols = [SENT_COLORS["NEGATIVO"]] * 2 + [SENT_COLORS["NEUTRO"]] + [SENT_COLORS["POSITIVO"]] * 2
    axes[0].bar(sc.index.astype(str), sc.values, color=cols, width=0.7)
    axes[0].set_title("Reviews por nota (estrelas)")
    lab = df["label"].value_counts().reindex(LABELS)
    axes[1].bar(lab.index, lab.values, color=[SENT_COLORS[l] for l in LABELS], width=0.6)
    for i, v in enumerate(lab.values):
        axes[1].text(i, v, f"{v} ({v/len(df):.0%})", ha="center", va="bottom", fontsize=9, color=INK2)
    axes[1].set_title("Classe-alvo (sentimento)")
    save(fig, FIGURES_DIR / "eda_alvo.png")
    out["dist_label"] = (lab / len(df)).round(4).to_dict()

    # 2. comprimento do texto por classe
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.arange(0, 130, 5)
    for l in LABELS:
        ax.hist(df.loc[df.label == l, "n_words"].clip(upper=125), bins=bins, histtype="step",
                linewidth=2, color=SENT_COLORS[l], label=l, density=True)
    ax.set_xlabel("palavras por review (cortado em 125)")
    ax.set_ylabel("densidade")
    ax.set_title("Comprimento do review por sentimento")
    ax.legend()
    save(fig, FIGURES_DIR / "eda_comprimento.png")
    out["mediana_palavras_por_classe"] = df.groupby("label")["n_words"].median().to_dict()

    # 3. volume mensal e % negativo
    m = df[df["at"] >= "2019-01-01"].groupby("month").agg(n=("label", "size"),
                                                         neg=("label", lambda s: (s == "NEGATIVO").mean()))
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    axes[0].bar(m.index, m["n"], color=SEQ_BLUE[3], width=0.8)
    axes[0].set_title("Volume de reviews por mês (2019–2020)")
    axes[1].plot(m.index, m["neg"] * 100, color=SENT_COLORS["NEGATIVO"], marker="o", markersize=4)
    axes[1].set_title("% de reviews negativos por mês")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    save(fig, FIGURES_DIR / "eda_temporal.png")

    # 4. sentimento por app (barras 100% empilhadas)
    t = pd.crosstab(df["app_name"], df["label"], normalize="index")[LABELS].sort_values("NEGATIVO")
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(t))
    for l in LABELS:
        ax.barh(t.index, t[l] * 100, left=left, color=SENT_COLORS[l], label=l, height=0.75,
                edgecolor="#fcfcfb", linewidth=1)
        left += t[l].values * 100
    ax.set_xlabel("% dos reviews do app")
    ax.set_title("Mix de sentimento por aplicativo")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES_DIR / "eda_apps.png")
    out["neg_por_app"] = t["NEGATIVO"].round(3).to_dict()

    # 5. correlação entre atributos numéricos
    num = df.assign(chars=df["content"].str.len(), tem_versao=df["reviewCreatedVersion"].notna().astype(int),
                    exclamacoes=df["content"].str.count("!"), maiusculas=df["content"].str.count(r"[A-Z]"))
    cols_num = ["score", "n_words", "chars", "thumbsUpCount", "exclamacoes", "maiusculas", "tem_versao"]
    corr = num[cols_num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols_num)), cols_num, rotation=45, ha="right")
    ax.set_yticks(range(len(cols_num)), cols_num)
    ax.grid(False)
    for i in range(len(cols_num)):
        for j in range(len(cols_num)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlação de Spearman (atributos numéricos)")
    save(fig, FIGURES_DIR / "eda_correlacao.png")
    out["spearman_score"] = corr["score"].drop("score").round(3).to_dict()

    # 6. termos mais frequentes por classe (unigramas e bigramas, sem stopwords só para leitura)
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
    X = cv.fit_transform(df["text_clean"])
    vocab = np.array(cv.get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    out["top_termos"] = {}
    for ax, l in zip(axes, LABELS):
        freq = np.asarray(X[(df["label"] == l).to_numpy()].sum(axis=0)).ravel()
        idx = freq.argsort()[::-1][:12]
        ax.barh(vocab[idx][::-1], freq[idx][::-1], color=SENT_COLORS[l], height=0.7)
        ax.set_title(l)
        ax.grid(axis="y", visible=False)
        out["top_termos"][l] = vocab[idx].tolist()
    fig.suptitle("Termos mais frequentes por classe", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, FIGURES_DIR / "eda_termos.png")

    with open(REPORTS_DIR / "eda.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print("[eda] figuras geradas em reports/figures/eda_*.png")


if __name__ == "__main__":
    main()
