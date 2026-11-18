"""Segunda análise — agrupamento das reclamações (reviews NEGATIVOS) com K-Means.

TF-IDF -> TruncatedSVD (LSA) -> normalização L2 -> K-Means; k escolhido por silhueta,
conferido com Davies-Bouldin. Saídas: models/clusters.joblib, reports/clusters.json, figuras.

Uso: python -m src.analysis.clustering
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs, params
from src.viz import SERIES, plt, save, setup

# palavras genéricas de review que não ajudam a separar temas de reclamação
EXTRA_STOP = {"app", "apps", "use", "using", "used", "just", "like", "dont", "im", "ive", "its", "doesnt",
              "cant", "didnt", "isnt", "wont", "really", "would", "want", "get", "got", "even", "one", "make",
              "please", "thing", "things", "also", "still", "know", "way", "good", "great", "nice", "love"}


def build_pipeline(cfg: dict, k: int, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP), min_df=5, max_df=0.5,
                                  max_features=cfg["max_features"], ngram_range=(1, 2), sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=cfg["svd_components"], random_state=seed)),
        ("norm", Normalizer(copy=False)),
        ("km", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def top_terms(pipe: Pipeline, n: int = 8) -> dict[int, list[str]]:
    terms = pipe.named_steps["tfidf"].get_feature_names_out()
    centers = pipe.named_steps["svd"].inverse_transform(pipe.named_steps["km"].cluster_centers_)
    return {i: [terms[j] for j in c.argsort()[::-1][:n]] for i, c in enumerate(centers)}


# Rótulo humano dado após ler os termos e 10 exemplos de cada grupo (interpretação, CRISP-DM "Evaluation")
# Nome dado após ler os termos e exemplos de cada grupo (seed fixa => grupos reprodutíveis).
NAME_BY_FIRST_TERM = {
    "time": "Configuração de horários / perda de tempo",
    "task": "Falhas gerais (widget, sync, notificações)",
    "calendar": "Integração com Google Calendar",
    "account": "Conta e login",
    "pay": "Recursos bloqueados na versão gratuita",
    "update": "Problemas após atualização",
    "tasks": "Tarefas recorrentes / concluídas",
    "premium": "Assinatura premium e suporte",
    "pro": "Versão Pro paga (Forest)",
    "ads": "Anúncios invasivos",
    "fix": "Pedidos de correção e usabilidade",
    "work": "Lembretes/notificações que não funcionam",
}


def auto_name(terms: list[str]) -> str:
    if terms and terms[0] in NAME_BY_FIRST_TERM:
        return NAME_BY_FIRST_TERM[terms[0]]
    rules = [
        ({"sync", "syncing", "calendar", "google calendar", "google", "outlook"}, "Sincronização / calendário"),
        ({"premium", "pay", "subscription", "money", "free", "paid", "refund", "price", "version"}, "Preço / assinatura premium"),
        ({"notification", "notifications", "reminder", "reminders", "alarm"}, "Notificações e lembretes"),
        ({"update", "updated", "new update", "latest update"}, "Problemas após atualização"),
        ({"crash", "crashes", "open", "work", "working", "doesnt work", "bug", "bugs", "fix"}, "Bugs / app não funciona"),
        ({"ads", "ad", "advertisement"}, "Anúncios"),
        ({"login", "account", "sign", "log", "password", "email"}, "Conta e login"),
        ({"tasks", "task", "list", "lists", "todo"}, "Gestão de tarefas / listas"),
        ({"widget", "widgets"}, "Widget"),
        ({"habit", "habits", "streak"}, "Hábitos / streaks"),
        ({"time", "timer", "tree", "trees", "forest"}, "Timer / foco"),
    ]
    head = terms[:4]
    for keys, name in rules:
        if any(t in keys for t in head):
            return name
    return " / ".join(terms[:3])


def main() -> None:
    ensure_dirs()
    setup()
    p = params()
    cfg, seed = p["clustering"], p["seed"]
    df = pd.read_parquet(DATA_PROCESSED / "reviews_clean.parquet")
    neg = df[(df["label"] == "NEGATIVO") & (df["n_words"] >= cfg["min_words"])].reset_index(drop=True)
    texts = neg["text_clean"].to_numpy()

    # representação fixa para comparar k (o K-Means é a única parte que muda)
    base = build_pipeline(cfg, 2, seed)
    Z = Pipeline(base.steps[:-1]).fit_transform(texts)
    scan = []
    for k in range(cfg["k_min"], cfg["k_max"] + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        scan.append({"k": k, "silhueta": float(silhouette_score(Z, lab, metric="cosine", sample_size=3000,
                                                                random_state=seed)),
                     "davies_bouldin": float(davies_bouldin_score(Z, lab))})
    best_k = max(scan, key=lambda s: s["silhueta"])["k"]

    pipe = build_pipeline(cfg, best_k, seed).fit(texts)
    neg["cluster"] = pipe.predict(texts)
    terms = top_terms(pipe)
    clusters = []
    for c in range(best_k):
        sub = neg[neg["cluster"] == c]
        clusters.append({
            "cluster_id": c, "nome": auto_name(terms[c]), "top_termos": terms[c], "n": int(len(sub)),
            "pct": round(len(sub) / len(neg) * 100, 1),
            "apps_principais": sub["app_name"].value_counts().head(3).to_dict(),
            "exemplos": sub.sort_values("thumbsUpCount", ascending=False)["content"].str.slice(0, 220).head(3).tolist(),
        })
    # nomes repetidos recebem sufixo com o termo mais forte
    seen: dict[str, int] = {}
    for cl in clusters:
        seen[cl["nome"]] = seen.get(cl["nome"], 0) + 1
    for cl in clusters:
        if seen[cl["nome"]] > 1:
            cl["nome"] = f"{cl['nome']} ({cl['top_termos'][0]})"

    joblib.dump({"pipeline": pipe, "nomes": {c["cluster_id"]: c["nome"] for c in clusters}},
                MODELS_DIR / "clusters.joblib")
    neg[["reviewId", "cluster"]].to_parquet(DATA_PROCESSED / "neg_clusters.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ks = [s["k"] for s in scan]
    axes[0].plot(ks, [s["silhueta"] for s in scan], marker="o", color=SERIES[0])
    axes[0].axvline(best_k, ls="--", color="#52514e", lw=1)
    axes[0].set(title="Silhueta (cosseno) por k — maior é melhor", xlabel="k")
    axes[1].plot(ks, [s["davies_bouldin"] for s in scan], marker="o", color=SERIES[1])
    axes[1].axvline(best_k, ls="--", color="#52514e", lw=1)
    axes[1].set(title="Davies-Bouldin por k — menor é melhor", xlabel="k")
    save(fig, FIGURES_DIR / "cluster_escolha_k.png")

    cl_sorted = sorted(clusters, key=lambda c: c["n"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * best_k + 1.2))
    ax.barh([c["nome"] for c in cl_sorted], [c["n"] for c in cl_sorted], color=SERIES[0], height=0.65)
    for i, c in enumerate(cl_sorted):
        ax.text(c["n"], i, f"  {c['n']} ({c['pct']}%)", va="center", fontsize=9, color="#52514e")
    ax.set_title("Grupos de reclamação (reviews negativos, K-Means)")
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES_DIR / "cluster_tamanhos.png")

    with open(REPORTS_DIR / "clusters.json", "w", encoding="utf-8") as fh:
        json.dump({"n_reviews_negativos": len(neg), "k_escolhido": best_k, "varredura_k": scan,
                   "variancia_explicada_svd": round(float(pipe.named_steps["svd"].explained_variance_ratio_.sum()), 4),
                   "clusters": clusters}, fh, ensure_ascii=False, indent=2)
    print(f"[clustering] k={best_k}: " + "; ".join(f"{c['nome']}={c['n']}" for c in clusters))


if __name__ == "__main__":
    main()
