"""ReviewLens — App Review Intelligence (dashboard Streamlit).

Rodar: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import insights  # noqa: E402
from src.config import DB_PATH  # noqa: E402
from src.db import connect, rows  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLORS = {"NEGATIVO": "#eb6834", "NEUTRO": "#8a8984", "POSITIVO": "#2a78d6"}

st.set_page_config(page_title="ReviewLens", page_icon="📊", layout="wide")
st.title("ReviewLens · App Review Intelligence")
st.caption("Sentimento de reviews da Google Play classificado por modelo de ML (tema 9 — Mineração de Dados)")


@st.cache_data(ttl=60)
def q(fn: str, *args):
    with connect(DB_PATH) as con:
        return getattr(insights, fn)(con, *args)


apps_df = pd.DataFrame(q("apps"))
col_f1, col_f2 = st.columns([2, 1])
app_sel = col_f1.selectbox("Aplicativo", ["(todos)"] + apps_df["app"].tolist())
app = None if app_sel == "(todos)" else app_sel
since = col_f2.selectbox("Período a partir de", ["2019-01", "2020-01", "2020-06", "2018-01"], index=0)

s = q("summary", app)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews", f"{s['n']:,}".replace(",", "."))
k2.metric("Negativos", f"{s['neg']:.1%}")
k3.metric("Neutros", f"{s['neu']:.1%}")
k4.metric("Positivos", f"{s['pos']:.1%}")
k5.metric("Nota média (estrelas)", f"{s['nota_media']:.2f}")

tabs = st.tabs(["Sentimento", "Evolução temporal", "Versões", "Principais reclamações", "Termos",
                "Classificar review"])

with tabs[0]:
    c1, c2 = st.columns([1, 2])
    mix = pd.DataFrame({"sentimento": list(COLORS), "pct": [s["neg"], s["neu"], s["pos"]]})
    c1.plotly_chart(px.pie(mix, names="sentimento", values="pct", hole=0.55, color="sentimento",
                           color_discrete_map=COLORS, title="Distribuição de sentimento"), use_container_width=True)
    a = apps_df.sort_values("pct_negativo")
    fig = px.bar(a, x="pct_negativo", y="app", orientation="h", title="% de reviews negativos por app",
                 labels={"pct_negativo": "% negativo", "app": ""}, color_discrete_sequence=[COLORS["NEGATIVO"]],
                 hover_data=["n", "nota_media"])
    fig.update_xaxes(tickformat=".0%")
    c2.plotly_chart(fig, use_container_width=True)
    st.caption(f"Concordância modelo × estrelas: {s['concordancia_estrelas']:.1%} — "
               "divergências indicam reviews com nota alta e texto de reclamação (ou o contrário).")

with tabs[1]:
    t = pd.DataFrame(q("timeline", app, since))
    if t.empty:
        st.info("Sem dados no período.")
    else:
        long = t.melt(id_vars=["month", "n"], value_vars=["neg", "neu", "pos"], var_name="s", value_name="pct")
        long["sentimento"] = long["s"].map({"neg": "NEGATIVO", "neu": "NEUTRO", "pos": "POSITIVO"})
        fig = px.area(long, x="month", y="pct", color="sentimento", color_discrete_map=COLORS,
                      title="Evolução mensal do sentimento (participação)", labels={"month": "mês", "pct": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.bar(t, x="month", y="n", title="Volume de reviews por mês",
                               labels={"month": "mês", "n": "reviews"},
                               color_discrete_sequence=["#3987e5"]), use_container_width=True)

with tabs[2]:
    minr = st.slider("Mínimo de reviews por versão", 5, 100, 20)
    v = pd.DataFrame(q("versions", app, minr, 20))
    if v.empty:
        st.info("Nenhuma versão com reviews suficientes.")
    else:
        v["rotulo"] = v["app"] + " " + v["versao"].astype(str)
        fig = px.bar(v.sort_values("pct_negativo"), x="pct_negativo", y="rotulo", orientation="h",
                     title="Versões com maior insatisfação (% negativo)", hover_data=["n", "nota_media", "primeira_review"],
                     labels={"pct_negativo": "% negativo", "rotulo": ""}, color_discrete_sequence=[COLORS["NEGATIVO"]])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(v.drop(columns="rotulo"), use_container_width=True, hide_index=True)

with tabs[3]:
    c = pd.DataFrame(q("complaints", app))
    if c.empty:
        st.info("Sem reclamações agrupadas.")
    else:
        fig = px.bar(c.sort_values("n"), x="n", y="nome", orientation="h", title="Grupos de reclamação (K-Means)",
                     labels={"n": "reviews negativos", "nome": ""}, color_discrete_sequence=["#2a78d6"],
                     hover_data={"pct": ":.1%"})
        st.plotly_chart(fig, use_container_width=True)
        grupo = st.selectbox("Ver exemplos do grupo", c["nome"].tolist())
        cid = int(c.loc[c["nome"] == grupo, "cluster_id"].iloc[0])
        st.write("**Termos característicos:** " + ", ".join(c.loc[c["nome"] == grupo, "top_termos"].iloc[0]))
        with connect(DB_PATH) as con:
            ex = rows(con, "SELECT app, score, version, substr(at,1,10) data, content FROM reviews "
                      "WHERE cluster_id=? " + ("AND app=? " if app else "") + "ORDER BY thumbs_up DESC LIMIT 8",
                      (cid, app) if app else (cid,))
        st.dataframe(pd.DataFrame(ex), use_container_width=True, hide_index=True)

with tabs[4]:
    lab = st.radio("Classe", list(COLORS), horizontal=True)
    tm = pd.DataFrame(q("terms", app, lab, 25))
    fig = px.bar(tm.sort_values("freq"), x="freq", y="termo", orientation="h", height=650,
                 title=f"Termos mais frequentes — {lab}", labels={"freq": "ocorrências", "termo": ""},
                 color_discrete_sequence=[COLORS[lab]])
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    txt = st.text_area("Texto do review (inglês)", "The app keeps crashing after the last update and I lost my tasks")
    app_in = st.text_input("App (opcional)", app or "")
    if st.button("Classificar", type="primary"):
        payload = {"text": txt, "app": app_in or None}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10).json()
        except Exception:
            from src.api.main import service  # fallback sem API rodando

            r = service().predict([txt])[0]
        c1, c2 = st.columns([1, 2])
        c1.metric("Sentimento", r["sentimento"], f"confiança {r['confianca']:.0%}")
        if r["alerta_negativo"]:
            c1.error("ALERTA: provável reclamação")
        if r.get("grupo_reclamacao"):
            c1.write(f"**Grupo:** {r['grupo_reclamacao']}")
        c1.write("**Termos influentes:** " + (", ".join(r["termos_influentes"]) or "—"))
        pr = r["probabilidades"]
        c2.plotly_chart(go.Figure(go.Bar(x=list(pr.values()), y=list(pr.keys()), orientation="h",
                                         marker_color=[COLORS[k] for k in pr])).update_layout(
            title="Probabilidades", xaxis_tickformat=".0%", height=260), use_container_width=True)
    with connect(DB_PATH) as con:
        hist = rows(con, "SELECT id, created_at, app, label, ROUND(p_negativo,3) p_neg, alerta, model_version, "
                         "substr(text,1,90) texto FROM predictions ORDER BY id DESC LIMIT 20")
    st.subheader("Histórico de predições")
    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("Modelo em produção")
    try:
        card = json.load(open(Path(__file__).resolve().parents[1] / "models" / "model_card.json", encoding="utf-8"))
        st.write(f"**{card['descricao']}**")
        st.write(f"Versão: `{card['versao']}`")
        st.write(f"F1-macro (CV): {card['cv']['f1_macro']:.3f}")
        st.write(f"ROC-AUC (CV): {card['cv']['roc_auc_ovr']:.3f}")
        st.write(f"Limiar de alerta NEG: {card['limiar_alerta_negativo']}")
    except FileNotFoundError:
        st.warning("Modelo não treinado. Rode `dvc repro`.")
