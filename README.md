# ReviewLens — App Review Intelligence

Projeto de Mineração de Dados (CRISP-DM + MLOps) — **Tema 9: Mineração de Sentimentos em Reviews de Aplicativos**
Aluno: Eduardo Luis Bremm · Engenharia de Software — Centro Universitário FAG

O ReviewLens classifica reviews da Google Play em **POSITIVO / NEUTRO / NEGATIVO**, agrupa as reclamações por tema
(K-Means) e mostra tudo em um dashboard: sentimento, evolução temporal, principais reclamações, termos mais
frequentes e versões com maior insatisfação.

| Item | Valor |
|---|---|
| Dados | 12.495 reviews de 15 apps de produtividade (Google Play, 2015–2020), coletados com `google-play-scraper` |
| Modelos comparados | Dummy, VADER (pré-treinado), Naive Bayes, SVM, Regressão Logística, vetores spaCy (pré-treinados) + LR, híbrido TF-IDF+VADER |
| Modelo final | Regressão Logística + TF-IDF (1–2 gramas) |
| Teste (20%, seed 42) | **F1-macro 0,622 · ROC-AUC 0,821 · acurácia 0,681** (VADER: F1 0,472) |
| Produto | API FastAPI + SQLite + dashboard Streamlit; MLflow + DVC; Docker; GitHub Actions |

## Como reproduzir

```bash
git clone https://github.com/EduBremm/reviewlens.git && cd reviewlens

# opção 1 — Docker (sobe API :8000, dashboard :8501 e MLflow :5000)
docker compose up --build
docker compose run --rm app dvc repro        # refaz dados -> treino -> avaliação

# opção 2 — local (Python 3.13)
pip install -r requirements.txt
dvc repro                                    # ou: make pipeline-nodvc
make test                                    # 33 testes, cobertura ~87% dos módulos de biblioteca
make api        # http://localhost:8000/docs
make dashboard  # http://localhost:8501
make mlflow     # http://localhost:5000
```

Nenhum passo manual: o CSV é baixado de uma URL fixada em commit e o SHA-256 é conferido (`params.yaml`).

## Pipeline (`dvc.yaml`)

| Etapa | Comando | Saída |
|---|---|---|
| download | `python -m src.data.download` | `data/raw/reviews.csv` (hash conferido) |
| validate | `python -m src.data.validate` | `reports/quality.json` |
| build_features | `python -m src.features.build` | `data/processed/{reviews_clean,train,test}.parquet` |
| eda | `python -m src.analysis.eda` | `reports/eda.json`, `reports/figures/eda_*.png` |
| train | `python -m src.models.train` | `models/model.joblib`, `models/model_card.json`, `reports/cv_results.json`, runs no MLflow |
| evaluate | `python -m src.models.evaluate` | `reports/metrics.json`, `reports/figures/eval_*.png` |
| clustering | `python -m src.analysis.clustering` | `models/clusters.joblib`, `reports/clusters.json` |
| load_db | `python -m src.analysis.load_db` | `data/reviewlens.db` |

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | status e modelo carregado |
| GET | `/model/info` | model card (versão, params, métricas de CV, limiar de alerta) |
| POST | `/predict` | classifica um review; grava em `predictions` |
| POST | `/predict/batch` | até 500 reviews por chamada |
| GET | `/predictions` | histórico (auditoria, com versão do modelo) |
| GET | `/apps`, `/insights/summary`, `/insights/timeline`, `/insights/versions`, `/insights/complaints`, `/insights/terms` | indicadores do dashboard (`?app=Todoist` filtra) |

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text":"After the last update the app crashes and my tasks do not sync anymore","app":"Todoist"}'
```

## Estrutura

```
src/data        download.py, validate.py
src/features    text.py, transformers.py, build.py
src/models      candidates.py, train.py, evaluate.py
src/analysis    eda.py, clustering.py, insights.py, load_db.py
src/api         main.py, schemas.py
dashboard/      app.py (Streamlit)
tests/          test_data.py, test_features.py, test_model.py, test_api.py, test_pipeline_parts.py
```

Uso exclusivamente acadêmico. Nomes e fotos de usuários são descartados na ingestão.
