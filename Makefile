.PHONY: install pipeline test lint api dashboard mlflow docker clean
install:
	pip install -r requirements.txt
pipeline:            ## refaz só o que mudou (DVC)
	dvc repro
pipeline-nodvc:      ## mesma sequência, sem DVC
	python -m src.data.download && python -m src.data.validate && python -m src.features.build && \
	python -m src.analysis.eda && python -m src.models.train && python -m src.models.evaluate && \
	python -m src.analysis.clustering && python -m src.analysis.load_db
test:
	pytest -q --cov=src --cov-report=term-missing
lint:
	ruff check src tests dashboard
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000
dashboard:
	streamlit run dashboard/app.py --server.port 8501
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
docker:
	docker compose up --build
