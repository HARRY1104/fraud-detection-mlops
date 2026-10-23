.PHONY: setup data train drift run run-streamlit run-api test lint

setup:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset

train:
	python -m src.models.train

drift:
	python -m src.monitoring.drift_check

run: run-streamlit

run-streamlit:
	streamlit run app.py

run-api:
	uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
