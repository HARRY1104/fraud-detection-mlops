.PHONY: setup data train drift run-api test lint

setup:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset

train:
	python -m src.models.train

drift:
	python -m src.monitoring.drift_check

run-api:
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
