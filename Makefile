.PHONY: setup audit train test lint format app api docker clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && \
	pip install -r requirements-dev.txt && pip install -e .

audit:
	python -m property_price.audit

train:
	python -m property_price.train

test:
	pytest --cov=property_price --cov-report=term-missing

lint:
	ruff check src tests app && black --check src tests app

format:
	black src tests app && ruff check --fix src tests app

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api.main:app --reload

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
