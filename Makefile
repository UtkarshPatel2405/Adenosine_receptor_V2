.PHONY: help install install-dev lint test test-cov test-fast test-all docker-build clean clean-data

help: ## Show this help
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: install ## Install dev dependencies
	pip install -r requirements-dev.txt
	pre-commit install

lint: ## Run ruff linter
	ruff check src/ tests/

typecheck: ## Run mypy type checker
	mypy src/ --ignore-missing-imports || true

test: ## Run tests (fast unit tests only)
	python -m pytest tests/ -v --tb=short -m "not slow and not integration" --ignore=tests/test_integration

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --tb=short -m "not slow and not integration" \
		--cov=src --cov-report=term-missing --cov-report=html --ignore=tests/test_integration

test-all: ## Run ALL tests (including slow/integration)
	python -m pytest tests/ -v --tb=short --cov=src

test-fast: ## Run fast tests in parallel
	python -m pytest tests/ -v --tb=short -m "not slow and not integration" -n auto --ignore=tests/test_integration

docker-build: ## Build Docker image
	docker build -t adenosine-selectivity-model .

docker-run: ## Run Streamlit app in Docker
	docker run -p 8501:8501 adenosine-selectivity-model

pipeline: ## Run full training pipeline
	python -m src.run_pipeline

pipeline-quick: ## Run pipeline (skip GNN, use default HPO)
	python -m src.run_pipeline --skip-gnn

data: ## Run data loading + cleaning only
	python -c "from src.data_loader import load_and_clean; df, lookup = load_and_clean(include_decoys=True); print('Data ready:', len(df), 'rows')"

dvc-init: ## Initialize DVC tracking
	dvc init
	dvc add data/raw
	dvc add data/processed
	dvc add models
	dvc add outputs

clean: ## Clean build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-data: ## Clean processed data (forces full reprocess)
	rm -rf data/processed/* models/* outputs/*
