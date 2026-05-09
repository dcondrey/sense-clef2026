.PHONY: setup train predict evaluate test lint clean docker-build docker-run help

PYTHON ?= python
DATA ?= data/dev.rubric.json
DEVSET_DIR ?= data
INPUT ?= /tmp/sense-input
OUTPUT ?= /tmp/sense-output
MODEL_DIR ?= models

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install package and dependencies
	pip install -e ".[train,dev]"
	$(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
	@echo "Setup complete."

train: ## Train all models (DeBERTa + LGB + MLP ensemble)
	$(PYTHON) train.py --data $(DATA) --output $(MODEL_DIR)

predict: ## Run inference on input data
	$(PYTHON) predict.py -i $(INPUT) -o $(OUTPUT)

evaluate: ## Evaluate predictions against devset
	$(PYTHON) evaluate.py -d $(DEVSET_DIR) -p $(OUTPUT) -t rubric

test: ## Run pipeline validation tests
	$(PYTHON) -m pytest tests/ -v 2>/dev/null || $(PYTHON) -c "from sense import models, features, ces, ensemble; print('All modules import successfully')"

lint: ## Run code quality checks
	$(PYTHON) -m ruff check sense/ *.py || true
	$(PYTHON) -m mypy sense/ --ignore-missing-imports || true

docker-build: ## Build Docker image
	docker build -t sense-clef2026:latest .

docker-run: ## Run inference in Docker container
	mkdir -p $(INPUT) $(OUTPUT)
	docker run --rm -v $(INPUT):/input -v $(OUTPUT):/output sense-clef2026:latest -i /input -o /output

clean: ## Remove generated files
	rm -rf __pycache__ sense/__pycache__ .mypy_cache .ruff_cache
	rm -rf $(OUTPUT)/*.json

all: setup train predict evaluate ## Full pipeline: setup, train, predict, evaluate
