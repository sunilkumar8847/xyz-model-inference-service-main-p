# ================================================================
# XYZ MDM 3.0 — Model Inference Service Makefile
# ================================================================
.PHONY: install run-local run-dev test lint fmt clean help

install: ## Install dependencies via Poetry
	poetry install

env-setup: ## Create .env.local from .env.local.example
	@if not exist .env.local (copy .env.local.example .env.local && echo .env.local created — fill in your values.) else (echo .env.local already exists.)

run-local: ## Run locally (loads .env + .env.local via Pydantic Settings)
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8090 --reload

run-dev: ## Run in dev mode with hot reload
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8090 --reload --log-level debug

run: ## Run without reload
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8090

test: ## Run tests
	poetry run pytest tests/ src/tests/ -v

test-cov: ## Run tests with coverage
	poetry run pytest tests/ src/tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linter
	poetry run ruff check src/ 2>nul || echo ruff not installed, skipping

fmt: ## Format code
	poetry run ruff check src/ --fix 2>nul || echo ruff not installed, skipping

docker-build: ## Build Docker image
	docker build -t xyz-mdm/model-inference-service:3.0.0 .

compose-up: ## Start with docker-compose
	docker-compose up -d

compose-down: ## Stop docker-compose
	docker-compose down

compose-logs: ## View logs
	docker-compose logs -f model-inference-service

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	find . -type f -name "*.pyc" -delete 2>nul || true
	rmdir /s /q .pytest_cache 2>nul || true
	rmdir /s /q htmlcov 2>nul || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
