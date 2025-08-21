.PHONY: dev lint test build run clean

# Default target
all: dev lint test

# Development setup
dev:
	@echo "Setting up development environment..."
	cd shared && pip install -e ".[dev]"
	cd api && pip install -e ".[dev]"
	cd worker && pip install -e ".[dev]"
	pre-commit install

# Linting
lint:
	@echo "Running linters..."
	ruff check .
	black --check .
	isort --check .
	cd shared && mypy vsr_shared

# Testing
test:
	@echo "Running tests..."
	cd api && pytest -xvs
	cd worker && pytest -xvs

# Build Docker images
build:
	@echo "Building Docker images..."
	docker build -t vsr-api:latest -f infra/api.Dockerfile .
	docker build -t vsr-worker:latest -f infra/worker.Dockerfile .

# Run local development environment
run:
	@echo "Starting local development environment..."
	docker compose -f infra/docker-compose.local.yml up

# Run local development environment in detached mode
run-detached:
	@echo "Starting local development environment in detached mode..."
	docker compose -f infra/docker-compose.local.yml up -d

# Stop local development environment
stop:
	@echo "Stopping local development environment..."
	docker compose -f infra/docker-compose.local.yml down

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "coverage.json" -delete
