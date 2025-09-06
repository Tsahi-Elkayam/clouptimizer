.PHONY: help install install-dev test lint format clean run-cli run-api docker-build docker-run

# Default target
help:
	@echo "Clouptimizer - Multi-cloud Cost Optimization Tool"
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install base dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  install-all   Install all dependencies (base + all providers)"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code with black"
	@echo "  clean         Clean temporary files"
	@echo "  run-cli       Run CLI interface"
	@echo "  run-api       Run API server"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"

# Installation targets
install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/base.txt
	pip install -r requirements/dev.txt

install-all:
	pip install -r requirements/base.txt
	pip install -r requirements/aws.txt
	pip install -r requirements/azure.txt
	pip install -r requirements/gcp.txt
	pip install -r requirements/dev.txt

# Development targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Clean targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/

# Run targets
run-cli:
	python -m src.cli.main

run-api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Docker targets
docker-build:
	docker build -t clouptimizer:latest .

docker-run:
	docker run -it --rm \
		-v ~/.aws:/root/.aws:ro \
		-v ~/.azure:/root/.azure:ro \
		-v ~/.config/gcloud:/root/.config/gcloud:ro \
		clouptimizer:latest

# Development shortcuts
dev: install-dev
	@echo "Development environment ready!"

quick-scan:
	python -m src.cli.main quick --provider aws

demo:
	python -m src.cli.main scan --provider aws --regions us-east-1 -o scan_demo.json
	python -m src.cli.main analyze --scan-file scan_demo.json -o analysis_demo.json
	python -m src.cli.main report --analysis-file analysis_demo.json --format html -o report_demo.html
	@echo "Demo complete! Check report_demo.html"