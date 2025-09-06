# Clouptimizer Makefile - Production Ready Commands

.PHONY: help install install-dev test test-unit test-integration lint format clean docker-build docker-up docker-down deploy

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := clouptimizer
VERSION := 0.1.0
REGISTRY := your-registry.com

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Clouptimizer - Production Ready Cloud Cost Optimization Tool$(NC)"
	@echo "$(YELLOW)Version: $(VERSION)$(NC)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(YELLOW)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements/prod.txt
	$(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements/dev.txt
	$(PIP) install -e .
	@echo "$(GREEN)Development installation complete!$(NC)"

dev: ## Run development server
	@echo "$(YELLOW)Starting development server...$(NC)"
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/ -v -m unit

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/ -v -m integration

test-security: ## Run security tests
	@echo "$(YELLOW)Running security scan...$(NC)"
	bandit -r src/ -f json -o security-report.json
	safety check --json

lint: ## Run code linting
	@echo "$(YELLOW)Running linters...$(NC)"
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	black src/ tests/
	ruff check --fix src/ tests/
	@echo "$(GREEN)Formatting complete!$(NC)"

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage .pytest_cache/
	rm -rf logs/* data/* reports/*
	@echo "$(GREEN)Clean complete!$(NC)"

# Cloud Operations
scan-aws: ## Run AWS cost scan
	@echo "$(YELLOW)Scanning AWS resources...$(NC)"
	python -m src.cli.main scan --provider aws

analyze: ## Run cost analysis
	@echo "$(YELLOW)Running cost analysis...$(NC)"
	python -m src.cli.main analyze

optimize: ## Generate optimization recommendations
	@echo "$(YELLOW)Generating optimization recommendations...$(NC)"
	python -m src.cli.main optimize --dry-run

report: ## Generate cost report
	@echo "$(YELLOW)Generating cost report...$(NC)"
	python -m src.cli.main report --format html

docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker build complete!$(NC)"

docker-up: ## Start Docker services
	@echo "$(YELLOW)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Prometheus: http://localhost:9091"
	@echo "Grafana: http://localhost:3000"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean Docker resources
	@echo "$(YELLOW)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER) system prune -f
	@echo "$(GREEN)Docker clean complete!$(NC)"

# Monitoring & Health
health-check: ## Check service health
	@echo "$(YELLOW)Checking service health...$(NC)"
	@curl -s http://localhost:8000/health/detailed | python -m json.tool || echo "$(RED)Service not responding$(NC)"

metrics: ## Show current metrics
	@echo "$(YELLOW)Fetching metrics...$(NC)"
	@curl -s http://localhost:8000/metrics/json | python -m json.tool || echo "$(RED)Metrics not available$(NC)"

monitor: ## Start monitoring dashboard
	@echo "$(YELLOW)Opening monitoring dashboard...$(NC)"
	@python -m webbrowser http://localhost:3000 2>/dev/null || echo "Open http://localhost:3000 in your browser"

# Deployment
deploy-prod: ## Deploy to production
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@echo "$(RED)WARNING: This will deploy to production!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(REGISTRY)/$(PROJECT_NAME):$(VERSION)
	$(DOCKER) push $(REGISTRY)/$(PROJECT_NAME):$(VERSION)
	@echo "$(GREEN)Production deployment complete!$(NC)"

# Utilities
validate-config: ## Validate configuration files
	@echo "$(YELLOW)Validating configuration...$(NC)"
	python -c "from src.core.config import get_settings; s = get_settings(); print('Config valid!') if s.validate_providers() else print('Config invalid!')"

security-scan: ## Run comprehensive security scan
	@echo "$(YELLOW)Running security scan...$(NC)"
	trivy fs --security-checks vuln,config .
	@echo "$(GREEN)Security scan complete!$(NC)"

.DEFAULT_GOAL := help