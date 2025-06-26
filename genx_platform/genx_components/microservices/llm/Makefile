# LLM Service Makefile
SERVICE_NAME = llm-service
REGISTRY ?= localhost:5000
VERSION ?= latest
ENV ?= dev

.PHONY: help build run test clean lock-deps

help:
	@echo "LLM Service Commands:"
	@echo "  make build ENV=dev      - Build for development"
	@echo "  make build ENV=staging  - Build for staging"  
	@echo "  make build ENV=prod     - Build for production"
	@echo "  make run ENV=dev        - Run locally"
	@echo "  make test               - Run tests"
	@echo "  make lock-deps          - Generate requirements.lock.txt"
	@echo "  make k8s-deploy ENV=dev - Deploy to Kubernetes"

# Generate locked dependencies
lock-deps:
	@echo "Generating requirements.lock.txt..."
	@docker run --rm -v $(PWD):/app -w /app python:3.10-slim sh -c \
		"pip install -r requirements.txt && pip freeze > requirements.lock.txt"
	@echo "Generated requirements.lock.txt"

# Build Docker image
build:
	@if [ ! -f requirements.lock.txt ]; then \
		echo "Warning: requirements.lock.txt not found. Generating..."; \
		$(MAKE) lock-deps; \
	fi
	docker build \
		--target $(ENV) \
		-t $(REGISTRY)/$(SERVICE_NAME):$(VERSION)-$(ENV) \
		.

# Run locally with docker
run:
	docker run --rm \
		--env-file envs/$(ENV).env \
		-p 50053:50053 \
		-p 9090:9090 \
		--name $(SERVICE_NAME)-$(ENV) \
		$(REGISTRY)/$(SERVICE_NAME):$(VERSION)-$(ENV)

# Run tests
test:
	docker build --target development -t $(SERVICE_NAME)-test .
	docker run --rm $(SERVICE_NAME)-test pytest tests/

# Health check running container
health:
	@docker exec $(SERVICE_NAME)-$(ENV) python /app/health_check.py || echo "Service unhealthy or not running"

# Security scan
scan:
	@echo "Running security scan..."
	@docker build --target security -t $(SERVICE_NAME)-security .
	@docker run --rm $(SERVICE_NAME)-security cat /scan/safety-report.json | jq '.' || true
	@docker run --rm $(SERVICE_NAME)-security cat /scan/audit-report.json | jq '.' || true

# Deploy to Kubernetes
k8s-deploy:
	kubectl apply -k k8s/overlays/$(ENV)/

# Clean up
clean:
	docker rmi $(REGISTRY)/$(SERVICE_NAME):$(VERSION)-$(ENV) || true
	docker rmi $(SERVICE_NAME)-test || true
	docker rmi $(SERVICE_NAME)-security || true