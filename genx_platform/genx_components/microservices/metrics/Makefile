# genx_platform/genx_components/microservices/metrics/Makefile
# Production Makefile for Metrics Service with Enhanced Features

# Variables
PROJECT_ROOT := ../../..
SERVICE_NAME := metrics-service
IMAGE_NAME := genx/$(SERVICE_NAME)
VERSION ?= 1.0.0
ENVIRONMENT ?= production
DOCKER_REGISTRY ?= docker.io
NAMESPACE ?= genx

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m

# Docker compose files
COMPOSE_BASE := docker-compose.yml
COMPOSE_GPU := docker-compose.gpu.yml
COMPOSE_MONITORING := docker-compose.monitoring.yml
COMPOSE_DEV := docker-compose.override.yml

.PHONY: help
help:
	@echo "$(GREEN)GenX Metrics Service - Production Commands$(NC)"
	@echo "Usage: make [target]"
	@echo ""
	@echo "$(BLUE)Build Commands:$(NC)"
	@echo "  $(GREEN)build$(NC)              - Build production Docker image"
	@echo "  $(GREEN)build-dev$(NC)          - Build development image"
	@echo "  $(GREEN)security-scan$(NC)      - Run security scans on image"
	@echo ""
	@echo "$(BLUE)Deployment Commands:$(NC)"
	@echo "  $(GREEN)up$(NC)                 - Start production stack"
	@echo "  $(GREEN)up-full$(NC)            - Start with monitoring stack"
	@echo "  $(GREEN)up-gpu$(NC)             - Start with GPU support"
	@echo "  $(GREEN)down$(NC)               - Stop all services"
	@echo "  $(GREEN)deploy$(NC)             - Full production deployment"
	@echo ""
	@echo "$(BLUE)Operations:$(NC)"
	@echo "  $(GREEN)status$(NC)             - Check service status"
	@echo "  $(GREEN)health$(NC)             - Health check all services"
	@echo "  $(GREEN)logs$(NC)               - View service logs"
	@echo "  $(GREEN)metrics$(NC)            - View Prometheus metrics"
	@echo "  $(GREEN)scale$(NC)              - Scale service (REPLICAS=n)"
	@echo ""
	@echo "$(BLUE)Monitoring:$(NC)"
	@echo "  $(GREEN)monitor$(NC)            - Open monitoring dashboards"
	@echo "  $(GREEN)alerts$(NC)             - View active alerts"
	@echo "  $(GREEN)traces$(NC)             - Open Jaeger UI"
	@echo ""
	@echo "$(BLUE)Certificate Management:$(NC)"
	@echo "  $(GREEN)certs-generate$(NC)     - Generate TLS certificates"
	@echo "  $(GREEN)certs-verify$(NC)       - Verify certificates"
	@echo ""
	@echo "$(BLUE)Testing:$(NC)"
	@echo "  $(GREEN)test$(NC)               - Run all tests"
	@echo "  $(GREEN)test-integration$(NC)   - Run integration tests"
	@echo "  $(GREEN)test-load$(NC)          - Run load tests"
	@echo "  $(GREEN)grpc-test$(NC)          - Test gRPC endpoints"

# Certificate Management
.PHONY: certs-generate
certs-generate:
	@echo "$(YELLOW)Generating TLS certificates...$(NC)"
	@mkdir -p certs
	@openssl req -x509 -newkey rsa:4096 -nodes -days 365 \
		-keyout certs/ca.key -out certs/ca.crt \
		-subj "/C=US/ST=CA/L=SF/O=GenX/CN=GenX-CA"
	@openssl req -newkey rsa:4096 -nodes \
		-keyout certs/server.key -out certs/server.csr \
		-subj "/C=US/ST=CA/L=SF/O=GenX/CN=metrics-service"
	@openssl x509 -req -days 365 -in certs/server.csr \
		-CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial \
		-out certs/server.crt
	@cp certs/server.crt certs/client.crt
	@cp certs/server.key certs/client.key
	@chmod 600 certs/*.key
	@echo "$(GREEN)✓ Certificates generated$(NC)"

.PHONY: certs-verify
certs-verify:
	@echo "$(YELLOW)Verifying certificates...$(NC)"
	@openssl verify -CAfile certs/ca.crt certs/server.crt
	@echo "$(GREEN)✓ Certificates valid$(NC)"

# Build Commands
.PHONY: build
build:
	@echo "$(YELLOW)Building production Docker image...$(NC)"
	@cd $(PROJECT_ROOT) && \
	docker build \
		--build-arg VERSION=$(VERSION) \
		--target runtime \
		-f genx_components/microservices/metrics/Dockerfile \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)✓ Build complete: $(IMAGE_NAME):$(VERSION)$(NC)"

.PHONY: build-dev
build-dev:
	@echo "$(YELLOW)Building development image...$(NC)"
	@cd $(PROJECT_ROOT) && \
	docker build \
		--build-arg VERSION=dev \
		--target builder \
		-f genx_components/microservices/metrics/Dockerfile \
		-t $(IMAGE_NAME):dev \
		.
	@echo "$(GREEN)✓ Dev build complete$(NC)"

.PHONY: security-scan
security-scan: build
	@echo "$(YELLOW)Running security scans...$(NC)"
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image $(IMAGE_NAME):$(VERSION)
	@echo "$(GREEN)✓ Security scan complete$(NC)"

# Deployment Commands
.PHONY: up
up: certs-verify
	@echo "$(YELLOW)Starting production stack...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=$(ENVIRONMENT) docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@make status

.PHONY: up-full
up-full: certs-verify
	@echo "$(YELLOW)Starting full production stack with monitoring...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=$(ENVIRONMENT) \
		docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_MONITORING) up -d
	@echo "$(GREEN)✓ Full stack started$(NC)"
	@make status

.PHONY: up-gpu
up-gpu: certs-verify
	@echo "$(YELLOW)Starting production stack with GPU support...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=$(ENVIRONMENT) \
		docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_GPU) up -d
	@echo "$(GREEN)✓ Services started with GPU support$(NC)"
	@make status

.PHONY: down
down:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_MONITORING) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

.PHONY: deploy
deploy: build security-scan certs-generate
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=production make up-full
	@make health
	@echo "$(GREEN)✓ Deployment complete$(NC)"

# Operations
.PHONY: status
status:
	@echo "$(YELLOW)Service Status:$(NC)"
	@docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_MONITORING) ps
	@echo ""
	@echo "$(YELLOW)Resource Usage:$(NC)"
	@docker stats --no-stream $$(docker-compose ps -q) 2>/dev/null || true

.PHONY: health
health:
	@echo "$(YELLOW)Health Checks:$(NC)"
	@echo -n "Metrics Service: "
	@docker exec genx-metrics-service python -m grpc_health.v1.health_check --address=localhost:50056 2>/dev/null && \
		echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Unhealthy$(NC)"
	@echo -n "Prometheus: "
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/-/healthy | grep -q "200" && \
		echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Unhealthy$(NC)"
	@echo -n "Grafana: "
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/health | grep -q "200" && \
		echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Unhealthy$(NC)"

.PHONY: logs
logs:
	@docker-compose logs -f --tail=100 metrics-service

.PHONY: logs-all
logs-all:
	@docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_MONITORING) logs -f --tail=50

.PHONY: scale
scale:
	@echo "$(YELLOW)Scaling metrics service to $(REPLICAS) replicas...$(NC)"
	@docker-compose up -d --scale metrics-service=$(REPLICAS)
	@echo "$(GREEN)✓ Scaled to $(REPLICAS) replicas$(NC)"

# Monitoring
.PHONY: monitor
monitor:
	@echo "$(GREEN)Monitoring Dashboards:$(NC)"
	@echo "  - Grafana: http://localhost:3001 (admin/changeme)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Jaeger: http://localhost:16686"
	@echo "  - AlertManager: http://localhost:9093"
	@echo "  - Consul: http://localhost:8500"
	@echo "  - Envoy Admin: http://localhost:9901"

.PHONY: alerts
alerts:
	@echo "$(YELLOW)Active Alerts:$(NC)"
	@curl -s http://localhost:9093/api/v1/alerts | jq '.data[] | {labels: .labels, state: .status.state}' || \
		echo "$(RED)AlertManager not available$(NC)"

.PHONY: traces
traces:
	@echo "$(GREEN)Opening Jaeger UI...$(NC)"
	@python -m webbrowser http://localhost:16686 2>/dev/null || \
		echo "Open http://localhost:16686 in your browser"

# Testing
.PHONY: test
test: test-unit test-integration

.PHONY: test-unit
test-unit:
	@echo "$(YELLOW)Running unit tests...$(NC)"
	@docker run --rm -v $(PWD):/app $(IMAGE_NAME):$(VERSION) pytest tests/unit
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

.PHONY: test-integration
test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@sleep 5  # Wait for services
	@docker run --rm --network metrics_genx-network \
		-v $(PWD)/tests:/tests \
		$(IMAGE_NAME):$(VERSION) pytest tests/integration
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

.PHONY: test-load
test-load:
	@echo "$(YELLOW)Running load tests...$(NC)"
	@docker run --rm --network metrics_genx-network \
		-v $(PWD)/tests:/tests \
		$(IMAGE_NAME):$(VERSION) locust -f tests/load/locustfile.py \
		--host=metrics-service:50056 --users=100 --spawn-rate=10 --run-time=60s
	@echo "$(GREEN)✓ Load tests complete$(NC)"

.PHONY: grpc-test
grpc-test:
	@echo "$(YELLOW)Testing gRPC endpoints...$(NC)"
	@docker run --rm --network metrics_genx-network \
		fullstorydev/grpcurl:latest \
		-plaintext metrics-service:50056 list
	@docker run --rm --network metrics_genx-network \
		fullstorydev/grpcurl:latest \
		-plaintext -d '{"metadata": {"request_id": "test-123"}}' \
		metrics-service:50056 genx.metrics.v1.MetricsService/GetSystemMetrics

# Utility Commands
.PHONY: shell
shell:
	@docker exec -it genx-metrics-service /bin/bash

.PHONY: metrics
metrics:
	@echo "$(YELLOW)Fetching Prometheus metrics...$(NC)"
	@curl -s http://localhost:9091/metrics | grep -E "^(genx_|grpc_|process_)" | head -20

.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@docker-compose -f $(COMPOSE_BASE) -f $(COMPOSE_MONITORING) down -v
	@docker rmi $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest 2>/dev/null || true
	@rm -rf certs logs
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Kubernetes Commands (Future)
.PHONY: k8s-generate
k8s-generate:
	@echo "$(YELLOW)Generating Kubernetes manifests...$(NC)"
	@mkdir -p k8s
	@helm template $(SERVICE_NAME) ./helm \
		--set image.tag=$(VERSION) \
		--set environment=$(ENVIRONMENT) \
		> k8s/manifests.yaml
	@echo "$(GREEN)✓ Kubernetes manifests generated$(NC)"

.PHONY: k8s-deploy
k8s-deploy: k8s-generate
	@echo "$(YELLOW)Deploying to Kubernetes...$(NC)"
	@kubectl apply -f k8s/manifests.yaml
	@echo "$(GREEN)✓ Deployed to Kubernetes$(NC)"

# Default target
.DEFAULT_GOAL := help