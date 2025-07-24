# Production Makefile for Metrics Service

# Variables
PROJECT_ROOT := ../../..
SERVICE_NAME := metrics-service
IMAGE_NAME := genx/$(SERVICE_NAME)
VERSION ?= 1.0.0
ENVIRONMENT ?= production

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

.PHONY: help
help:
	@echo "$(GREEN)GenX Metrics Service - Production Commands$(NC)"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  $(GREEN)build$(NC)       - Build production Docker image"
	@echo "  $(GREEN)up$(NC)          - Start all services in production mode"
	@echo "  $(GREEN)down$(NC)        - Stop all services"
	@echo "  $(GREEN)restart$(NC)     - Restart metrics service"
	@echo "  $(GREEN)logs$(NC)        - View service logs"
	@echo "  $(GREEN)test$(NC)        - Run production tests"
	@echo "  $(GREEN)monitor$(NC)     - Open monitoring dashboards"
	@echo "  $(GREEN)scale$(NC)       - Scale metrics service (REPLICAS=n)"
	@echo "  $(GREEN)status$(NC)      - Check service status"
	@echo "  $(GREEN)clean$(NC)       - Clean up everything"

.PHONY: build
build:
	@echo "$(YELLOW)Building production Docker image...$(NC)"
	@cd $(PROJECT_ROOT) && \
	docker build \
		--build-arg VERSION=$(VERSION) \
		-f genx_components/microservices/metrics/Dockerfile \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)✓ Build complete: $(IMAGE_NAME):$(VERSION)$(NC)"

.PHONY: up
up:
	@echo "$(YELLOW)Starting production stack...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=$(ENVIRONMENT) docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@make status

.PHONY: up-gpu
up-gpu:
	@echo "$(YELLOW)Starting production stack with GPU support...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=$(ENVIRONMENT) docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
	@echo "$(GREEN)✓ Services started with GPU support$(NC)"
	@make status

.PHONY: up-dev
up-dev:
	@echo "$(YELLOW)Starting development stack (no GPU)...$(NC)"
	@ENVIRONMENT=development docker-compose up -d
	@echo "$(GREEN)✓ Development services started$(NC)"
	@make status

.PHONY: down
down:
	@echo "$(YELLOW)Stopping production stack...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

.PHONY: restart
restart:
	@echo "$(YELLOW)Restarting metrics service...$(NC)"
	@docker-compose restart metrics-service
	@echo "$(GREEN)✓ Service restarted$(NC)"

.PHONY: logs
logs:
	@docker-compose logs -f --tail=100 metrics-service

.PHONY: logs-all
logs-all:
	@docker-compose logs -f --tail=100

.PHONY: test
test:
	@echo "$(YELLOW)Running production tests...$(NC)"
	@sleep 5  # Wait for services to be ready
	@python scripts/test_production.py || (echo "$(RED)✗ Tests failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ All tests passed$(NC)"

.PHONY: monitor
monitor:
	@echo "$(GREEN)Monitoring Dashboards:$(NC)"
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Jaeger: http://localhost:16686"
	@echo "  - Metrics Service: localhost:50056"

.PHONY: scale
scale:
	@echo "$(YELLOW)Scaling metrics service to $(REPLICAS) replicas...$(NC)"
	@docker-compose up -d --scale metrics-service=$(REPLICAS)
	@echo "$(GREEN)✓ Scaled to $(REPLICAS) replicas$(NC)"

.PHONY: status
status:
	@echo "$(YELLOW)Service Status:$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(YELLOW)Resource Usage:$(NC)"
	@docker stats --no-stream $$(docker-compose ps -q)

.PHONY: health
health:
	@echo "$(YELLOW)Health Check:$(NC)"
	@docker exec genx-metrics-service python -m grpc_health.v1.health_check --address=localhost:50056 && \
		echo "$(GREEN)✓ Service is healthy$(NC)" || \
		echo "$(RED)✗ Service is unhealthy$(NC)"

.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@docker-compose down -v
	@docker rmi $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Production deployment commands
.PHONY: deploy
deploy: build
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@VERSION=$(VERSION) ENVIRONMENT=production docker-compose up -d
	@make health
	@echo "$(GREEN)✓ Deployment complete$(NC)"

.PHONY: rollback
rollback:
	@echo "$(YELLOW)Rolling back to previous version...$(NC)"
	@docker-compose down
	@VERSION=$(PREVIOUS_VERSION) docker-compose up -d
	@echo "$(GREEN)✓ Rollback complete$(NC)"

# Debug commands
.PHONY: shell
shell:
	@docker exec -it genx-metrics-service /bin/bash

.PHONY: grpc-test
grpc-test:
	@echo "$(YELLOW)Testing gRPC endpoint...$(NC)"
	@docker run --rm --network metrics_genx-network \
		fullstorydev/grpcurl:latest \
		-plaintext metrics-service:50056 list

.PHONY: metrics
metrics:
	@echo "$(YELLOW)Fetching Prometheus metrics...$(NC)"
	@curl -s http://localhost:9091/metrics | head -20

# Monitoring setup
.PHONY: setup-monitoring
setup-monitoring:
	@echo "$(YELLOW)Setting up monitoring configuration...$(NC)"
	@mkdir -p monitoring/prometheus/alerts
	@mkdir -p monitoring/grafana/provisioning/dashboards
	@mkdir -p monitoring/grafana/provisioning/datasources
	@mkdir -p monitoring/grafana/dashboards
	@echo "$(GREEN)✓ Monitoring directories created$(NC)"