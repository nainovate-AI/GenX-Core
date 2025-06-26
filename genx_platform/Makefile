ENV ?= dev

.PHONY: deploy deploy-dev deploy-staging deploy-prod

deploy: deploy-$(ENV)

deploy-dev:
	@echo "Deploying to Development"
	docker-compose -f environments/dev/docker-compose.yml up -d

deploy-staging:
	@echo "Deploying to Staging"
	@$(MAKE) build ENV=staging
	@$(MAKE) test ENV=staging
	docker-compose -f environments/staging/docker-compose.yml up -d

deploy-prod:
	@echo "Deploying to Production"
	@$(MAKE) security-scan ENV=production
	@kubectl apply -k k8s/overlays/production/

# Environment-specific builds
build:
	docker build -f docker/Dockerfile.$(ENV) -t llm-service:$(ENV) .

# K8s deployments
k8s-deploy:
	kubectl apply -k k8s/overlays/$(ENV)/