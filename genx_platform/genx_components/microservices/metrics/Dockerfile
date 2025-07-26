# genx_platform/genx_components/microservices/metrics/Dockerfile
# Production Dockerfile for Metrics Service with Enhanced Security

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy and install Python dependencies with specific versions
COPY genx_components/microservices/metrics/requirements.txt .

# Ensure grpcio and grpcio-tools versions match
RUN pip install --no-cache-dir --user \
    grpcio==1.74.0 \
    grpcio-tools==1.74.0 \
    grpcio-reflection==1.74.0 \
    grpcio-health-checking==1.74.0

# Install remaining requirements
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Proto Generator
FROM python:3.11-slim AS proto-generator

# Install protobuf compiler and sed
RUN apt-get update && apt-get install -y sed && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir grpcio-tools==1.74.0

# Copy proto files
WORKDIR /genx_platform
COPY protos/ ./protos/

# Create full directory structure
RUN mkdir -p genx_components/common/grpc \
    && mkdir -p genx_components/microservices/grpc \
    && mkdir -p genx_components/microservices/metrics/src/utils \
    && mkdir -p genx_components/microservices/metrics/src/collectors \
    && mkdir -p genx_components/microservices/metrics/src/service

# Generate proto files
RUN python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./genx_components/common/grpc \
    --grpc_python_out=./genx_components/common/grpc \
    ./protos/common.proto

RUN python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./genx_components/microservices/grpc \
    --grpc_python_out=./genx_components/microservices/grpc \
    ./protos/common.proto ./protos/metrics_service.proto

# Fix imports - CORRECTED VERSION
# Fix the import in metrics_service_pb2.py to use absolute import
RUN sed -i 's/^import common_pb2 as common__pb2$/from genx_components.common.grpc import common_pb2 as common__pb2/g' \
    /genx_platform/genx_components/microservices/grpc/metrics_service_pb2.py

# Fix the imports in the grpc files
RUN sed -i 's/^import common_pb2/from . import common_pb2/g' \
    /genx_platform/genx_components/microservices/grpc/common_pb2_grpc.py 2>/dev/null || true \
    && sed -i 's/^import metrics_service_pb2/from . import metrics_service_pb2/g' \
    /genx_platform/genx_components/microservices/grpc/metrics_service_pb2_grpc.py

# Create all __init__.py files for proper Python package structure
# Create all __init__.py files
RUN touch /genx_platform/genx_components/__init__.py \
    && touch /genx_platform/genx_components/common/__init__.py \
    && touch /genx_platform/genx_components/common/grpc/__init__.py \
    && touch /genx_platform/genx_components/microservices/__init__.py \
    && touch /genx_platform/genx_components/microservices/grpc/__init__.py \
    && touch /genx_platform/genx_components/microservices/metrics/__init__.py \
    && touch /genx_platform/genx_components/microservices/metrics/src/__init__.py \
    && touch /genx_platform/genx_components/microservices/metrics/src/utils/__init__.py \
    && touch /genx_platform/genx_components/microservices/metrics/src/collectors/__init__.py \
    && touch /genx_platform/genx_components/microservices/metrics/src/service/__init__.py
# Stage 3: Security Scanner
FROM python:3.11-slim AS security-scanner

# Install security scanning tools
RUN pip install --no-cache-dir safety bandit

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Run security scans
WORKDIR /scan
COPY genx_components/microservices/metrics/requirements.txt .
RUN safety check -r requirements.txt || true
COPY genx_components/microservices/metrics/src ./src
RUN bandit -r src -f json -o /tmp/bandit-report.json || true

# Stage 4: Runtime
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash metrics \
    && groupadd -g 1001 metricsgroup \
    && usermod -a -G metricsgroup metrics

# Create necessary directories with proper permissions
RUN mkdir -p /genx_platform /models /logs /certs \
    && chown -R metrics:metricsgroup /genx_platform /models /logs /certs \
    && chmod 750 /models /logs \
    && chmod 755 /certs

# Set working directory
WORKDIR /genx_platform

# Copy Python packages from builder
COPY --from=builder /root/.local /home/metrics/.local

# Copy application code
COPY --chown=metrics:metricsgroup . .

# Copy generated proto files from proto-generator
COPY --from=proto-generator --chown=metrics:metricsgroup \
    /genx_platform/genx_components/common/grpc \
    ./genx_components/common/grpc
COPY --from=proto-generator --chown=metrics:metricsgroup \
    /genx_platform/genx_components/microservices/grpc \
    ./genx_components/microservices/grpc

# Copy security scan results
COPY --from=security-scanner --chown=metrics:metricsgroup \
    /tmp/bandit-report.json /tmp/security-scan.json

# Set environment
ENV PATH=/home/metrics/.local/bin:$PATH \
    PYTHONPATH=/genx_platform \
    PYTHONUNBUFFERED=1 \
    GRPC_VERBOSITY=ERROR \
    GRPC_ENABLE_FORK_SUPPORT=1 \
    PYTHONDONTWRITEBYTECODE=1

# Production environment variables with secure defaults
ENV SERVICE_NAME=metrics-service \
    SERVICE_VERSION=${VERSION:-1.0.0} \
    SERVICE_PORT=50056 \
    ENVIRONMENT=production \
    DEBUG=false \
    TELEMETRY_ENABLED=true \
    TELEMETRY_ENDPOINT=http://otel-collector:4317 \
    GRPC_MAX_WORKERS=8 \
    GRPC_MAX_MESSAGE_LENGTH=104857600 \
    METRICS_PORT=9090 \
    CACHE_TTL_SECONDS=30 \
    BACKGROUND_COLLECTION_INTERVAL=30 \
    MODEL_STORAGE_PATH=/models \
    LOG_PATH=/logs \
    GRPC_TLS_ENABLED=true \
    ENABLE_AUTH=true \
    RATE_LIMIT_ENABLED=true

# Security: Remove shell access for production
RUN rm /bin/sh && ln -s /bin/false /bin/sh

# Switch to non-root user
USER metrics:metricsgroup

# Set working directory for service
WORKDIR /genx_platform/genx_components/microservices/metrics

# Expose ports (non-privileged)
EXPOSE 50056 9090

# Labels for container metadata
LABEL maintainer="GenX Platform Team" \
      version="${VERSION:-1.0.0}" \
      description="OPEA-compliant Metrics Microservice" \
      security.scan="/tmp/security-scan.json"

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -m grpc_health.v1.health_check --address=localhost:50056 || exit 1

# Entry point with exec to ensure proper signal handling
ENTRYPOINT ["python", "-m", "src.main"]