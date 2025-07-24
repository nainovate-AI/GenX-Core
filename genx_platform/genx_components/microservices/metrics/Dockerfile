# Production Dockerfile for Metrics Service
# Multi-stage build for optimal image size and security

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

# Copy and install Python dependencies
COPY genx_components/microservices/metrics/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Proto Generator
FROM python:3.11-slim AS proto-generator

# Install protobuf compiler
RUN pip install --no-cache-dir grpcio-tools

# Copy proto files
WORKDIR /genx_platform
COPY protos/ ./protos/

# Create output directories
RUN mkdir -p genx_components/common/grpc \
    && mkdir -p genx_components/microservices/grpc

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

# Fix imports
RUN sed -i 's/^import common_pb2/from . import common_pb2/g' \
    /genx_platform/genx_components/microservices/grpc/*_pb2_grpc.py \
    && sed -i 's/^import metrics_service_pb2/from . import metrics_service_pb2/g' \
    /genx_platform/genx_components/microservices/grpc/*_pb2_grpc.py

# Create __init__.py files
RUN touch /genx_platform/genx_components/common/grpc/__init__.py \
    && touch /genx_platform/genx_components/microservices/grpc/__init__.py

# Stage 3: Runtime
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash metrics

# Set working directory
WORKDIR /genx_platform

# Copy Python packages from builder
COPY --from=builder /root/.local /home/metrics/.local

# Copy application code
COPY --chown=metrics:metrics . .

# Copy generated proto files from proto-generator
COPY --from=proto-generator --chown=metrics:metrics \
    /genx_platform/genx_components/common/grpc \
    ./genx_components/common/grpc
COPY --from=proto-generator --chown=metrics:metrics \
    /genx_platform/genx_components/microservices/grpc \
    ./genx_components/microservices/grpc

# Set environment
ENV PATH=/home/metrics/.local/bin:$PATH \
    PYTHONPATH=/genx_platform \
    PYTHONUNBUFFERED=1 \
    GRPC_VERBOSITY=ERROR \
    GRPC_ENABLE_FORK_SUPPORT=1

# Production environment variables
ENV SERVICE_NAME=metrics-service \
    SERVICE_VERSION=1.0.0 \
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
    MODEL_STORAGE_PATH=/models

# Create necessary directories
RUN mkdir -p /models && chown metrics:metrics /models

# Switch to non-root user
USER metrics

# Set working directory
WORKDIR /genx_platform/genx_components/microservices/metrics

# Expose ports
EXPOSE 50056 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -m grpc_health.v1.health_check --address=localhost:50056 || exit 1

# Entry point
ENTRYPOINT ["python", "-m", "src.main"]