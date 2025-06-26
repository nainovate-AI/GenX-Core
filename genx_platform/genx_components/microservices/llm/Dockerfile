# Single Dockerfile supporting all environments
ARG PYTHON_VERSION=3.10

#################################################
# Base stage - Common for all environments
#################################################
FROM python:${PYTHON_VERSION}-slim as base

# Install common system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements.lock.txt* ./

#################################################
# Development stage
#################################################
FROM base as development

# Install dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with dev dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    ipython \
    ipdb \
    pytest \
    pytest-asyncio \
    black \
    ruff \
    debugpy

# Copy entire application
COPY . .

# Copy scripts and make executable
COPY scripts/health_check.py /app/health_check.py
RUN chmod +x /app/health_check.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python /app/health_check.py

# Development command
CMD ["python", "start_service.py"]

#################################################
# Security scanner stage
#################################################
FROM python:${PYTHON_VERSION}-slim as security

WORKDIR /scan

# Copy requirements files
COPY requirements.txt requirements.lock.txt* ./

# Create requirements.lock.txt if it doesn't exist
RUN if [ ! -f requirements.lock.txt ]; then \
        pip install --no-cache-dir -r requirements.txt && \
        pip freeze > requirements.lock.txt; \
    fi

# Run security scans
RUN pip install --no-cache-dir safety pip-audit && \
    safety check -r requirements.lock.txt -o json > safety-report.json || true && \
    pip-audit -r requirements.lock.txt -f json -o audit-report.json || true

#################################################
# Builder stage - Build dependencies
#################################################
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use requirements.lock.txt for reproducible builds
COPY requirements.txt requirements.lock.txt* ./
RUN if [ -f requirements.lock.txt ]; then \
        pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.lock.txt; \
    else \
        pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt; \
    fi

#################################################
# Production stage
#################################################
FROM python:${PYTHON_VERSION}-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 genx && \
    useradd -m -u 1000 -g genx genx && \
    mkdir -p /app /models /cache /logs && \
    chown -R genx:genx /app /models /cache /logs

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy security scan results
COPY --from=security /scan/*-report.json /app/security/

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=genx:genx . .

# Copy scripts and make executable
COPY --chown=genx:genx scripts/health_check.py /app/health_check.py
RUN chmod +x /app/health_check.py

# Set environment
ENV PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/cache \
    TRANSFORMERS_CACHE=/cache

# Switch to non-root user
USER genx

# Expose ports
EXPOSE 50053 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/health_check.py || exit 1

# Use tini for signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the service with optimization flag
CMD ["python", "-O", "start_service.py"]

#################################################
# Staging stage - Production with debug capability
#################################################
FROM production as staging

# Switch back to root to install debug tools
USER root

# Install minimal debug tools
RUN pip install --no-cache-dir py-spy memory-profiler

# Copy debug script
COPY --chown=genx:genx scripts/debug.sh /app/debug.sh
RUN chmod +x /app/debug.sh

# Switch back to non-root user
USER genx

# Same health check as production
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/health_check.py || exit 1

# Same entrypoint and command as production
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-O", "start_service.py"]