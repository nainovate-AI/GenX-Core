import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from core.opentelemetry_setup import setup_opentelemetry
from api.v1 import metrics, logs, traces, health, config
from services import (
    prometheus_service,
    loki_service,
    jaeger_service,
    grafana_service,
    alertmanager_service,
    otel_collector_service
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("Starting Monitoring Service...")
    
    # Initialize OpenTelemetry
    setup_opentelemetry(settings.SERVICE_NAME)
    
    # Start monitoring components
    await start_monitoring_stack()
    
    # Health check loop
    asyncio.create_task(health_check_loop())
    
    yield
    
    # Shutdown
    print("Shutting down Monitoring Service...")
    await stop_monitoring_stack()

app = FastAPI(
    title="Monitoring Microservice",
    description="Centralized monitoring service with OpenTelemetry",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
app.include_router(logs.router, prefix="/api/v1/logs", tags=["logs"])
app.include_router(traces.router, prefix="/api/v1/traces", tags=["traces"])
app.include_router(config.router, prefix="/api/v1/config", tags=["config"])

async def start_monitoring_stack():
    """Start all monitoring components"""
    await otel_collector_service.start()
    await prometheus_service.start()
    await loki_service.start()
    await jaeger_service.start()
    await alertmanager_service.start()
    await grafana_service.start()

async def stop_monitoring_stack():
    """Stop all monitoring components"""
    await grafana_service.stop()
    await alertmanager_service.stop()
    await jaeger_service.stop()
    await loki_service.stop()
    await prometheus_service.stop()
    await otel_collector_service.stop()

async def health_check_loop():
    """Continuous health monitoring"""
    while True:
        await asyncio.sleep(30)
        # Check health of all services
        # Update status metrics