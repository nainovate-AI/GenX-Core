# monitoring-service/src/core/__init__.py
"""Core module initialization"""

from .config import settings, validate_settings
from .health_checker import HealthChecker, HealthStatus, health_checker
from .opentelemetry_setup import (
    setup_opentelemetry,
    get_tracer,
    get_meter,
    trace_span,
    trace_function,
    MetricsCollector,
    metrics_collector,
    telemetry_manager
)

__all__ = [
    # Configuration
    "settings",
    "validate_settings",
    
    # Health checking
    "HealthChecker",
    "HealthStatus", 
    "health_checker",
    
    # OpenTelemetry
    "setup_opentelemetry",
    "get_tracer",
    "get_meter",
    "trace_span",
    "trace_function",
    "MetricsCollector",
    "metrics_collector",
    "telemetry_manager"
]

# Module version
__version__ = "1.0.0"

# Initialize logging for core module
import logging
logger = logging.getLogger(__name__)
logger.info(f"Core module initialized (version {__version__})")