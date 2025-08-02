# monitoring-service/src/api/v1/__init__.py
"""API v1 module initialization"""

from .health import router as health_router
from .metrics import router as metrics_router
from .logs import router as logs_router
from .traces import router as traces_router
from .config import router as config_router

__all__ = [
    "health_router",
    "metrics_router", 
    "logs_router",
    "traces_router",
    "config_router"
]