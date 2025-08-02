# monitoring-service/src/integrations/__init__.py
"""Integrations module for monitoring service"""

from .grpc_integration import (
    MonitoringClientInterceptor,
    MonitoringServerInterceptor,
    AsyncMonitoringServerInterceptor,
    create_monitored_channel,
    create_async_monitored_channel,
    add_monitoring_interceptors,
    add_async_monitoring_interceptors,
    monitor_grpc_method,
    GrpcMonitoringMixin
)

from .http_integration import (
    MonitoringHTTPMiddleware,
    MonitoringRoute,
    HTTPXMonitoringTransport,
    MonitoringHTTPClient,
    create_monitored_http_client,
    monitor_endpoint,
    get_trace_context,
    monitored_error_handler,
    http_operation_context
)

from .database_integration import (
    DatabaseType,
    DatabaseMonitor,
    QueryMetrics,
    SQLAlchemyMonitor,
    AsyncPGMonitor,
    MongoDBMonitor,
    RedisMonitor,
    MonitoredAsyncPGPool,
    MonitoredMongoClient,
    MonitoredMongoDatabase,
    MonitoredMongoCollection,
    monitor_db_operation,
    create_monitored_engine,
    create_monitored_asyncpg_pool,
    create_monitored_mongo_client
)

__all__ = [
    # gRPC Integration
    "MonitoringClientInterceptor",
    "MonitoringServerInterceptor",
    "AsyncMonitoringServerInterceptor",
    "create_monitored_channel",
    "create_async_monitored_channel",
    "add_monitoring_interceptors",
    "add_async_monitoring_interceptors",
    "monitor_grpc_method",
    "GrpcMonitoringMixin",
    
    # HTTP Integration
    "MonitoringHTTPMiddleware",
    "MonitoringRoute",
    "HTTPXMonitoringTransport",
    "MonitoringHTTPClient",
    "create_monitored_http_client",
    "monitor_endpoint",
    "get_trace_context",
    "monitored_error_handler",
    "http_operation_context",
    
    # Database Integration
    "DatabaseType",
    "DatabaseMonitor",
    "QueryMetrics",
    "SQLAlchemyMonitor",
    "AsyncPGMonitor",
    "MongoDBMonitor",
    "RedisMonitor",
    "MonitoredAsyncPGPool",
    "MonitoredMongoClient",
    "MonitoredMongoDatabase",
    "MonitoredMongoCollection",
    "monitor_db_operation",
    "create_monitored_engine",
    "create_monitored_asyncpg_pool",
    "create_monitored_mongo_client"
]

# Module version
__version__ = "1.0.0"

# Initialize logging for integrations module
import logging
logger = logging.getLogger(__name__)
logger.info(f"Integrations module initialized (version {__version__})")