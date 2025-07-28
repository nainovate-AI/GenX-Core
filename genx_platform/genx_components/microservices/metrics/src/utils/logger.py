"""
Production-grade structured logging setup
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
import os

import structlog
from pythonjsonlogger import jsonlogger

# Global flag to prevent duplicate Prometheus initialization
_prometheus_initialized = False

def setup_logging(name: str) -> structlog.BoundLogger:
    """
    Setup structured logging for production
    Returns a structlog logger instance
    """
    
    # Get log level from environment
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    # Configure Python's logging
    if not logging.getLogger().handlers:
        # Create console handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        
        # Use JSON formatter for production
        if os.environ.get("ENVIRONMENT") == "production":
            formatter = jsonlogger.JsonFormatter(
                fmt='%(timestamp)s %(level)s %(name)s %(message)s',
                rename_fields={
                    'timestamp': '@timestamp',
                    'level': 'level',
                    'name': 'logger',
                    'message': 'message'
                }
            )
        else:
            # Human-readable format for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(getattr(logging, log_level))
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_service_context,
            structlog.processors.JSONRenderer() if os.environ.get("ENVIRONMENT") == "production" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Return a bound logger for the module
    return structlog.get_logger(name)


def add_service_context(logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service context to all log entries"""
    
    # Add service metadata
    event_dict["service"] = "metrics-service"
    event_dict["version"] = "1.0.0"
    event_dict["environment"] = os.environ.get("ENVIRONMENT", "development")
    
    # Add hostname if available
    hostname = os.environ.get("HOSTNAME")
    if hostname:
        event_dict["hostname"] = hostname
    
    # Add pod name for Kubernetes
    pod_name = os.environ.get("POD_NAME")
    if pod_name:
        event_dict["pod_name"] = pod_name
    
    # Add trace ID if available (from OpenTelemetry)
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            event_dict["trace_id"] = format(span_context.trace_id, '032x')
            event_dict["span_id"] = format(span_context.span_id, '016x')
    except ImportError:
        pass
    
    return event_dict


class MetricsLogger:
    """
    Wrapper for common logging patterns in metrics service
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_metric_collection(self, metric_type: str, duration: float, success: bool, error: str = None):
        """Log metric collection event"""
        self.logger.info(
            "metric_collection",
            metric_type=metric_type,
            duration_ms=int(duration * 1000),
            success=success,
            error=error
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool = None):
        """Log cache operation"""
        self.logger.debug(
            "cache_operation",
            operation=operation,
            key=key,
            hit=hit
        )
    
    def log_grpc_request(self, method: str, request_id: str, duration: float = None, status_code: str = None):
        """Log gRPC request"""
        self.logger.info(
            "grpc_request",
            method=method,
            request_id=request_id,
            duration_ms=int(duration * 1000) if duration else None,
            status_code=status_code
        )
    
    def log_alert(self, resource: str, metric: str, value: float, threshold: float):
        """Log resource alert"""
        self.logger.warning(
            "resource_alert",
            resource=resource,
            metric=metric,
            value=value,
            threshold=threshold,
            exceeded=value > threshold
        )
    
    def log_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(
            f"{operation}_error",
            error=str(error),
            error_type=type(error).__name__,
            context=context or {},
            exc_info=True
        )
    
    def log_initialization(self, component: str, details: Dict[str, Any] = None):
        """Log component initialization"""
        self.logger.info(
            "component_initialized",
            component=component,
            details=details or {}
        )
    
    def log_shutdown(self, component: str, graceful: bool = True):
        """Log component shutdown"""
        self.logger.info(
            "component_shutdown",
            component=component,
            graceful=graceful
        )
    
    def log_health_check(self, status: str, details: Dict[str, Any] = None):
        """Log health check result"""
        level = "info" if status == "healthy" else "warning"
        getattr(self.logger, level)(
            "health_check",
            status=status,
            details=details or {}
        )


# Example usage patterns for the metrics service
def get_logger_with_context(name: str, **context) -> structlog.BoundLogger:
    """Get a logger with additional context"""
    logger = setup_logging(name)
    return logger.bind(**context)


def log_with_performance(logger: structlog.BoundLogger, operation: str):
    """Context manager for logging with performance metrics"""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _log_performance():
        start_time = time.time()
        logger.debug(f"{operation}_started")
        
        try:
            yield
            duration = time.time() - start_time
            logger.info(
                f"{operation}_completed",
                duration_ms=int(duration * 1000),
                success=True
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{operation}_failed",
                duration_ms=int(duration * 1000),
                error=str(e),
                exc_info=True
            )
            raise
    
    return _log_performance()


def setup_prometheus_logging():
    """Setup Prometheus metrics for logging events"""
    global _prometheus_initialized
    
    if _prometheus_initialized:
        return  # Prevent duplicate initialization
    
    try:
        # Import prometheus_client only when needed
        import prometheus_client
        from prometheus_client import Counter, Histogram, Gauge
        
        # Define Prometheus metrics
        log_events_total = Counter(
            'log_events_total',
            'Total number of log events',
            ['level', 'logger'],
            registry=prometheus_client.REGISTRY
        )
        
        log_errors_total = Counter(
            'log_errors_total',
            'Total number of error logs',
            ['logger', 'error_type'],
            registry=prometheus_client.REGISTRY
        )
        
        # Add Prometheus processor to structlog
        def prometheus_processor(logger, method_name, event_dict):
            level = event_dict.get('level', 'info').lower()
            logger_name = event_dict.get('logger', 'unknown')
            
            # Increment log event counter
            log_events_total.labels(level=level, logger=logger_name).inc()
            
            # Track errors specifically
            if level == 'error':
                error_type = event_dict.get('error_type', 'unknown')
                log_errors_total.labels(
                    logger=logger_name,
                    error_type=error_type
                ).inc()
            
            return event_dict
        
        # Add to structlog processors
        current_processors = structlog.get_config()['processors']
        if prometheus_processor not in current_processors:
            current_processors.insert(-1, prometheus_processor)  # Insert before renderer
            structlog.configure(processors=current_processors)
        
        _prometheus_initialized = True
        
    except ImportError:
        # Prometheus client not available, skip prometheus logging
        pass
    except Exception as e:
        # Log the error but don't crash the application
        import logging
        logging.error(f"Failed to setup Prometheus logging: {e}")

# Initialize Prometheus logging if available
setup_prometheus_logging()