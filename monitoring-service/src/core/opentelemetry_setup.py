# monitoring-service/src/core/opentelemetry_setup.py
"""OpenTelemetry configuration and setup"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace, metrics, baggage
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace import Status, StatusCode
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.sdk.metrics import MeterProvider, Counter, Histogram, UpDownCounter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics.view import View, DropAggregation, ExplicitBucketHistogramAggregation

from core.config import settings

logger = logging.getLogger(__name__)

# Global tracer and meter instances
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None
_initialized: bool = False

class TelemetryManager:
    """Manages OpenTelemetry configuration and instrumentation"""
    
    def __init__(self):
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.resource: Optional[Resource] = None
        
    def create_resource(self) -> Resource:
        """Create resource with service information"""
        return Resource.create({
            SERVICE_NAME: settings.SERVICE_NAME,
            SERVICE_VERSION: settings.SERVICE_VERSION,
            DEPLOYMENT_ENVIRONMENT: settings.ENVIRONMENT,
            "service.namespace": "monitoring",
            "service.instance.id": f"{settings.SERVICE_NAME}-001",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "cloud.provider": "docker",
            "cloud.platform": "docker-compose"
        })
    
    def setup_tracing(self):
        """Configure tracing with OTLP exporter"""
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=self.resource,
            sampler=ParentBased(
                root=TraceIdRatioBased(settings.TRACE_SAMPLING_RATE)
            ) if settings.ENABLE_TRACE_SAMPLING else None
        )
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=settings.OTEL_EXPORTER_OTLP_INSECURE,
            headers=()
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            max_export_timeout_millis=30000,
            export_timeout_millis=30000
        )
        self.tracer_provider.add_span_processor(span_processor)
        
        # Add console exporter for debugging in development
        if settings.DEBUG:
            console_processor = BatchSpanProcessor(ConsoleSpanExporter())
            self.tracer_provider.add_span_processor(console_processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Configure propagators for distributed tracing
        set_global_textmap(
            CompositePropagator([
                B3MultiFormat(),
                JaegerPropagator()
            ])
        )
        
        logger.info("Tracing configured with OTLP exporter")
    
    def setup_metrics(self):
        """Configure metrics with OTLP exporter"""
        # Configure metric views for better control
        views = [
            # Drop some high-cardinality metrics
            View(
                instrument_name="http.server.request.size",
                aggregation=DropAggregation()
            ),
            # Configure histogram buckets for latency metrics
            View(
                instrument_name="http.server.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
                )
            )
        ]
        
        # Create metric reader with OTLP exporter
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=settings.OTEL_EXPORTER_OTLP_INSECURE,
                headers=()
            ),
            export_interval_millis=10000  # Export every 10 seconds
        )
        
        # Add console exporter for debugging
        readers = [metric_reader]
        if settings.DEBUG:
            console_reader = PeriodicExportingMetricReader(
                exporter=ConsoleMetricExporter(),
                export_interval_millis=60000  # Every minute
            )
            readers.append(console_reader)
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=readers,
            views=views
        )
        
        # Set as global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        logger.info("Metrics configured with OTLP exporter")
    
    def setup_instrumentation(self):
        """Setup automatic instrumentation for libraries"""
        # FastAPI instrumentation
        FastAPIInstrumentor.instrument(
            excluded_urls="/health,/metrics,/api/v1/health/*"
        )
        
        # HTTP client instrumentation
        RequestsInstrumentor().instrument(
            excluded_urls="localhost:4317,localhost:4318"  # Exclude OTLP endpoints
        )
        HTTPXClientInstrumentor().instrument()
        
        # gRPC instrumentation
        GrpcInstrumentorClient().instrument()
        GrpcInstrumentorServer().instrument()
        
        # Async instrumentation
        AsyncioInstrumentor().instrument()
        
        # Logging instrumentation
        LoggingInstrumentor().instrument(
            set_logging_format=True,
            log_level=logging.INFO
        )
        
        # System metrics instrumentation
        if settings.ENABLE_METRICS_ENDPOINT:
            SystemMetricsInstrumentor().instrument()
        
        logger.info("Auto-instrumentation configured")
    
    def initialize(self):
        """Initialize OpenTelemetry with all components"""
        # Create resource
        self.resource = self.create_resource()
        
        # Setup components
        self.setup_tracing()
        self.setup_metrics()
        self.setup_instrumentation()
        
        # Create global instances
        global _tracer, _meter
        _tracer = trace.get_tracer(
            settings.SERVICE_NAME,
            settings.SERVICE_VERSION
        )
        _meter = metrics.get_meter(
            settings.SERVICE_NAME,
            settings.SERVICE_VERSION
        )
        
        logger.info(f"OpenTelemetry initialized for {settings.SERVICE_NAME}")
    
    def shutdown(self):
        """Shutdown OpenTelemetry providers"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")

# Global telemetry manager instance
telemetry_manager = TelemetryManager()

def setup_opentelemetry(service_name: Optional[str] = None):
    """Setup OpenTelemetry for the service"""
    global _initialized
    
    if _initialized:
        logger.warning("OpenTelemetry already initialized")
        return
    
    if service_name:
        settings.SERVICE_NAME = service_name
    
    try:
        telemetry_manager.initialize()
        _initialized = True
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        raise

def get_tracer(name: Optional[str] = None) -> trace.Tracer:
    """Get a tracer instance"""
    if not _initialized:
        setup_opentelemetry()
    
    return trace.get_tracer(
        name or settings.SERVICE_NAME,
        settings.SERVICE_VERSION
    )

def get_meter(name: Optional[str] = None) -> metrics.Meter:
    """Get a meter instance"""
    if not _initialized:
        setup_opentelemetry()
    
    return metrics.get_meter(
        name or settings.SERVICE_NAME,
        settings.SERVICE_VERSION
    )

# Context managers for manual instrumentation
@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
):
    """Context manager for creating a span"""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes or {}
    ) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

# Decorator for tracing functions
def trace_function(
    name: Optional[str] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
):
    """Decorator to trace function execution"""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        async def async_wrapper(*args, **kwargs):
            with trace_span(span_name, kind=kind) as span:
                result = await func(*args, **kwargs)
                return result
        
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, kind=kind) as span:
                result = func(*args, **kwargs)
                return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Custom metrics
class MetricsCollector:
    """Helper class for collecting custom metrics"""
    
    def __init__(self):
        self.meter = get_meter()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup custom metrics"""
        # Request counter
        self.request_counter = self.meter.create_counter(
            name="monitoring_service_requests_total",
            description="Total number of requests to monitoring service",
            unit="1"
        )
        
        # Request duration histogram
        self.request_duration = self.meter.create_histogram(
            name="monitoring_service_request_duration",
            description="Request duration in seconds",
            unit="s"
        )
        
        # Active requests gauge
        self.active_requests = self.meter.create_up_down_counter(
            name="monitoring_service_active_requests",
            description="Number of active requests",
            unit="1"
        )
        
        # Component health gauge
        self.component_health = self.meter.create_up_down_counter(
            name="monitoring_service_component_health",
            description="Health status of monitoring components (1=healthy, 0=unhealthy)",
            unit="1"
        )
        
        # Query performance metrics
        self.query_duration = self.meter.create_histogram(
            name="monitoring_service_query_duration",
            description="Duration of queries to monitoring backends",
            unit="s"
        )
        
        # Error counter
        self.error_counter = self.meter.create_counter(
            name="monitoring_service_errors_total",
            description="Total number of errors",
            unit="1"
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        self.request_counter.add(1, labels)
        self.request_duration.record(duration, labels)
    
    def record_active_request(self, delta: int):
        """Record active request count change"""
        self.active_requests.add(delta)
    
    def record_component_health(self, component: str, healthy: bool):
        """Record component health status"""
        self.component_health.add(
            1 if healthy else -1,
            {"component": component}
        )
    
    def record_query(self, backend: str, operation: str, duration: float, success: bool):
        """Record query metrics"""
        labels = {
            "backend": backend,
            "operation": operation,
            "success": str(success)
        }
        self.query_duration.record(duration, labels)
        if not success:
            self.error_counter.add(1, {"type": "query_error", "backend": backend})
    
    def record_error(self, error_type: str, service: str):
        """Record error metrics"""
        self.error_counter.add(1, {"type": error_type, "service": service})

# Global metrics collector
metrics_collector = MetricsCollector() if _initialized else None

# Shutdown hook
import atexit
atexit.register(lambda: telemetry_manager.shutdown() if _initialized else None)