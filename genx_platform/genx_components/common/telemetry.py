"""
genx_platform/genx_components/common/telemetry.py
OpenTelemetry Integration for GenX Platform
Provides distributed tracing, metrics, and logging correlation
"""
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer, GrpcInstrumentorClient
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


class GenxTelemetry:
    """Manages OpenTelemetry setup for GenX microservices"""
    
    def __init__(self, service_name: str, service_version: str, 
                 telemetry_endpoint: str = "http://localhost:4317",
                 environment: str = "development"):
        """
        Initialize telemetry for a microservice
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            telemetry_endpoint: OTLP endpoint for telemetry data
            environment: Deployment environment
        """
        self.service_name = service_name
        self.service_version = service_version
        self.telemetry_endpoint = telemetry_endpoint
        self.environment = environment
        
        # Initialize resource
        self.resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
        })
        
        # Initialize providers
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None
        
        # Metrics collectors
        self._request_counter = None
        self._request_duration = None
        self._active_requests = None
        
    def initialize(self) -> None:
        """Initialize telemetry providers and instrumentation"""
        try:
            # Set up tracing
            self._setup_tracing()
            
            # Set up metrics
            self._setup_metrics()
            
            # Set up propagation
            set_global_textmap(TraceContextTextMapPropagator())
            
            # Instrument gRPC
            GrpcInstrumentorServer().instrument()
            GrpcInstrumentorClient().instrument()
            
            logger.info(f"Telemetry initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            # Don't fail the service if telemetry fails
            
    def _setup_tracing(self) -> None:
        """Set up distributed tracing"""
        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.telemetry_endpoint,
            insecure=True  # Use TLS in production
        )
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=self.resource)
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Set as global
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(self.service_name, self.service_version)
        
    def _setup_metrics(self) -> None:
        """Set up metrics collection"""
        # Create OTLP exporter
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.telemetry_endpoint,
            insecure=True
        )
        
        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=10000  # 10 seconds
        )
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[metric_reader]
        )
        
        # Set as global
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(self.service_name, self.service_version)
        
        # Initialize common metrics
        self._request_counter = self.meter.create_counter(
            "requests_total",
            description="Total number of requests",
            unit="1"
        )
        
        self._request_duration = self.meter.create_histogram(
            "request_duration_seconds",
            description="Request duration in seconds",
            unit="s"
        )
        
        self._active_requests = self.meter.create_up_down_counter(
            "active_requests",
            description="Number of active requests",
            unit="1"
        )
    
    @contextmanager
    def trace_operation(self, operation: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing an operation
        
        Args:
            operation: Name of the operation
            attributes: Optional attributes to add to the span
        """
        if not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(operation) as span:
            # Add attributes
            if attributes:
                span.set_attributes(attributes)
            
            # Track metrics
            self._active_requests.add(1, {"operation": operation})
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                self._active_requests.add(-1, {"operation": operation})
                self._request_counter.add(1, {"operation": operation})
    
    def record_metric(self, name: str, value: float, attributes: Optional[Dict[str, Any]] = None):
        """Record a custom metric"""
        if not self.meter:
            return
            
        # This is a simplified version - in production, you'd cache these
        gauge = self.meter.create_gauge(name)
        gauge.set(value, attributes or {})
    
    def shutdown(self):
        """Shutdown telemetry providers"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()