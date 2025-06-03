from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
import logging

class GenxTelemetry:
    def __init__(self, service_name: str = "genx-llm-loader"):
        self.resource = Resource(attributes={"service.name": service_name})
        self._setup_metrics()
        self._setup_tracing()
        self._setup_logging()

    def _setup_metrics(self):
        metrics.set_meter_provider(
            MeterProvider(
                resource=self.resource,
                metric_readers=[PrometheusMetricReader()],
            )
        )
        self.meter = metrics.get_meter("genx-llm-loader")
        self.model_load_counter = self.meter.create_counter(
            name="model_load_total", description="Total model loads"
        )

    def _setup_tracing(self):
        trace.set_tracer_provider(TracerProvider(resource=self.resource))
        otlp_exporter = OTLPSpanExporter(endpoint="tempo:4317", insecure=True)
        trace.get_tracer_provider().add_span_processor(
            trace.BatchSpanProcessor(otlp_exporter)
        )
        self.tracer = trace.get_tracer("genx-llm-loader")

    def _setup_logging(self):
        # Configure logging to send to Loki (via a logging handler, e.g., Grafana Loki)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("genx-llm-loader")

    def record_model_load(self, model_id: str, controller: str):
        self.model_load_counter.add(1, {"model_id": model_id, "controller": controller})

    def start_span(self, name: str):
        return self.tracer.start_as_current_span(name)