import logging
import asyncio
import grpc
from qdrant_client import QdrantClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import aio_server_interceptor
from prometheus_client import start_http_server, Counter, Histogram
import protos.vector_service_pb2 as pb2
import protos.vector_service_pb2_grpc as pb2_grpc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="tempo:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Prometheus metrics
COLLECTION_CREATE_TOTAL = Counter(
    "collection_create_total", "Total collection create operations", ["status"]
)
UPSERT_TOTAL = Counter(
    "upsert_total", "Total upsert operations", ["collection_id", "status"]
)
UPSERT_DURATION = Histogram(
    "upsert_duration_seconds", "Upsert operation duration", ["collection_id"]
)

class VectorStoreService(pb2_grpc.VectorStoreServiceServicer):
    def __init__(self):
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        logger.info("Vector Store Service initialized")

    async def CreateCollection(self, request: pb2.CreateCollectionRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("create_collection") as span:
            span.set_attribute("collection_id", request.collection_id)
            span.set_attribute("vector_size", request.vector_size)
            
            try:
                logger.info(f"Creating collection {request.collection_id}")
                self.qdrant.create_collection(
                    collection_name=request.collection_id,
                    vectors_config={"size": request.vector_size, "distance": "Cosine"}
                )
                COLLECTION_CREATE_TOTAL.labels(status="completed").inc()
                return pb2.CreateCollectionResponse(
                    collection_id=request.collection_id,
                    status="completed"
                )
            except Exception as e:
                logger.error(f"Failed to create collection {request.collection_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                COLLECTION_CREATE_TOTAL.labels(status="failed").inc()
                return pb2.CreateCollectionResponse(
                    collection_id=request.collection_id,
                    status="failed",
                    error=str(e)
                )

    async def UpsertVectors(self, request: pb2.UpsertVectorsRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("upsert_vectors") as span:
            span.set_attribute("collection_id", request.collection_id)
            span.set_attribute("point_count", len(request.points))
            
            try:
                logger.info(f"Upserting vectors to collection {request.collection_id}")
                with UPSERT_DURATION.labels(collection_id=request.collection_id).time():
                    points = [
                        {
                            "id": point.id,
                            "vector": point.vector,
                            "payload": point.payload
                        }
                        for point in request.points
                    ]
                    self.qdrant.upsert(
                        collection_name=request.collection_id,
                        points=points
                    )
                UPSERT_TOTAL.labels(collection_id=request.collection_id, status="completed").inc()
                return pb2.UpsertVectorsResponse(
                    collection_id=request.collection_id,
                    status="completed"
                )
            except Exception as e:
                logger.error(f"Failed to upsert vectors to {request.collection_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                UPSERT_TOTAL.labels(collection_id=request.collection_id, status="failed").inc()
                return pb2.UpsertVectorsResponse(
                    collection_id=request.collection_id,
                    status="failed",
                    error=str(e)
                )

async def serve():
    server = grpc.aio.server(interceptors=[aio_server_interceptor()])
    pb2_grpc.add_VectorStoreServiceServicer_to_server(VectorStoreService(), server)
    server.add_insecure_port("[::]:50054")
    logger.info("Starting Vector Store gRPC server on port 50054")
    start_http_server(9094)  # Prometheus metrics
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())