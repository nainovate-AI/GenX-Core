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
import protos.retrieval_service_pb2 as pb2
import protos.retrieval_service_pb2_grpc as pb2_grpc

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
SEARCH_TOTAL = Counter(
    "search_total", "Total search operations", ["collection_id", "status"]
)
SEARCH_DURATION = Histogram(
    "search_duration_seconds", "Search operation duration", ["collection_id"]
)

class RetrievalService(pb2_grpc.RetrievalServiceServicer):
    def __init__(self):
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        logger.info("Retrieval Service initialized")

    async def SearchVectors(self, request: pb2.SearchVectorsRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("search_vectors") as span:
            span.set_attribute("collection_id", request.collection_id)
            span.set_attribute("top_k", request.top_k)
            
            try:
                logger.info(f"Searching vectors in collection {request.collection_id}")
                with SEARCH_DURATION.labels(collection_id=request.collection_id).time():
                    results = self.qdrant.search(
                        collection_name=request.collection_id,
                        query_vector=request.query_vector,
                        limit=request.top_k
                    )
                
                response = pb2.SearchVectorsResponse(status="completed")
                for hit in results:
                    response.results.append(
                        pb2.SearchResult(
                            document_id=hit.id,
                            score=hit.score,
                            payload=hit.payload
                        )
                    )
                
                SEARCH_TOTAL.labels(collection_id=request.collection_id, status="completed").inc()
                return response
            except Exception as e:
                logger.error(f"Failed to search vectors in {request.collection_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                SEARCH_TOTAL.labels(collection_id=request.collection_id, status="failed").inc()
                return pb2.SearchVectorsResponse(
                    status="failed",
                    error=str(e)
                )

async def serve():
    server = grpc.aio.server(interceptors=[aio_server_interceptor()])
    pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalService(), server)
    server.add_insecure_port("[::]:50055")
    logger.info("Starting Retrieval gRPC server on port 50055")
    start_http_server(9095)  # Prometheus metrics
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())