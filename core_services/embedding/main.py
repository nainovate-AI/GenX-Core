import logging
import asyncio
import grpc
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import aio_server_interceptor
from prometheus_client import start_http_server, Counter, Histogram
import protos.core_service_pb2 as pb2
import protos.core_service_pb2_grpc as pb2_grpc
import protos.mcp_pb2 as mcp_pb2
import protos.mcp_pb2_grpc as mcp_pb2_grpc
import protos.vector_service_pb2 as vector_pb2
import protos.vector_service_pb2_grpc as vector_pb2_grpc
import protos.retrieval_service_pb2 as retrieval_pb2
import protos.retrieval_service_pb2_grpc as retrieval_pb2_grpc

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
EMBEDDING_TOTAL = Counter(
    "embedding_total", "Total embedding operations", ["model_id", "status"]
)
EMBEDDING_DURATION = Histogram(
    "embedding_duration_seconds", "Embedding operation duration", ["model_id"]
)

class CoreService(pb2_grpc.CoreServiceServicer):
    def __init__(self):
        self.model_channel = grpc.aio.insecure_channel("model_loading:50053")
        self.model_stub = mcp_pb2_grpc.ModelContextServiceStub(self.model_channel)
        self.vector_channel = grpc.aio.insecure_channel("vector_store:50054")
        self.vector_stub = vector_pb2_grpc.VectorStoreServiceStub(self.vector_channel)
        self.retrieval_channel = grpc.aio.insecure_channel("retrieval:50055")
        self.retrieval_stub = retrieval_pb2_grpc.RetrievalServiceStub(self.retrieval_channel)
        logger.info("Embedding service initialized")

    async def LoadModel(self, request: pb2.LoadModelRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("load_model") as span:
            span.set_attribute("model_id", request.model_id)
            span.set_attribute("model_name", request.model_name)
            span.set_attribute("model_type", request.model_type)
            
            model_request = mcp_pb2.LoadModelRequest(
                model_id=request.model_id,
                model_name=request.model_name,
                model_type=request.model_type,
                config=request.config
            )
            response = await self.model_stub.LoadModel(model_request)
            
            return pb2.LoadModelResponse(
                model_id=response.model_id,
                status=response.status,
                error=response.error
            )

    async def GenerateEmbedding(self, request: pb2.EmbeddingRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("generate_embedding") as span:
            span.set_attribute("job_id", request.job_id)
            span.set_attribute("model_id", request.model_id)
            span.set_attribute("collection_id", request.collection_id)
            
            try:
                # Check if model is loaded
                model_context = await self.model_stub.GetModelContext(
                    mcp_pb2.GetModelContextRequest(model_id=request.model_id)
                )
                if model_context.status != "loaded":
                    error_msg = f"Model {request.model_id} not loaded"
                    logger.error(error_msg)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                    EMBEDDING_TOTAL.labels(model_id=request.model_id, status="failed").inc()
                    return pb2.EmbeddingResponse(
                        job_id=request.job_id,
                        status="failed",
                        error=error_msg,
                        collection_id=request.collection_id
                    )
                
                # Create collection if needed
                await self.vector_stub.CreateCollection(
                    vector_pb2.CreateCollectionRequest(
                        collection_id=request.collection_id,
                        vector_size=384  # Hardcoded for all-MiniLM-L6-v2
                    )
                )
                
                logger.info(f"Generating embeddings for job {request.job_id}")
                with EMBEDDING_DURATION.labels(model_id=request.model_id).time():
                    embedding_response = await self.model_stub.GenerateEmbedding(
                        mcp_pb2.GenerateEmbeddingRequest(model_id=request.model_id, texts=request.texts)
                    )
                    if embedding_response.status != "completed":
                        raise Exception(embedding_response.error)
                    embeddings = embedding_response.embeddings
                
                # Store in Vector Store Service
                points = [
                    vector_pb2.VectorPoint(
                        id=f"doc_{i}",
                        vector=embeddings[i * 384:(i + 1) * 384],
                        payload=request.metadata
                    )
                    for i in range(len(request.texts))
                ]
                upsert_response = await self.vector_stub.UpsertVectors(
                    vector_pb2.UpsertVectorsRequest(collection_id=request.collection_id, points=points)
                )
                if upsert_response.status != "completed":
                    raise Exception(upsert_response.error)
                
                EMBEDDING_TOTAL.labels(model_id=request.model_id, status="completed").inc()
                return pb2.EmbeddingResponse(
                    job_id=request.job_id,
                    status="completed",
                    embeddings=embeddings,
                    collection_id=request.collection_id
                )
            except Exception as e:
                logger.error(f"Error in job {request.job_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                EMBEDDING_TOTAL.labels(model_id=request.model_id, status="failed").inc()
                return pb2.EmbeddingResponse(
                    job_id=request.job_id,
                    status="failed",
                    error=str(e),
                    collection_id=request.collection_id
                )

    async def QueryEmbedding(self, request: pb2.QueryRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("query_embedding") as span:
            span.set_attribute("query_id", request.query_id)
            span.set_attribute("model_id", request.model_id)
            span.set_attribute("collection_id", request.collection_id)
            
            try:
                # Check if model is loaded
                model_context = await self.model_stub.GetModelContext(
                    mcp_pb2.GetModelContextRequest(model_id=request.model_id)
                )
                if model_context.status != "loaded":
                    error_msg = f"Model {request.model_id} not loaded"
                    logger.error(error_msg)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                    EMBEDDING_TOTAL.labels(model_id=request.model_id, status="failed").inc()
                    return pb2.QueryResponse(
                        query_id=request.query_id,
                        status="failed",
                        error=error_msg
                    )
                
                logger.info(f"Querying embeddings for query {request.query_id}")
                with EMBEDDING_DURATION.labels(model_id=request.model_id).time():
                    embedding_response = await self.model_stub.GenerateEmbedding(
                        mcp_pb2.GenerateEmbeddingRequest(model_id=request.model_id, texts=[request.query_text])
                    )
                    if embedding_response.status != "completed":
                        raise Exception(embedding_response.error)
                    query_vector = embedding_response.embeddings[:384]  # Single text
                    
                    search_response = await self.retrieval_stub.SearchVectors(
                        retrieval_pb2.SearchVectorsRequest(
                            collection_id=request.collection_id,
                            query_vector=query_vector,
                            top_k=request.top_k
                        )
                    )
                    if search_response.status != "completed":
                        raise Exception(search_response.error)
                
                response = pb2.QueryResponse(
                    query_id=request.query_id,
                    status="completed"
                )
                for result in search_response.results:
                    response.results.append(
                        pb2.Result(
                            document_id=result.document_id,
                            score=result.score,
                            metadata=result.payload
                        )
                    )
                
                EMBEDDING_TOTAL.labels(model_id=request.model_id, status="completed").inc()
                return response
            except Exception as e:
                logger.error(f"Error in query {request.query_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                EMBEDDING_TOTAL.labels(model_id=request.model_id, status="failed").inc()
                return pb2.QueryResponse(
                    query_id=request.query_id,
                    status="failed",
                    error=str(e)
                )

async def serve():
    server = grpc.aio.server(interceptors=[aio_server_interceptor()])
    pb2_grpc.add_CoreServiceServicer_to_server(CoreService(), server)
    server.add_insecure_port("[::]:50052")
    logger.info("Starting Embedding gRPC server on port 50052")
    start_http_server(9092)  # Prometheus metrics
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())