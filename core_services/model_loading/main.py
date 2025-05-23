import logging
import asyncio
import grpc
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import aio_server_interceptor
from prometheus_client import start_http_server, Counter, Histogram
import protos.mcp_pb2 as pb2
import protos.mcp_pb2_grpc as pb2_grpc

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
MODEL_LOAD_TOTAL = Counter(
    "model_load_total", "Total model load operations", ["model_type", "status"]
)
MODEL_LOAD_DURATION = Histogram(
    "model_load_duration_seconds", "Model load duration", ["model_type"]
)
EMBEDDING_GENERATION_TOTAL = Counter(
    "embedding_generation_total", "Total embedding generation operations", ["model_id", "status"]
)
EMBEDDING_GENERATION_DURATION = Histogram(
    "embedding_generation_duration_seconds", "Embedding generation duration", ["model_id"]
)

class ModelRegistry:
    def __init__(self):
        self.models = {}  # {model_id: (model, tokenizer, model_name, model_type, config)}
        self.lock = asyncio.Lock()

    async def load_model(self, model_id: str, model_name: str, model_type: str, config: dict):
        async with self.lock:
            if model_id in self.models:
                logger.info(f"Model {model_id} already loaded")
                return True, None
            
            try:
                with MODEL_LOAD_DURATION.labels(model_type=model_type).time():
                    if model_type == "embedding":
                        model = SentenceTransformer(model_name)
                        tokenizer = None
                    elif model_type == "llm":
                        device = config.get("device", "cpu")
                        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    else:
                        return False, f"Unsupported model type: {model_type}"
                
                self.models[model_id] = (model, tokenizer, model_name, model_type, config)
                logger.info(f"Loaded model {model_id} ({model_name}, {model_type})")
                MODEL_LOAD_TOTAL.labels(model_type=model_type, status="completed").inc()
                return True, None
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}")
                MODEL_LOAD_TOTAL.labels(model_type=model_type, status="failed").inc()
                return False, str(e)

    async def unload_model(self, model_id: str):
        async with self.lock:
            if model_id not in self.models:
                logger.info(f"Model {model_id} not loaded")
                return False, f"Model {model_id} not found"
            
            try:
                model, tokenizer, _, _, _ = self.models.pop(model_id)
                del model
                if tokenizer:
                    del tokenizer
                logger.info(f"Unloaded model {model_id}")
                return True, None
            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {str(e)}")
                return False, str(e)

    async def get_model_context(self, model_id: str):
        async with self.lock:
            if model_id not in self.models:
                logger.info(f"Model {model_id} not loaded")
                return None, None, None, None, "not_loaded"
            model, tokenizer, model_name, model_type, config = self.models[model_id]
            return model_name, model_type, config, "loaded", None

    async def generate_embedding(self, model_id: str, texts: list):
        async with self.lock:
            if model_id not in self.models:
                logger.info(f"Model {model_id} not loaded")
                return False, None, f"Model {model_id} not loaded"
            
            model, _, _, model_type, _ = self.models[model_id]
            if model_type != "embedding":
                return False, None, f"Model {model_id} is not an embedding model"
            
            try:
                with EMBEDDING_GENERATION_DURATION.labels(model_id=model_id).time():
                    embeddings = model.encode(texts).flatten().tolist()
                EMBEDDING_GENERATION_TOTAL.labels(model_id=model_id, status="completed").inc()
                logger.info(f"Generated embeddings for texts using model {model_id}")
                return True, embeddings, None
            except Exception as e:
                logger.error(f"Failed to generate embeddings for model {model_id}: {str(e)}")
                EMBEDDING_GENERATION_TOTAL.labels(model_id=model_id, status="failed").inc()
                return False, None, str(e)

class ModelContextService(pb2_grpc.ModelContextServiceServicer):
    def __init__(self):
        self.registry = ModelRegistry()
        logger.info("Model Context Service initialized")

    async def LoadModel(self, request: pb2.LoadModelRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("load_model") as span:
            span.set_attribute("model_id", request.model_id)
            span.set_attribute("model_name", request.model_name)
            span.set_attribute("model_type", request.model_type)
            
            success, error = await self.registry.load_model(
                request.model_id, request.model_name, request.model_type, request.config
            )
            
            if success:
                return pb2.LoadModelResponse(
                    model_id=request.model_id,
                    status="completed"
                )
            else:
                span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                return pb2.LoadModelResponse(
                    model_id=request.model_id,
                    status="failed",
                    error=error
                )

    async def UnloadModel(self, request: pb2.UnloadModelRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("unload_model") as span:
            span.set_attribute("model_id", request.model_id)
            
            success, error = await self.registry.unload_model(request.model_id)
            
            if success:
                return pb2.UnloadModelResponse(
                    model_id=request.model_id,
                    status="completed"
                )
            else:
                span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                return pb2.UnloadModelResponse(
                    model_id=request.model_id,
                    status="failed",
                    error=error
                )

    async def GetModelContext(self, request: pb2.GetModelContextRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("get_model_context") as span:
            span.set_attribute("model_id", request.model_id)
            
            model_name, model_type, config, status, error = await self.registry.get_model_context(request.model_id)
            
            if error:
                span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                return pb2.GetModelContextResponse(
                    model_id=request.model_id,
                    status="not_loaded",
                    error=error
                )
            logger.info(f"Model context for {request.model_id}: {model_name}, {model_type}, {status}, {config}")
            return pb2.GetModelContextResponse(
                model_id=request.model_id,
                model_name=model_name,
                model_type=model_type,
                config=config,
                status=status
            )

    async def GenerateEmbedding(self, request: pb2.GenerateEmbeddingRequest, context: grpc.aio.ServicerContext):
        with tracer.start_as_current_span("generate_embedding") as span:
            span.set_attribute("model_id", request.model_id)
            span.set_attribute("text_count", len(request.texts))
            
            success, embeddings, error = await self.registry.generate_embedding(request.model_id, request.texts)
            
            if success:
                return pb2.GenerateEmbeddingResponse(
                    model_id=request.model_id,
                    status="completed",
                    embeddings=embeddings
                )
            else:
                span.set_status(trace.Status(trace.StatusCode.ERROR, error))
                return pb2.GenerateEmbeddingResponse(
                    model_id=request.model_id,
                    status="failed",
                    error=error
                )

async def serve():
    server = grpc.aio.server(interceptors=[aio_server_interceptor()])
    pb2_grpc.add_ModelContextServiceServicer_to_server(ModelContextService(), server)
    server.add_insecure_port("[::]:50053")
    logger.info("Starting Model Context gRPC server on port 50053")
    start_http_server(9093)  # Prometheus metrics
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())