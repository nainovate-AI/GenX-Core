import pytest
import pytest_asyncio
import grpc
import logging
import sys
import os
from protos.core_service_pb2 import LoadModelRequest, EmbeddingRequest, QueryRequest
from protos.core_service_pb2_grpc import CoreServiceStub

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add genx-platform to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Configuration
GRPC_PORT = "50052"

@pytest_asyncio.fixture(scope="module")
async def grpc_client():
    logger.debug("Setting up grpc_client fixture")
    channel = grpc.aio.insecure_channel(f"localhost:{GRPC_PORT}")
    client = CoreServiceStub(channel)
    yield client
    logger.debug("Tearing down grpc_client fixture")
    await channel.close()

@pytest_asyncio.fixture(autouse=True)
async def load_model(grpc_client):
    logger.debug("Loading test model")
    request = LoadModelRequest(
        model_id="test-model",
        model_name="all-MiniLM-L6-v2",
        model_type="embedding",
        config={"device": "cpu"}
    )
    response = await grpc_client.LoadModel(request)
    assert response.model_id == "test-model"
    assert response.status == "completed"
    assert not response.error
    logger.debug("Test model loaded")

@pytest.mark.asyncio
async def test_generate_embedding(grpc_client):
    logger.debug("Running test_generate_embedding")
    request = EmbeddingRequest(
        job_id="test-embedding-job",
        model_id="test-model",
        texts=["This is a test sentence."],
        collection_id="test-collection",
        metadata={"source": "test"}
    )
    
    response = await grpc_client.GenerateEmbedding(request)
    assert response.job_id == "test-embedding-job"
    assert response.status == "completed"
    assert len(response.embeddings) == 384  # Expected dimension for all-MiniLM-L6-v2
    assert response.collection_id == "test-collection"
    assert not response.error
    logger.debug("test_generate_embedding completed")

@pytest.mark.asyncio
async def test_query_embedding(grpc_client):
    logger.debug("Running test_query_embedding")
    request = QueryRequest(
        query_id="test-query-job",
        model_id="test-model",
        query_text="Test sentence",
        collection_id="test-collection",
        top_k=1
    )
    
    response = await grpc_client.QueryEmbedding(request)
    assert response.query_id == "test-query-job"
    assert response.status == "completed"
    assert len(response.results) <= 1
    if response.results:
        assert response.results[0].document_id.startswith("doc_")
        assert response.results[0].score >= 0
    assert not response.error
    logger.debug("test_query_embedding completed")

@pytest.mark.asyncio
async def test_invalid_model_id(grpc_client):
    logger.debug("Running test_invalid_model_id")
    request = EmbeddingRequest(
        job_id="test-invalid-model-job",
        model_id="invalid-model",
        texts=["This is a test sentence."],
        collection_id="test-collection",
        metadata={"source": "test"}
    )
    
    response = await grpc_client.GenerateEmbedding(request)
    assert response.job_id == "test-invalid-model-job"
    assert response.status == "failed"
    assert "Model invalid-model not loaded" in response.error
    assert response.collection_id == "test-collection"
    logger.debug("test_invalid_model_id completed")