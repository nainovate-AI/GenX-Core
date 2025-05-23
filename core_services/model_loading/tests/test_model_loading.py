import pytest
import pytest_asyncio
import grpc
import asyncio
import logging
import sys
import os


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add genx-platform to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from protos.mcp_pb2 import (
    LoadModelRequest,
    UnloadModelRequest,
    GetModelContextRequest,
    GenerateEmbeddingRequest
)
from protos.mcp_pb2_grpc import ModelContextServiceStub

# Configuration
GRPC_PORT = "50053"

@pytest_asyncio.fixture(scope="function")
async def grpc_client():
    """Create a module-scoped gRPC client."""
    logger.debug("Setting up grpc_client fixture")
    channel = grpc.aio.insecure_channel(f"localhost:{GRPC_PORT}")
    client = ModelContextServiceStub(channel)
    yield client
    logger.debug("Tearing down grpc_client fixture")
    await channel.close()

@pytest.mark.asyncio
async def test_load_embedding_model(grpc_client):
    logger.debug("Running test_load_embedding_model")
    request = LoadModelRequest(
        model_id="test-embedding-model",
        model_name="all-MiniLM-L6-v2",
        model_type="embedding",
        config={"device": "cpu"}
    )
    response = await grpc_client.LoadModel(request)
    assert response.model_id == "test-embedding-model"
    assert response.status == "completed"
    assert not response.error
    logger.debug("test_load_embedding_model completed")

@pytest.mark.asyncio
async def test_load_llm_model(grpc_client):
    logger.debug("Running test_load_llm_model")
    request = LoadModelRequest(
        model_id="test-llm-model",
        model_name="distilbert-base-uncased",
        model_type="llm",
        config={"device": "cpu"}
    )
    response = await grpc_client.LoadModel(request)
    assert response.model_id == "test-llm-model"
    assert response.status == "completed"
    assert not response.error
    logger.debug("test_load_llm_model completed")

@pytest.mark.asyncio
async def test_get_model_context(grpc_client):
    logger.debug("Running test_get_model_context")
    request = GetModelContextRequest(model_id="test-embedding-model")
    response = await grpc_client.GetModelContext(request)
    assert response.model_id == "test-embedding-model"
    assert response.model_name == "all-MiniLM-L6-v2"
    assert response.model_type == "embedding"
    assert response.status == "loaded"
    assert response.config == {"device": "cpu"}
    assert not response.error
    logger.debug("test_get_model_context completed")

@pytest.mark.asyncio
async def test_generate_embedding(grpc_client):
    logger.debug("Running test_generate_embedding")
    request = GenerateEmbeddingRequest(
        model_id="test-embedding-model",
        texts=["This is a test sentence."]
    )
    response = await grpc_client.GenerateEmbedding(request)
    assert response.model_id == "test-embedding-model"
    assert response.status == "completed"
    assert len(response.embeddings) == 384  # Expected dimension for all-MiniLM-L6-v2
    assert all(isinstance(x, float) for x in response.embeddings)
    assert not response.error
    logger.debug("test_generate_embedding completed")

@pytest.mark.asyncio
async def test_unload_model(grpc_client):
    logger.debug("Running test_unload_model")
    request = UnloadModelRequest(model_id="test-embedding-model")
    response = await grpc_client.UnloadModel(request)
    assert response.model_id == "test-embedding-model"
    assert response.status == "completed"
    assert not response.error
    logger.debug("test_unload_model completed")

@pytest.mark.asyncio
async def test_load_invalid_model(grpc_client):
    logger.debug("Running test_load_invalid_model")
    request = LoadModelRequest(
        model_id="invalid-model",
        model_name="nonexistent-model",
        model_type="embedding",
        config={"device": "cpu"}
    )
    response = await grpc_client.LoadModel(request)
    assert response.model_id == "invalid-model"
    assert response.status == "failed"
    assert "is not a valid model identifier" in response.error
    logger.debug("test_load_invalid_model completed")

@pytest.mark.asyncio
async def test_get_nonexistent_model_context(grpc_client):
    logger.debug("Running test_get_nonexistent_model_context")
    request = GetModelContextRequest(model_id="nonexistent-model")
    response = await grpc_client.GetModelContext(request)
    assert response.model_id == "nonexistent-model"
    assert response.status == "not_loaded"
    assert response.error
    logger.debug("test_get_nonexistent_model_context completed")

@pytest.mark.asyncio
async def test_generate_embedding_invalid_model(grpc_client):
    logger.debug("Running test_generate_embedding_invalid_model")
    request = GenerateEmbeddingRequest(
        model_id="nonexistent-model",
        texts=["This is a test sentence."]
    )
    response = await grpc_client.GenerateEmbedding(request)
    assert response.model_id == "nonexistent-model"
    assert response.status == "failed"
    assert response.error == "Model nonexistent-model not loaded"
    logger.debug("test_generate_embedding_invalid_model completed")

# @pytest.mark.asyncio
# async def test_generate_embedding_non_embedding_model(grpc_client):
#     logger.debug("Running test_generate_embedding_non_embedding_model")
#     request = GenerateEmbeddingRequest(
#         model_id="test-llm-model",
#         texts=["This is a test sentence."]
#     )
#     response = await grpc_client.GenerateEmbedding(request)
#     assert response.model_id == "test-llm-model"
#     assert response.status == "failed"
#     assert response.error == "Model test-llm-model is not an embedding model"
#     logger.debug("test_generate_embedding_non_embedding_model completed")

@pytest.mark.asyncio
async def test_unload_nonexistent_model(grpc_client):
    logger.debug("Running test_unload_nonexistent_model")
    request = UnloadModelRequest(model_id="nonexistent-model")
    response = await grpc_client.UnloadModel(request)
    assert response.model_id == "nonexistent-model"
    assert response.status == "failed"
    assert response.error == "Model nonexistent-model not found"
    logger.debug("test_unload_nonexistent_model completed")