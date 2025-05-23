import pytest
import pytest_asyncio
import grpc
import logging
import sys
import os
from protos.vector_service_pb2 import CreateCollectionRequest, UpsertVectorsRequest, VectorPoint
from protos.vector_service_pb2_grpc import VectorStoreServiceStub

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add genx-platform to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Configuration
GRPC_PORT = "50054"

@pytest_asyncio.fixture(scope="module")
async def grpc_client():
    logger.debug("Setting up grpc_client fixture")
    channel = grpc.aio.insecure_channel(f"localhost:{GRPC_PORT}")
    client = VectorStoreServiceStub(channel)
    yield client
    logger.debug("Tearing down grpc_client fixture")
    await channel.close()

@pytest.mark.asyncio
async def test_create_collection(grpc_client):
    logger.debug("Running test_create_collection")
    request = CreateCollectionRequest(collection_id="test-collection", vector_size=384)
    response = await grpc_client.CreateCollection(request)
    assert response.collection_id == "test-collection"
    assert response.status == "completed"
    assert not response.error
    logger.debug("test_create_collection completed")

@pytest.mark.asyncio
async def test_upsert_vectors(grpc_client):
    logger.debug("Running test_upsert_vectors")
    request = UpsertVectorsRequest(
        collection_id="test-collection",
        points=[
            VectorPoint(
                id="doc_1",
                vector=[0.1] * 384,
                payload={"source": "test"}
            )
        ]
    )
    response = await grpc_client.UpsertVectors(request)
    assert response.collection_id == "test-collection"
    assert response.status == "completed"
    assert not response.error
    logger.debug("test_upsert_vectors completed")