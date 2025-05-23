import pytest
import pytest_asyncio
import grpc
import logging
import sys
import os
from protos.retrieval_service_pb2 import SearchVectorsRequest
from protos.retrieval_service_pb2_grpc import RetrievalServiceStub

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add genx-platform to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Configuration
GRPC_PORT = "50055"

@pytest_asyncio.fixture(scope="module")
async def grpc_client():
    logger.debug("Setting up grpc_client fixture")
    channel = grpc.aio.insecure_channel(f"localhost:{GRPC_PORT}")
    client = RetrievalServiceStub(channel)
    yield client
    logger.debug("Tearing down grpc_client fixture")
    await channel.close()

@pytest.mark.asyncio
async def test_search_vectors(grpc_client):
    logger.debug("Running test_search_vectors")
    request = SearchVectorsRequest(
        collection_id="test-collection",
        query_vector=[0.1] * 384,
        top_k=1
    )
    response = await grpc_client.SearchVectors(request)
    assert response.status == "completed"
    assert len(response.results) <= 1
    if response.results:
        assert response.results[0].document_id == "doc_1"
        assert response.results[0].score >= 0
    assert not response.error
    logger.debug("test_search_vectors completed")