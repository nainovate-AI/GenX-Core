import pytest
import pytest_asyncio
import grpc
import asyncio
from pathlib import Path
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add genx-platform to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from protos.core_service_pb2 import IngestRequest, JobResponse
from protos.core_service_pb2_grpc import CoreServiceStub
from motor.motor_asyncio import AsyncIOMotorClient

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
GRPC_PORT = "50051"

@pytest_asyncio.fixture(scope="function")
async def grpc_client():
    logger.debug("Setting up grpc_client fixture")
    try:
        channel = grpc.aio.insecure_channel(f"localhost:{GRPC_PORT}")
        client = CoreServiceStub(channel)
        yield client
        logger.debug("Tearing down grpc_client fixture")
        await channel.close()
    except Exception as e:
        logger.error(f"Failed to set up gRPC client: {e}")
        raise

@pytest_asyncio.fixture(scope="function")
async def mongo_client():
    logger.debug("Setting up mongo_client fixture")
    client = AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        logger.debug("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    yield client
    logger.debug("Tearing down mongo_client fixture")

@pytest_asyncio.fixture(autouse=True)
async def setup_mongo(mongo_client):
    logger.debug("Setting up mongo data")
    try:
        db = mongo_client["applicationDB"]
        await db.clientApiKeys.delete_many({})
        await db.clientApiKeys.insert_one({
            "clientApiKey": "test-key",
            "orgId": "test-org"
        })
        
        org_db = mongo_client["test-org"]
        await org_db.splitterDbConfig.delete_many({})
        await org_db.splitterDbConfig.insert_one({
            "_id": "test-splitter",
            "params": {
                "chunkSize": 512,
                "chunkOverlap": 50
            }
        })
        
        test_db = mongo_client["test_db"]
        await test_db.test_collection.delete_many({})
        await test_db.test_collection.insert_many([
            {"content": "Test record 1"},
            {"content": "Test record 2"}
        ])
        logger.debug("Mongo data setup complete")
    except Exception as e:
        logger.error(f"Failed to set up MongoDB data: {e}")
        raise

@pytest.mark.asyncio
async def test_file_ingestion_langchain(grpc_client):
    logger.debug("Running test_file_ingestion_langchain")
    request = IngestRequest(
        job_id="test-file-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "sample.txt"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-file-job"
    assert response.status == "completed"
    assert response.activity_type == "ingest"
    assert not response.error
    logger.debug("test_file_ingestion_langchain completed")

@pytest.mark.asyncio
async def test_file_ingestion_llamaindex(grpc_client):
    logger.debug("Running test_file_ingestion_llamaindex")
    request = IngestRequest(
        job_id="test-file-job",
        framework="llamaindex",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "sample.csv"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-file-job"
    assert response.status == "completed"
    assert response.activity_type == "ingest"
    assert not response.error
    logger.debug("test_file_ingestion_llamaindex completed")

@pytest.mark.asyncio
async def test_db_ingestion_langchain(grpc_client):
    logger.debug("Running test_db_ingestion_langchain")
    request = IngestRequest(
        job_id="test-db-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "db",
            "dbName": "test_db",
            "collectionName": "test_collection"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-db-job"
    assert response.status == "completed"
    assert response.activity_type == "ingest"
    assert not response.error
    logger.debug("test_db_ingestion_langchain completed")

@pytest.mark.asyncio
async def test_db_ingestion_llamaindex(grpc_client):
    logger.debug("Running test_db_ingestion_llamaindex")
    request = IngestRequest(
        job_id="test-db-job",
        framework="llamaindex",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "db",
            "dbName": "test_db",
            "collectionName": "test_collection"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-db-job"
    assert response.status == "completed"
    assert response.activity_type == "ingest"
    assert not response.error
    logger.debug("test_db_ingestion_llamaindex completed")

@pytest.mark.asyncio
async def test_invalid_file_type(grpc_client):
    logger.debug("Running test_invalid_file_type")
    request = IngestRequest(
        job_id="test-invalid-file-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "invalid.docx"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-invalid-file-job"
    assert response.status == "failed"
    assert response.activity_type == "ingest"
    assert "Unsupported file type: .docx" in response.error
    logger.debug("test_invalid_file_type completed")

@pytest.mark.asyncio
async def test_missing_file(grpc_client):
    logger.debug("Running test_missing_file")
    request = IngestRequest(
        job_id="test-missing-file-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "nonexistent.txt"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-missing-file-job"
    assert response.status == "failed"
    assert response.activity_type == "ingest"
    assert "No such file or directory" in response.error or "not found" in response.error
    logger.debug("test_missing_file completed")

@pytest.mark.asyncio
async def test_missing_collection(grpc_client):
    logger.debug("Running test_missing_collection")
    request = IngestRequest(
        job_id="test-missing-collection-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "db",
            "dbName": "test_db",
            "collectionName": "nonexistent_collection"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-missing-collection-job"
    assert response.status == "completed"  # Empty collection is valid
    assert response.activity_type == "ingest"
    assert not response.error
    logger.debug("test_missing_collection completed")

@pytest.mark.asyncio
async def test_invalid_api_key(grpc_client):
    logger.debug("Running test_invalid_api_key")
    request = IngestRequest(
        job_id="test-invalid-api-job",
        framework="langchain",
        config={
            "clientApiKey": "invalid-key",
            "deployId": "test-deploy",
            "splitterConfigId": "test-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "sample.txt"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-invalid-api-job"
    assert response.status == "failed"
    assert response.activity_type == "ingest"
    assert "Invalid API key" in response.error
    logger.debug("test_invalid_api_key completed")

@pytest.mark.asyncio
async def test_missing_splitter_config(grpc_client):
    logger.debug("Running test_missing_splitter_config")
    request = IngestRequest(
        job_id="test-missing-splitter-job",
        framework="langchain",
        config={
            "clientApiKey": "test-key",
            "deployId": "test-deploy",
            "splitterConfigId": "nonexistent-splitter"
        },
        input_data_config={
            "type": "file",
            "fileNames": "sample.txt"
        },
        loader_params={}
    )
    
    response = await grpc_client.Ingest(request)
    assert response.job_id == "test-missing-splitter-job"
    assert response.status == "failed"
    assert response.activity_type == "ingest"
    assert "Splitter configuration not found" in response.error
    logger.debug("test_missing_splitter_config completed")