import asyncio
import grpc
import sys
import os

# Add genx-platform/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from protos.core_service_pb2 import IngestRequest, JobResponse
from protos.core_service_pb2_grpc import CoreServiceStub

async def test_ingest_file(framework: str):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = CoreServiceStub(channel)
        request = IngestRequest(
            job_id="test-file-job",
            config={
                "clientApiKey": "test-key",
                "deployId": "68188bcb4c9d2180cdecf9e2",
                "splitterConfigId": "test-splitter"
            },
            input_data_config={
                "type": "file",
                "fileNames": "sample.txt,sample.csv"  # Adjusted to available test files
            },
            loader_params={},
            framework=framework
        )
        response = await stub.Ingest(request)
        print(f"File Ingestion ({framework}): {response}")

async def test_ingest_db(framework: str):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = CoreServiceStub(channel)
        request = IngestRequest(
            job_id="test-db-job",
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
            loader_params={},
            framework=framework
        )
        response = await stub.Ingest(request)
        print(f"DB Ingestion ({framework}): {response}")

if __name__ == "__main__":
    asyncio.run(test_ingest_file("langchain"))
    asyncio.run(test_ingest_file("llamaindex"))
    asyncio.run(test_ingest_db("langchain"))
    asyncio.run(test_ingest_db("llamaindex"))