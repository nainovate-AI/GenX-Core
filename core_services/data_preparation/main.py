import asyncio
from typing import Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient
import grpc
import PyPDF2
import pandas as pd
import openpyxl
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timezone
import os
from prometheus_client import Counter, start_http_server
import requests
import json

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_core.documents import Document

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LlamaDocument

# Import gRPC stubs
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
from protos.core_service_pb2 import IngestRequest, JobResponse
from protos.core_service_pb2_grpc import CoreServiceServicer, add_CoreServiceServicer_to_server

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Loki logging handler
class LokiHandler(logging.Handler):
    def __init__(self, loki_url):
        super().__init__()
        self.loki_url = loki_url

    def emit(self, record):
        try:
            log_entry = self.format(record)
            payload = {
                "streams": [{
                    "stream": {"app": "data_preparation"},
                    "values": [[str(int(datetime.now().timestamp() * 1e9)), log_entry]]
                }]
            }
            requests.post(f"{self.loki_url}/loki/api/v1/push", json=payload)
        except Exception as e:
            print(f"Failed to send log to Loki: {e}")

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
LOKI_ENDPOINT = os.getenv("LOKI_ENDPOINT", "http://localhost:3100")
SUPPORTED_FILE_TYPES = {"pdf", "txt", "csv", "xlsx"}
GRPC_PORT = "50051"
PROMETHEUS_PORT = 9090

# Prometheus metrics
ingestion_count = Counter('ingestion_total', 'Total number of ingestion jobs', ['framework', 'status'])
error_count = Counter('ingestion_errors_total', 'Total number of ingestion errors', ['framework'])

# Setup Loki logging
loki_handler = LokiHandler(LOKI_ENDPOINT)
loki_handler.setLevel(logging.DEBUG)
logger.addHandler(loki_handler)

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="tempo:4317", insecure=True)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# MongoDB client (initialized in serve)
mongo_client = None

# Abstract Ingestor interface
class Ingestor(ABC):
    @abstractmethod
    async def process_file(self, file_path: Path, file_type: str, params: Dict[str, Any]) -> List[Dict]:
        pass

    @abstractmethod
    async def process_db(self, db_config: Dict[str, Any]) -> List[Dict]:
        pass

# LangChain Ingestor
class LangChainIngestor(Ingestor):
    async def process_file(self, file_path: Path, file_type: str, params: Dict[str, Any]) -> List[Dict]:
        with tracer.start_as_current_span(f"langchain_process_file_{file_path.name}") as span:
            span.set_attribute("file_name", file_path.name)
            span.set_attribute("file_type", file_type)
            try:
                logger.debug(f"Processing file {file_path} with loop {id(asyncio.get_event_loop())}")

                if not file_path.exists():
                    error_msg = f"No such file or directory: {file_path}"
                    logger.error(f"File not found: {error_msg}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                    error_count.labels(framework="langchain").inc()
                    raise FileNotFoundError(error_msg)
                
                loader_map = {
                    "pdf": PyPDFLoader,
                    "txt": TextLoader,
                    "csv": CSVLoader,
                    "xlsx": UnstructuredExcelLoader
                }
                loader_class = loader_map.get(file_type)
                if not loader_class:
                    raise ValueError(f"Unsupported file type for LangChain: {file_type}")

                loader = loader_class(str(file_path))
                documents = loader.load()

                return [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata | {"file_name": file_path.name}
                    }
                    for doc in documents
                ]
            except FileNotFoundError as e:
                error_msg = str(e)  # Already set to "No such file or directory: {file_path}"
                logger.error(f"LangChain file processing error for {file_path}: {error_msg}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                error_count.labels(framework="langchain").inc()
                raise ValueError(error_msg) from e
            except Exception as e:
                logger.error(f"LangChain file processing error for {file_path}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_count.labels(framework="langchain").inc()
                raise

    async def process_db(self, db_config: Dict[str, Any]) -> List[Dict]:
        with tracer.start_as_current_span("langchain_process_db") as span:
            span.set_attribute("db_name", db_config.get("dbName", ""))
            span.set_attribute("collection_name", db_config.get("collectionName", ""))
            try:
                logger.debug(f"Processing DB with loop {id(asyncio.get_event_loop())}")
                db_name = db_config.get("dbName")
                collection_name = db_config.get("collectionName")
                if not db_name or not collection_name:
                    raise ValueError("Missing dbName or collectionName")

                db = mongo_client[db_name]
                collection = db[collection_name]
                cursor = collection.find()
                documents = [
                    Document(
                        page_content=str(doc.get("content", str(doc))),
                        metadata={"dbName": db_name, "collectionName": collection_name}
                    )
                    async for doc in cursor
                ]

                return [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            except Exception as e:
                logger.error(f"LangChain DB processing error: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_count.labels(framework="langchain").inc()
                raise

# LlamaIndex Ingestor
class LlamaIndexIngestor(Ingestor):
    async def process_file(self, file_path: Path, file_type: str, params: Dict[str, Any]) -> List[Dict]:
        with tracer.start_as_current_span(f"llamaindex_process_file_{file_path.name}") as span:
            span.set_attribute("file_name", file_path.name)
            span.set_attribute("file_type", file_type)
            try:
                logger.debug(f"Processing file {file_path} with loop {id(asyncio.get_event_loop())}")

                if not file_path.exists():
                    error_msg = f"No such file or directory: {file_path}"
                    logger.error(f"File not found: {error_msg}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                    error_count.labels(framework="langchain").inc()
                    raise FileNotFoundError(error_msg)
                
                reader = SimpleDirectoryReader(
                    input_files=[str(file_path)],
                    file_metadata=lambda x: {"file_name": Path(x).name}
                )
                documents = reader.load_data()

                return [
                    {
                        "text": doc.text,
                        "metadata": doc.metadata | {"file_name": file_path.name}
                    }
                    for doc in documents
                ]
            except FileNotFoundError as e:
                error_msg = str(e)  # Already set to "No such file or directory: {file_path}"
                logger.error(f"LangChain file processing error for {file_path}: {error_msg}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                error_count.labels(framework="langchain").inc()
                raise ValueError(error_msg) from e
            except Exception as e:
                logger.error(f"LangChain file processing error for {file_path}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_count.labels(framework="langchain").inc()
                raise

    async def process_db(self, db_config: Dict[str, Any]) -> List[Dict]:
        with tracer.start_as_current_span("llamaindex_process_db") as span:
            span.set_attribute("db_name", db_config.get("dbName", ""))
            span.set_attribute("collection_name", db_config.get("collectionName", ""))
            try:
                logger.debug(f"Processing DB with loop {id(asyncio.get_event_loop())}")
                db_name = db_config.get("dbName")
                collection_name = db_config.get("collectionName")
                if not db_name or not collection_name:
                    raise ValueError("Missing dbName or collectionName")

                db = mongo_client[db_name]
                collection = db[collection_name]
                cursor = collection.find()
                documents = [
                    LlamaDocument(
                        text=str(doc.get("content", str(doc))),
                        metadata={"dbName": db_name, "collectionName": collection_name}
                    )
                    async for doc in cursor
                ]

                return [
                    {
                        "text": doc.text,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            except Exception as e:
                logger.error(f"LlamaIndex DB processing error: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_count.labels(framework="llamaindex").inc()
                raise

# Data Preparation Service
class DataPreparationService(CoreServiceServicer):
    def __init__(self):
        self.ingestors = {
            "langchain": LangChainIngestor(),
            "llamaindex": LlamaIndexIngestor()
        }

    async def Ingest(self, request: IngestRequest, context: grpc.aio.ServicerContext) -> JobResponse:
        with tracer.start_as_current_span("ingest") as span:
            span.set_attribute("job_id", request.job_id)
            span.set_attribute("framework", request.framework)
            job_id = request.job_id
            framework = request.framework
            logger.debug(f"Received ingestion request for job {job_id} with framework {framework}")
            logger.debug(f"Request details: {request}")
            logger.debug(f"Request config: {request.config}")
            config = dict(request.config)
            logger.debug(f"Config: {config}")
            logger.debug(f"Input data config: {request.input_data_config}")
            input_data_config = MessageToDict(request.input_data_config)
            loader_params = dict(request.loader_params)

            try:
                logger.debug(f"Starting ingestion for job {job_id} with loop {id(asyncio.get_event_loop())}")
                # Validate framework
                if framework not in self.ingestors:
                    logger.error(f"Invalid framework: {framework}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, f"Unsupported framework: {framework}"))
                    error_count.labels(framework=framework).inc()
                    return JobResponse(
                        job_id=job_id,
                        status="failed",
                        activity_type="ingest",
                        error=f"Unsupported framework: {framework}"
                    )

                # Fetch orgId from MongoDB
                db = mongo_client["applicationDB"]
                client_api_key = config.get("clientApiKey")
                if not client_api_key:
                    logger.error("Missing clientApiKey")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Missing clientApiKey"))
                    error_count.labels(framework=framework).inc()
                    return JobResponse(
                        job_id=job_id,
                        status="failed",
                        activity_type="ingest",
                        error="Missing clientApiKey"
                    )

                key = await db.clientApiKeys.find_one({"clientApiKey": client_api_key})
                if not key:
                    logger.error(f"Invalid API key: {client_api_key}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid API key"))
                    error_count.labels(framework=framework).inc()
                    return JobResponse(
                        job_id=job_id,
                        status="failed",
                        activity_type="ingest",
                        error="Invalid API key"
                    )
                org_id = key["orgId"]

                # Initialize org-specific DB
                org_db = mongo_client[org_id]

                # Track ingestion state: Queued
                await org_db.IngestionStates.insert_one({
                    "jobId": job_id,
                    "deployId": config.get("deployId"),
                    "clientApiKey": client_api_key,
                    "states": [{"state": "queued", "timestamp": datetime.now(timezone.utc)}],
                    "status": "queued",
                    "timestamp": datetime.now(timezone.utc)
                })

                # Update state: Reading
                await org_db.IngestionStates.update_one(
                    {"jobId": job_id},
                    {"$push": {"states": {"state": "reading", "timestamp": datetime.now(timezone.utc)}}}
                )

                # Process input based on type
                ingestor = self.ingestors[framework]
                documents = []
                if input_data_config.get("type") == "file":
                    file_names = input_data_config.get("fileNames", "").split(",")
                    documents = await self._process_files(
                        ingestor, file_names, loader_params, job_id, org_db
                    )
                elif input_data_config.get("type") == "db":
                    documents = await self._process_db(
                        ingestor, input_data_config, job_id, org_db
                    )
                else:
                    raise ValueError(f"Unsupported input type: {input_data_config.get('type')}")

                # Apply chunking
                splitter_config = await org_db.splitterDbConfig.find_one({"_id": config.get("splitterConfigId")})
                if not splitter_config:
                    raise ValueError("Splitter configuration not found")
                chunks = await self._apply_chunking(documents, splitter_config, job_id, org_db)

                # Update state: Completed
                await org_db.IngestionStates.update_one(
                    {"jobId": job_id},
                    {
                        "$push": {"states": {"state": "completed", "timestamp": datetime.now(timezone.utc)}},
                        "$set": {"status": "completed"}
                    }
                )

                logger.info(f"Job {job_id}: Ingestion completed, {len(chunks)} chunks generated")
                span.set_status(trace.Status(trace.StatusCode.OK))
                ingestion_count.labels(framework=framework, status="completed").inc()
                return JobResponse(job_id=job_id, status="completed", activity_type="ingest")
            except ValueError as e:
                logger.error(f"Ingestion error for job {job_id}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                error_count.labels(framework=framework).inc()
                if 'org_db' in locals():
                    await org_db.IngestionStates.update_one(
                        {"jobId": job_id},
                        {
                            "$push": {
                                "states": {
                                    "state": "failed",
                                    "timestamp": datetime.now(timezone.utc),
                                    "details": {"error": str(e)}
                                }
                            },
                            "$set": {"status": "failed", "error": str(e)}
                        }
                    )
                return JobResponse(job_id=job_id, status="failed", activity_type="ingest", error=str(e))
            except Exception as e:
                error_msg = f"Error processing job {job_id}: {str(e)}"
                logger.error(error_msg)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                error_count.labels(framework=framework).inc()
                if 'org_db' in locals():
                    await org_db.IngestionStates.update_one(
                        {"jobId": job_id},
                        {
                            "$push": {
                                "states": {
                                    "state": "failed",
                                    "timestamp": datetime.now(timezone.utc),
                                    "details": {"error": str(e)}
                                }
                            },
                            "$set": {"status": "failed", "error": str(e)}
                        }
                    )
                return JobResponse(job_id=job_id, status="failed", activity_type="ingest", error=str(e))

    async def _process_files(self, ingestor: Ingestor, file_names: List[str], loader_params: Dict, job_id: str, org_db):
        documents = []
        for file_name in file_names:
            file_path = Path(f"/data/{file_name.strip()}")
            if file_path.suffix[1:].lower() not in SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            docs = await self._process_single_file(ingestor, file_path, loader_params, job_id, org_db)
            documents.extend(docs)

        return documents

    async def _process_single_file(self, ingestor: Ingestor, file_path: Path, loader_params: Dict, job_id: str, org_db):
        with tracer.start_as_current_span(f"process_file_{file_path.name}") as span:
            span.set_attribute("file_name", file_path.name)
            file_type = file_path.suffix[1:].lower()
            await org_db.IngestionStates.update_one(
                {"jobId": job_id},
                {
                    "$push": {
                        "states": {
                            "state": f"reading_{file_type}",
                            "timestamp": datetime.now(timezone.utc),
                            "details": {"fileName": file_path.name}
                        }
                    }
                }
            )

            try:
                docs = await ingestor.process_file(file_path, file_type, loader_params)
                return docs
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    async def _process_db(self, ingestor: Ingestor, db_config: Dict[str, Any], job_id: str, org_db):
        with tracer.start_as_current_span("process_db") as span:
            span.set_attribute("db_name", db_config.get("dbName", ""))
            span.set_attribute("collection_name", db_config.get("collectionName", ""))
            await org_db.IngestionStates.update_one(
                {"jobId": job_id},
                {
                    "$push": {
                        "states": {
                            "state": "reading_db",
                            "timestamp": datetime.now(timezone.utc),
                            "details": db_config
                        }
                    }
                }
            )

            try:
                docs = await ingestor.process_db(db_config)
                return docs
            except Exception as e:
                logger.error(f"Error processing DB: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    async def _apply_chunking(self, documents: List[Dict], splitter_config: Dict, job_id: str, org_db):
        with tracer.start_as_current_span("chunking") as span:
            span.set_attribute("job_id", job_id)
            chunk_size = splitter_config["params"].get("chunkSize", 512)
            chunk_overlap = splitter_config["params"].get("chunkOverlap", 50)
            chunks = []

            await org_db.IngestionStates.update_one(
                {"jobId": job_id},
                {
                    "$push": {
                        "states": {
                            "state": "chunking",
                            "timestamp": datetime.now(timezone.utc),
                            "details": {"chunkSize": chunk_size, "chunkOverlap": chunk_overlap}
                        }
                    }
                }
            )

            for doc in documents:
                text = doc["text"]
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i:i + chunk_size]
                    chunks.append({"text": chunk, "metadata": doc["metadata"]})

            return chunks

# gRPC server setup
async def serve():
    global mongo_client
    loop = asyncio.get_event_loop()
    mongo_client = AsyncIOMotorClient(MONGO_URI, io_loop=loop)
    try:
        await mongo_client.admin.command("ping")
        logger.info(f"MongoDB connection established with loop {id(loop)}")
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise

    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")

    server = grpc.aio.server()
    add_CoreServiceServicer_to_server(DataPreparationService(), server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    logger.info(f"Starting gRPC server on port {GRPC_PORT} with loop {id(loop)}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())