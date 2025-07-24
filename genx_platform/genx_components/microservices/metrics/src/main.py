#!/usr/bin/env python3
"""
Metrics Microservice - Production Grade
Collects and serves system metrics via gRPC
"""
import asyncio
import logging
import os
import signal
import sys
from concurrent import futures
from typing import Optional

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

# Add genx_platform to Python path
current_file = os.path.abspath(__file__)
metrics_src = os.path.dirname(current_file)
metrics_root = os.path.dirname(metrics_src)
microservices_dir = os.path.dirname(metrics_root)
genx_components = os.path.dirname(microservices_dir)
genx_platform = os.path.dirname(genx_components)
sys.path.insert(0, genx_platform)

# Now we can import from genx_components
from genx_components.common.config import BaseServiceConfig
from genx_components.common.telemetry import GenxTelemetry

# Add metrics src to path for local imports
sys.path.insert(0, metrics_src)

# Import generated protobuf files from common location
from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    metrics_service_pb2,
    metrics_service_pb2_grpc,
)
from service.metrics_service import MetricsService
from utils.logger import setup_logging


class MetricsServiceConfig(BaseServiceConfig):
    """Configuration specific to Metrics Service"""
    service_name: str = "metrics-service"
    service_port: int = 50056
    
    # Cache settings
    cache_ttl_seconds: int = 30
    
    # Collection settings
    background_collection_interval: int = 30
    
    # Model storage
    model_storage_path: str = "/models"

# Setup logging
logger = setup_logging(__name__)

# Global instances
_server: Optional[grpc.aio.Server] = None
_telemetry: Optional[GenxTelemetry] = None
_config: Optional[MetricsServiceConfig] = None


def get_config() -> MetricsServiceConfig:
    """Get configuration instance"""
    global _config
    if _config is None:
        _config = MetricsServiceConfig()
    return _config


async def serve() -> None:
    """Start the gRPC server with production configurations"""
    global _server, _telemetry
    
    config = get_config()
    
    # Setup telemetry
    if config.telemetry_enabled:
        _telemetry = GenxTelemetry(
            service_name=config.service_name,
            service_version=config.service_version,
            telemetry_endpoint=config.telemetry_endpoint,
            environment=config.environment
        )
        _telemetry.initialize()
    
    # Log startup
    logger.info(f"Starting {config.service_name} on port {config.service_port}")
    
    # Create server with production options
    server_options = [
        ('grpc.max_send_message_length', config.grpc_max_message_length),
        ('grpc.max_receive_message_length', config.grpc_max_message_length),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.max_connection_idle_ms', 300000),  # 5 minutes
        ('grpc.max_connection_age_ms', 3600000),  # 1 hour
        ('grpc.max_connection_age_grace_ms', 5000),
        ('grpc.http2.max_frame_size', 16384),
        ('grpc.enable_retries', 1),
        ('grpc.service_config', '{"retryPolicy": {"maxAttempts": 3, "initialBackoff": "0.1s", "maxBackoff": "1s", "backoffMultiplier": 2}}')
    ]
    
    # Create async server
    _server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
        options=server_options,
        maximum_concurrent_rpcs=100
    )
    
    # Initialize and add services
    metrics_service = MetricsService(config, _telemetry)
    await metrics_service.initialize()
    
    # Add main service
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(
        metrics_service, _server
    )
    
    # Add health service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, _server)
    health_servicer.set(
        "genx.metrics.v1.MetricsService",
        health_pb2.HealthCheckResponse.SERVING
    )
    
    # Add reflection for debugging
    if config.debug or config.environment == "development":
        SERVICE_NAMES = (
            metrics_service_pb2.DESCRIPTOR.services_by_name['MetricsService'].full_name,
            health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, _server)
    
    # Bind to port
    listen_addr = f'[::]:{config.service_port}'
    _server.add_insecure_port(listen_addr)
    logger.info(f"{config.service_name} starting on {listen_addr}")
    
    # Start server
    await _server.start()
    
    logger.info(
        f"{config.service_name} started successfully",
        extra={
            "port": config.service_port,
            "environment": config.environment,
            "workers": config.grpc_max_workers,
            "telemetry_enabled": config.telemetry_enabled,
            "reflection_enabled": config.debug or config.environment == "development"
        }
    )
    
    try:
        await _server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("Server cancelled, shutting down...")
        await shutdown()


async def shutdown() -> None:
    """Graceful shutdown"""
    global _server
    
    if _server:
        logger.info("Initiating graceful shutdown...")
        
        # Stop accepting new requests
        await _server.stop(grace=5.0)
        
        logger.info("Metrics service stopped")


def handle_signal(signum: int, frame) -> None:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    
    # Create task for shutdown
    asyncio.create_task(shutdown())


def main() -> None:
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    # Run server
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()