# genx_platform/genx_components/microservices/metrics/src/main.py
#!/usr/bin/env python3
"""
Metrics Microservice - Production Grade
Collects and serves system metrics via gRPC with security features
"""
import asyncio
import logging
import os
import signal
import sys
from concurrent import futures
from typing import Optional
from pathlib import Path

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

# Import the appropriate service based on security configuration
use_secure_service = os.environ.get('GRPC_TLS_ENABLED', 'true').lower() == 'true'
if use_secure_service:
    from genx_components.microservices.metrics.src.service.secure_metrics_service import SecureMetricsService as MetricsServiceImpl
    from genx_components.microservices.metrics.src.service.secure_metrics_service import create_secure_server
else:
    from genx_components.microservices.metrics.src.service.metrics_service import MetricsService as MetricsServiceImpl

from genx_components.microservices.metrics.src.utils.logger import setup_logging


class MetricsServiceConfig(BaseServiceConfig):
    """Configuration specific to Metrics Service"""
    # Service metadata
    service_name: str = "metrics-service"
    service_version: str = "1.0.0"  # Added to match logger.py and MetricsService
    service_port: int = 50056
    environment: str = "production"  # Added to match logger.py

    # Cache settings
    cache_ttl_seconds: int = 30

    # Collection settings
    background_collection_interval: int = 30

    # Model storage
    model_storage_path: str = "/models"

    # TLS settings
    grpc_tls_enabled: bool = True
    grpc_tls_cert_path: str = "/certs/server.crt"
    grpc_tls_key_path: str = "/certs/server.key"
    grpc_tls_ca_path: str = "/certs/ca.crt"

    # Auth settings
    enable_auth: bool = True
    auth_token: str = "default-token"  # Renamed from metrics_auth_token for consistency
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 1000
    rate_limit_burst: int = 100

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_success_threshold: int = 2

    # Alert thresholds
    alert_cpu_threshold: float = 80.0
    alert_memory_threshold: float = 85.0
    alert_disk_threshold: float = 90.0

    # Grafana credentials
    grafana_user: str = "admin"
    grafana_password: str = "admin"

    # SMTP settings for email alerts
    smtp_host: str = "smtp.gmail.com:587"
    smtp_user: str = "your-email@gmail.com"
    smtp_password: str = "your-app-password"
    alert_email_from: str = "alerts@genx.ai"

    # Notification integrations
    slack_webhook_url: str = ""
    pagerduty_service_key: str = ""

    # Telemetry settings (from BaseServiceConfig, if needed)
    telemetry_enabled: bool = True
    telemetry_endpoint: str = ""
    grpc_max_message_length: int = 4 * 1024 * 1024  # 4MB default
    grpc_max_workers: int = 10  # Default thread pool size


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
    logger.info(f"TLS enabled: {config.grpc_tls_enabled}")
    logger.info(f"Authentication enabled: {config.enable_auth}")
    logger.info(f"Rate limiting enabled: {config.rate_limit_enabled}")
    
    # Create server based on security configuration
    if use_secure_service:
        _server = await create_secure_server(config, _telemetry)
    else:
        # Create standard server for development
        server_options = [
            ('grpc.max_send_message_length', config.grpc_max_message_length),
            ('grpc.max_receive_message_length', config.grpc_max_message_length),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
        ]
        
        _server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
            options=server_options
        )
        
        # Add metrics service
        metrics_service = MetricsServiceImpl(config, _telemetry)
        await metrics_service.initialize()
        metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(
            metrics_service, _server
        )
        
        # Add health service
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, _server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        
        # Add reflection
        service_names = [
            metrics_service_pb2.DESCRIPTOR.services_by_name['MetricsService'].full_name,
            health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
            reflection.SERVICE_NAME,
        ]
        reflection.enable_server_reflection(service_names, _server)
        
        # Add port
        _server.add_insecure_port(f'[::]:{config.service_port}')
    
    # Start server
    await _server.start()
    logger.info(f"Server started successfully on port {config.service_port}")
    
    # Setup shutdown handler
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(shutdown())
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Keep server running
    try:
        await _server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")


async def shutdown():
    """Graceful shutdown"""
    global _server, _telemetry
    
    logger.info("Starting graceful shutdown...")
    
    if _server:
        # Stop accepting new requests
        await _server.stop(grace=10)
        logger.info("Server stopped")
    
    if _telemetry:
        _telemetry.shutdown()
        logger.info("Telemetry shutdown")
    
    logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    try:
        await serve()
    except Exception as e:
        logger.error(f"Server failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Check if TLS certificates exist when TLS is enabled
    if use_secure_service:
        cert_path = os.environ.get('GRPC_TLS_CERT_PATH', '/certs/server.crt')
        key_path = os.environ.get('GRPC_TLS_KEY_PATH', '/certs/server.key')
        ca_path = os.environ.get('GRPC_TLS_CA_PATH', '/certs/ca.crt')
        
        if not all(Path(p).exists() for p in [cert_path, key_path, ca_path]):
            logger.warning("TLS certificates not found. Please run 'make certs-generate' first.")
            logger.info("Starting in insecure mode for development...")
            use_secure_service = False
            os.environ['GRPC_TLS_ENABLED'] = 'false'
    
    asyncio.run(main())