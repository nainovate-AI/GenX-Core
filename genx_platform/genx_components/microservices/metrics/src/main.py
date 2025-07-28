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
from pydantic import Field

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
    # Service metadata - these override BaseServiceConfig defaults
    service_name: str = Field(default="metrics-service", env='SERVICE_NAME')
    service_version: str = Field(default="1.0.0", env='SERVICE_VERSION')
    service_port: int = Field(default=50056, env='SERVICE_PORT')
    environment: str = Field(default="production", env='ENVIRONMENT')

    # Cache settings
    cache_ttl_seconds: int = Field(default=30, env='CACHE_TTL_SECONDS')

    # Collection settings
    background_collection_interval: int = Field(default=30, env='BACKGROUND_COLLECTION_INTERVAL')

    # Model storage
    model_storage_path: str = Field(default="/models", env='MODEL_STORAGE_PATH')

    # TLS settings
    grpc_tls_enabled: bool = Field(default=True, env='GRPC_TLS_ENABLED')
    grpc_tls_cert_path: str = Field(default="/certs/server.crt", env='GRPC_TLS_CERT_PATH')
    grpc_tls_key_path: str = Field(default="/certs/server.key", env='GRPC_TLS_KEY_PATH')
    grpc_tls_ca_path: str = Field(default="/certs/ca.crt", env='GRPC_TLS_CA_PATH')

    # Auth settings - Fixed to use AUTH_TOKEN environment variable
    enable_auth: bool = Field(default=True, env='ENABLE_AUTH')
    auth_token: str = Field(default="default-token", env='AUTH_TOKEN')  # Maps to AUTH_TOKEN env var
    rate_limit_enabled: bool = Field(default=True, env='RATE_LIMIT_ENABLED')
    rate_limit_requests_per_minute: int = Field(default=1000, env='RATE_LIMIT_REQUESTS_PER_MINUTE')
    rate_limit_burst: int = Field(default=100, env='RATE_LIMIT_BURST')

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(default=True, env='CIRCUIT_BREAKER_ENABLED')
    circuit_breaker_failure_threshold: int = Field(default=5, env='CIRCUIT_BREAKER_FAILURE_THRESHOLD')
    circuit_breaker_timeout_seconds: int = Field(default=60, env='CIRCUIT_BREAKER_TIMEOUT_SECONDS')
    circuit_breaker_success_threshold: int = Field(default=2, env='CIRCUIT_BREAKER_SUCCESS_THRESHOLD')

    # Alert thresholds
    alert_cpu_threshold: float = Field(default=80.0, env='ALERT_CPU_THRESHOLD')
    alert_memory_threshold: float = Field(default=85.0, env='ALERT_MEMORY_THRESHOLD')
    alert_disk_threshold: float = Field(default=90.0, env='ALERT_DISK_THRESHOLD')

    # Grafana credentials
    grafana_user: str = Field(default="admin", env='GRAFANA_USER')
    grafana_password: str = Field(default="admin", env='GRAFANA_PASSWORD')

    # SMTP settings for email alerts
    smtp_host: str = Field(default="smtp.gmail.com:587", env='SMTP_HOST')
    smtp_user: str = Field(default="your-email@gmail.com", env='SMTP_USER')
    smtp_password: str = Field(default="your-app-password", env='SMTP_PASSWORD')
    alert_email_from: str = Field(default="alerts@genx.ai", env='ALERT_EMAIL_FROM')

    # Notification integrations
    slack_webhook_url: str = Field(default="", env='SLACK_WEBHOOK_URL')
    pagerduty_service_key: str = Field(default="", env='PAGERDUTY_SERVICE_KEY')

    # Telemetry settings (inherited from BaseServiceConfig but can be overridden)
    telemetry_enabled: bool = Field(default=True, env='TELEMETRY_ENABLED')
    telemetry_endpoint: str = Field(default="http://otel-collector:4317", env='TELEMETRY_ENDPOINT')
    grpc_max_message_length: int = Field(default=4 * 1024 * 1024, env='GRPC_MAX_MESSAGE_LENGTH')  # 4MB default
    grpc_max_workers: int = Field(default=10, env='GRPC_MAX_WORKERS')  # Default thread pool size

    class Config:
        # This allows the configuration to accept extra fields from environment
        # without throwing validation errors
        extra = 'ignore'  # Changed from 'forbid' to 'ignore' to handle extra env vars
        env_file = '.env'
        case_sensitive = False


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
        reflection.enable_server_reflection(
            (
                metrics_service_pb2.DESCRIPTOR.services_by_name["MetricsService"].full_name,
                health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
                reflection.SERVICE_NAME,
            ),
            _server,
        )
        
        # Start server
        _server.add_insecure_port(f"[::]:{config.service_port}")
        await _server.start()
    
    logger.info(f"{config.service_name} started successfully")
    
    # Keep server running
    try:
        await _server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("Server cancelled")


async def shutdown(signal_received=None):
    """Graceful shutdown"""
    logger.info(f"Shutdown signal received: {signal_received}")
    
    if _server:
        logger.info("Stopping gRPC server...")
        await _server.stop(grace=30)
    
    if _telemetry:
        logger.info("Flushing telemetry...")
        _telemetry.shutdown()
    
    logger.info("Shutdown complete")


def handle_signal(sig, frame):
    """Handle shutdown signals"""
    asyncio.create_task(shutdown(sig))


async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    try:
        await serve()
    except Exception as e:
        logger.error(f"Server failed with error: {e}", exc_info=True)
        await shutdown()
        raise


if __name__ == "__main__":
    # Set up asyncio event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)