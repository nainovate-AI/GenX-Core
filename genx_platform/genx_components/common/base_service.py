"""
genx_platform/genx_components/common/base_service.py
Base Microservice Class for GenX Platform
All GenX microservices should inherit from this base class
"""
import asyncio
import signal
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from concurrent import futures
import grpc
# Fix the import - it should be grpc_health.v1 not grpc_health_checking
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .config import BaseServiceConfig
from .telemetry import GenxTelemetry
# from .registry import ServiceRegistry

logger = logging.getLogger(__name__)


class GenxMicroservice(ABC):
    """Base class for all GenX microservices"""
    
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize base microservice
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.telemetry: Optional[GenxTelemetry] = None
        self.registry: Optional[ServiceRegistry] = None
        self.grpc_server: Optional[grpc.Server] = None
        self.health_servicer: Optional[health.HealthServicer] = None
        self._is_running = False
        self._start_time = 0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure structured logging"""
        log_level = logging.DEBUG if self.config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                # Add file handler in production
            ]
        )
        
    async def start(self):
        """Start the microservice"""
        try:
            logger.info(f"Starting {self.config.service_name} v{self.config.service_version}")
            self._start_time = time.time()
            
            # Initialize telemetry
            if self.config.telemetry_enabled:
                self.telemetry = GenxTelemetry(
                    service_name=self.config.service_name,
                    service_version=self.config.service_version,
                    telemetry_endpoint=self.config.telemetry_endpoint,
                    environment=self.config.environment
                )
                self.telemetry.initialize()
            
            # Initialize service registry
            if self.config.registry_enabled:
                # We'll implement this later
                pass
            
            # Initialize service-specific components FIRST
            await self.initialize()
            
            # THEN Initialize gRPC server
            await self._init_grpc_server()
            
            # Start gRPC server
            await self._start_grpc_server()
            
            # Register with service discovery
            if self.registry:
                await self.registry.register()
            
            self._is_running = True
            logger.info(f"{self.config.service_name} started successfully on port {self.config.service_port}")
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Keep running
            await self._run_forever()
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            await self.stop()
            raise
    
    async def _init_grpc_server(self):
        """Initialize gRPC server with health checks"""
        # Create server with options
        options = [
            ('grpc.max_send_message_length', self.config.grpc_max_message_length),
            ('grpc.max_receive_message_length', self.config.grpc_max_message_length),
        ]
        
        self.grpc_server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.config.grpc_max_workers),
            options=options
        )
        
        # Add health service
        self.health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(
            self.health_servicer, self.grpc_server
        )
        
        # Service starts as NOT_SERVING
        self.health_servicer.set(
            self.config.service_name,
            health_pb2.HealthCheckResponse.NOT_SERVING
        )
        
        # Add service-specific handlers
        await self.add_grpc_services(self.grpc_server)
    
    async def _start_grpc_server(self):
        """Start the gRPC server"""
        address = f'[::]:{self.config.service_port}'
        self.grpc_server.add_insecure_port(address)
        await self.grpc_server.start()
        
        # Update health status
        self.health_servicer.set(
            self.config.service_name,
            health_pb2.HealthCheckResponse.SERVING
        )
    
    async def stop(self):
        """Gracefully stop the microservice"""
        logger.info(f"Stopping {self.config.service_name}")
        self._is_running = False
        
        # Update health status
        if self.health_servicer:
            self.health_servicer.set(
                self.config.service_name,
                health_pb2.HealthCheckResponse.NOT_SERVING
            )
        
        # Deregister from service discovery
        if self.registry:
            await self.registry.deregister()
        
        # Service-specific cleanup
        await self.cleanup()
        
        # Stop gRPC server
        if self.grpc_server:
            await self.grpc_server.stop(grace=10)
        
        # Shutdown telemetry
        if self.telemetry:
            self.telemetry.shutdown()
            
        logger.info(f"{self.config.service_name} stopped")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_forever(self):
        """Keep the service running"""
        while self._is_running:
            await asyncio.sleep(1)
            
            # Periodic health checks or maintenance tasks
            if int(time.time()) % 60 == 0:  # Every minute
                await self.health_check()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        uptime = time.time() - self._start_time
        health_status = {
            "service": self.config.service_name,
            "version": self.config.service_version,
            "status": "healthy",
            "uptime_seconds": uptime,
            "environment": self.config.environment
        }
        
        # Add service-specific health checks
        try:
            service_health = await self.check_health()
            health_status.update(service_health)
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    # Abstract methods that services must implement
    @abstractmethod
    async def initialize(self):
        """Initialize service-specific components"""
        pass
    
    @abstractmethod
    async def add_grpc_services(self, server: grpc.Server):
        """Add service-specific gRPC services to the server"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up service-specific resources"""
        pass
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Service-specific health checks"""
        pass