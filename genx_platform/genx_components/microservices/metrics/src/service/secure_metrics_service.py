# genx_platform/genx_components/microservices/metrics/src/service/secure_metrics_service.py
"""
Enhanced Metrics Service with TLS, Authentication, and Rate Limiting
"""
import asyncio
import grpc
import time
import uuid
import os
import sys
import ssl
from typing import Optional, Dict, Any, AsyncIterator
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent import futures

# Add genx_platform to path
current_file = os.path.abspath(__file__)
service_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(service_dir)
metrics_root = os.path.dirname(src_dir)
microservices_dir = os.path.dirname(metrics_root)
genx_components = os.path.dirname(microservices_dir)
genx_platform = os.path.dirname(genx_components)
sys.path.insert(0, genx_platform)

# Import base service - use absolute imports
from genx_components.microservices.metrics.src.service.metrics_service import MetricsService
from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    metrics_service_pb2,
    metrics_service_pb2_grpc,
)
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

# Circuit breaker implementation
class CircuitBreaker:
    """Simple circuit breaker implementation"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.success_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def can_proceed(self) -> bool:
        """Check if operation can proceed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True


class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, requests_per_minute: int = 1000, burst: int = 100):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.user_buckets = defaultdict(lambda: {"tokens": burst, "last_update": time.time()})
    
    def allow_request(self, user_id: str = "anonymous") -> bool:
        """Check if request is allowed"""
        now = time.time()
        bucket = self.user_buckets[user_id]
        
        # Refill tokens
        time_passed = now - bucket["last_update"]
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(self.burst, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


class AuthInterceptor(grpc.aio.ServerInterceptor):
    """gRPC interceptor for authentication"""
    def __init__(self, auth_token: str, enable_auth: bool = True):
        self.auth_token = auth_token
        self.enable_auth = enable_auth
    
    async def intercept_service(self, continuation, handler_call_details):
        """Intercept and authenticate requests"""
        if not self.enable_auth:
            return await continuation(handler_call_details)
        
        # Skip auth for health checks
        if handler_call_details.method == "/grpc.health.v1.Health/Check":
            return await continuation(handler_call_details)
        
        # Check for auth token
        metadata = dict(handler_call_details.invocation_metadata or [])
        provided_token = metadata.get('x-auth-token', '')
        
        if provided_token != self.auth_token:
            async def abort_unauthorized(request, context):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    'Invalid authentication token'
                )
            return grpc.unary_unary_rpc_method_handler(abort_unauthorized)
        
        return await continuation(handler_call_details)


class SecureMetricsService(MetricsService):
    """Enhanced Metrics Service with security features"""
    
    def __init__(self, config=None, telemetry=None):
        super().__init__(config, telemetry)
        
        # Rate limiting
        rate_limit_enabled = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        if rate_limit_enabled:
            requests_per_minute = int(os.environ.get('RATE_LIMIT_REQUESTS_PER_MINUTE', '1000'))
            burst = int(os.environ.get('RATE_LIMIT_BURST', '100'))
            self.rate_limiter = RateLimiter(requests_per_minute, burst)
        else:
            self.rate_limiter = None
        
        # Circuit breaker
        cb_enabled = os.environ.get('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
        if cb_enabled:
            failure_threshold = int(os.environ.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
            timeout = int(os.environ.get('CIRCUIT_BREAKER_TIMEOUT_SECONDS', '60'))
            success_threshold = int(os.environ.get('CIRCUIT_BREAKER_SUCCESS_THRESHOLD', '2'))
            self.circuit_breaker = CircuitBreaker(failure_threshold, timeout, success_threshold)
        else:
            self.circuit_breaker = None
    
    async def _check_rate_limit(self, context, user_id: str = "anonymous") -> bool:
        """Check rate limit for request"""
        if not self.rate_limiter:
            return True
        
        if not self.rate_limiter.allow_request(user_id):
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Rate limit exceeded. Please try again later."
            )
            return False
        return True
    
    async def _check_circuit_breaker(self, context) -> bool:
        """Check circuit breaker status"""
        if not self.circuit_breaker:
            return True
        
        if not self.circuit_breaker.can_proceed():
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "Service temporarily unavailable. Circuit breaker is open."
            )
            return False
        return True
    
    async def GetSystemMetrics(
        self,
        request: metrics_service_pb2.GetSystemMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetSystemMetricsResponse:
        """Enhanced GetSystemMetrics with security checks"""
        
        # Extract user from metadata
        metadata = dict(context.invocation_metadata() or [])
        user_id = metadata.get('user-id', 'anonymous')
        
        # Check rate limit
        if not await self._check_rate_limit(context, user_id):
            return
        
        # Check circuit breaker
        if not await self._check_circuit_breaker(context):
            return
        
        try:
            # Call parent implementation
            response = await super().GetSystemMetrics(request, context)
            
            # Record success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            return response
            
        except Exception as e:
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise


async def create_secure_server(config, telemetry) -> grpc.aio.Server:
    """Create secure gRPC server with TLS and interceptors"""
    
    # Create server options
    options = [
        ('grpc.max_send_message_length', config.grpc_max_message_length),
        ('grpc.max_receive_message_length', config.grpc_max_message_length),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.http2.max_pings_without_data', 0),
    ]
    
    # Create interceptors
    interceptors = []
    
    # Add auth interceptor
    if os.environ.get('ENABLE_AUTH', 'true').lower() == 'true':
        auth_token = os.environ.get('AUTH_TOKEN', 'default-token')
        interceptors.append(AuthInterceptor(auth_token))
    
    # Create server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
        options=options,
        interceptors=interceptors
    )
    
    # Add metrics service
    metrics_service = SecureMetricsService(config, telemetry)
    await metrics_service.initialize()
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(
        metrics_service, server
    )
    
    # Add health service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    
    # Add reflection
    service_names = [
        metrics_service_pb2.DESCRIPTOR.services_by_name['MetricsService'].full_name,
        health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
        reflection.SERVICE_NAME,
    ]
    reflection.enable_server_reflection(service_names, server)
    
    # Configure TLS if enabled
    if os.environ.get('GRPC_TLS_ENABLED', 'true').lower() == 'true':
        # Load certificates
        cert_path = os.environ.get('GRPC_TLS_CERT_PATH', '/certs/server.crt')
        key_path = os.environ.get('GRPC_TLS_KEY_PATH', '/certs/server.key')
        ca_path = os.environ.get('GRPC_TLS_CA_PATH', '/certs/ca.crt')
        
        with open(cert_path, 'rb') as f:
            server_cert = f.read()
        with open(key_path, 'rb') as f:
            server_key = f.read()
        with open(ca_path, 'rb') as f:
            ca_cert = f.read()
        
        # Create server credentials
        server_credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True
        )
        
        # Add secure port
        server.add_secure_port(f'[::]:{config.service_port}', server_credentials)
    else:
        # Add insecure port
        server.add_insecure_port(f'[::]:{config.service_port}')
    
    return server