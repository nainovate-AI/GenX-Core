# genx_platform/genx_components/microservices/metrics/tests/integration/test_metrics_integration.py
"""
Integration tests for Metrics Service
Tests the complete stack including TLS, authentication, and rate limiting
"""
import pytest
import grpc
import asyncio
import time
import os
from typing import AsyncGenerator

# Add genx_platform to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    metrics_service_pb2,
    metrics_service_pb2_grpc,
)


class TestMetricsIntegration:
    """Integration tests for metrics service"""
    
    @pytest.fixture
    async def secure_channel(self) -> AsyncGenerator[grpc.aio.Channel, None]:
        """Create secure gRPC channel with TLS"""
        # Load certificates
        with open('/certs/ca.crt', 'rb') as f:
            ca_cert = f.read()
        with open('/certs/client.crt', 'rb') as f:
            client_cert = f.read()
        with open('/certs/client.key', 'rb') as f:
            client_key = f.read()
        
        # Create credentials
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )
        
        # Create channel
        channel = grpc.aio.secure_channel(
            'metrics-service:50056',
            credentials
        )
        
        yield channel
        await channel.close()
    
    @pytest.fixture
    async def insecure_channel(self) -> AsyncGenerator[grpc.aio.Channel, None]:
        """Create insecure gRPC channel for testing"""
        channel = grpc.aio.insecure_channel('metrics-service:50056')
        yield channel
        await channel.close()
    
    @pytest.fixture
    def auth_metadata(self):
        """Get authentication metadata"""
        auth_token = os.environ.get('AUTH_TOKEN', 'test-token')
        return [('x-auth-token', auth_token)]
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_with_auth(self, secure_channel, auth_metadata):
        """Test getting system metrics with authentication"""
        stub = metrics_service_pb2_grpc.MetricsServiceStub(secure_channel)
        
        request = metrics_service_pb2.GetSystemMetricsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-123",
                user_id="test-user"
            )
        )
        
        response = await stub.GetSystemMetrics(
            request,
            metadata=auth_metadata
        )
        
        assert response.metadata.request_id == "test-123"
        assert response.metrics is not None
        assert response.metrics.cpu is not None
        assert response.metrics.memory is not None
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, secure_channel):
        """Test authentication failure"""
        stub = metrics_service_pb2_grpc.MetricsServiceStub(secure_channel)
        
        request = metrics_service_pb2.GetSystemMetricsRequest(
            metadata=common_pb2.RequestMetadata(request_id="test-124")
        )
        
        with pytest.raises(grpc.RpcError) as exc_info:
            await stub.GetSystemMetrics(
                request,
                metadata=[('x-auth-token', 'invalid-token')]
            )
        
        assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, secure_channel, auth_metadata):
        """Test rate limiting functionality"""
        stub = metrics_service_pb2_grpc.MetricsServiceStub(secure_channel)
        
        # Get rate limit from environment
        rate_limit = int(os.environ.get('RATE_LIMIT_REQUESTS_PER_MINUTE', '1000'))
        burst = int(os.environ.get('RATE_LIMIT_BURST', '100'))
        
        # Make burst + 1 requests rapidly
        request_count = 0
        rate_limited = False
        
        for i in range(burst + 10):
            request = metrics_service_pb2.GetSystemMetricsRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id=f"rate-test-{i}",
                    user_id="rate-test-user"
                )
            )
            
            try:
                await stub.GetSystemMetrics(
                    request,
                    metadata=auth_metadata + [('user-id', 'rate-test-user')]
                )
                request_count += 1
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    rate_limited = True
                    break
        
        assert rate_limited, f"Rate limiting not triggered after {request_count} requests"
    
    @pytest.mark.asyncio
    async def test_streaming_metrics(self, secure_channel, auth_metadata):
        """Test streaming metrics functionality"""
        stub = metrics_service_pb2_grpc.MetricsServiceStub(secure_channel)
        
        request = metrics_service_pb2.StreamSystemMetricsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="stream-test-1",
                user_id="test-user"
            ),
            interval_seconds=1,
            max_duration_seconds=5
        )
        
        updates_received = 0
        async for update in stub.StreamSystemMetrics(request, metadata=auth_metadata):
            updates_received += 1
            assert update.metrics is not None
            
            # Check first update is INITIAL
            if updates_received == 1:
                assert update.type == metrics_service_pb2.SystemMetricsUpdate.UPDATE_TYPE_INITIAL
        
        assert updates_received >= 3  # Should get at least initial + 2 updates
    
    @pytest.mark.asyncio
    async def test_health_check(self, secure_channel):
        """Test health check endpoint"""
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        
        health_stub = health_pb2_grpc.HealthStub(secure_channel)
        request = health_pb2.HealthCheckRequest(service="")
        
        response = await health_stub.Check(request)
        assert response.status == health_pb2.HealthCheckResponse.SERVING
    
    @pytest.mark.asyncio
    async def test_metrics_history(self, secure_channel, auth_metadata):
        """Test getting historical metrics"""
        stub = metrics_service_pb2_grpc.MetricsServiceStub(secure_channel)
        
        # Get current time
        from google.protobuf.timestamp_pb2 import Timestamp
        now = Timestamp()
        now.GetCurrentTime()
        
        # Get time 1 hour ago
        start_time = Timestamp()
        start_time.FromSeconds(int(time.time() - 3600))
        
        request = metrics_service_pb2.GetMetricsHistoryRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="history-test-1",
                user_id="test-user"
            ),
            start_time=start_time,
            end_time=now,
            resolution_seconds=300  # 5 minute resolution
        )
        
        response = await stub.GetMetricsHistory(request, metadata=auth_metadata)
        
        assert response.history_metadata is not None
        assert len(response.data_points) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])