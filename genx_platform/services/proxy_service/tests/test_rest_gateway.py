# services/proxy_service/tests/test_rest_gateway.py
"""
Test suite for REST API Gateway
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import jwt
import json

from ..src.api.rest_gateway import router, GrpcClientManager
from ..src.generated import metrics_service_pb2
from ..src.core.config import get_settings

settings = get_settings()


@pytest.fixture
def app():
    """Create FastAPI app with router"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Create valid auth token"""
    payload = {
        "user_id": "test-user-123",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "nbf": datetime.utcnow(),
        "iss": settings.APP_NAME,
        "jti": "test-token-id",
        "type": "access"
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers"""
    return {"Authorization": f"Bearer {auth_token}"}


class MockMetricsProto:
    """Mock protobuf metrics object"""
    def __init__(self):
        self.cpu = Mock()
        self.cpu.usage_percent = 45.5
        self.cpu.per_core_percent = [40.0, 50.0, 45.0, 48.0]
        self.cpu.frequency = Mock(current_mhz=2400.0, min_mhz=800.0, max_mhz=3600.0)
        self.cpu.load_average = Mock(one_minute=1.5, five_minutes=1.8, fifteen_minutes=2.0)
        self.cpu.count = 4
        self.cpu.count_logical = 8
        
        self.memory = Mock()
        self.memory.total_bytes = 17179869184
        self.memory.available_bytes = 8589934592
        self.memory.used_bytes = 8589934592
        self.memory.free_bytes = 4294967296
        self.memory.percent = 50.0
        self.memory.swap = Mock(
            total_bytes=2147483648,
            used_bytes=0,
            free_bytes=2147483648,
            percent=0.0
        )
        
        self.gpu = []
        self.disk = Mock()
        self.network = Mock()
    
    def HasField(self, field):
        return hasattr(self, field)


class TestRESTGateway:
    """Test REST API Gateway endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_unauthorized_request(self, client):
        """Test request without authentication"""
        response = client.get("/api/v1/metrics/system")
        
        assert response.status_code == 403  # Forbidden (no auth header)
    
    def test_invalid_token(self, client):
        """Test request with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.side_effect = jwt.InvalidTokenError("Invalid")
            
            response = client.get("/api/v1/metrics/system", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid token" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, client, auth_headers):
        """Test getting system metrics"""
        mock_metrics = MockMetricsProto()
        mock_response = Mock()
        mock_response.metrics = mock_metrics
        mock_response.source = "fresh"
        mock_response.timestamp = Mock()
        mock_response.timestamp.ToDatetime.return_value = datetime.utcnow()
        
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            # Mock authentication
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.rest_gateway.grpc_client') as mock_grpc:
                # Mock gRPC client
                mock_stub = AsyncMock()
                mock_stub.GetSystemMetrics.return_value = mock_response
                mock_grpc.get_metrics_stub.return_value = mock_stub
                
                response = client.get(
                    "/api/v1/metrics/system",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "success"
                assert data["source"] == "fresh"
                assert data["data"]["cpu"]["usage_percent"] == 45.5
                assert len(data["data"]["cpu"]["per_core_percent"]) == 4
    
    @pytest.mark.asyncio
    async def test_get_metrics_with_types(self, client, auth_headers):
        """Test getting specific metric types"""
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.rest_gateway.grpc_client') as mock_grpc:
                mock_stub = AsyncMock()
                mock_grpc.get_metrics_stub.return_value = mock_stub
                
                # Capture the gRPC request
                grpc_request = None
                async def capture_request(req):
                    nonlocal grpc_request
                    grpc_request = req
                    mock_response = Mock()
                    mock_response.metrics = MockMetricsProto()
                    mock_response.source = "fresh"
                    mock_response.timestamp = Mock()
                    mock_response.timestamp.ToDatetime.return_value = datetime.utcnow()
                    return mock_response
                
                mock_stub.GetSystemMetrics.side_effect = capture_request
                
                response = client.get(
                    "/api/v1/metrics/system?metric_types=cpu,memory",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                
                # Verify correct metric types were requested
                assert metrics_service_pb2.METRIC_TYPE_CPU in grpc_request.metric_types
                assert metrics_service_pb2.METRIC_TYPE_MEMORY in grpc_request.metric_types
    
    @pytest.mark.asyncio
    async def test_get_metrics_force_refresh(self, client, auth_headers):
        """Test force refresh parameter"""
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.rest_gateway.grpc_client') as mock_grpc:
                mock_stub = AsyncMock()
                mock_grpc.get_metrics_stub.return_value = mock_stub
                
                # Capture the gRPC request
                grpc_request = None
                async def capture_request(req):
                    nonlocal grpc_request
                    grpc_request = req
                    mock_response = Mock()
                    mock_response.metrics = MockMetricsProto()
                    mock_response.source = "fresh"
                    mock_response.timestamp = Mock()
                    mock_response.timestamp.ToDatetime.return_value = datetime.utcnow()
                    return mock_response
                
                mock_stub.GetSystemMetrics.side_effect = capture_request
                
                response = client.get(
                    "/api/v1/metrics/system?force_refresh=true",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                assert grpc_request.force_refresh is True
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting"""
        # This would require setting up the rate limiter properly
        # For now, we'll just verify the decorator is present
        from ..src.api.rest_gateway import get_system_metrics
        
        # Check if rate limit decorator is applied
        assert hasattr(get_system_metrics, '_rate_limit')
    
    @pytest.mark.asyncio
    async def test_metrics_history(self, client, auth_headers):
        """Test getting metrics history"""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        mock_response = Mock()
        mock_response.data_points = []
        mock_response.metadata = Mock(
            start_time=Mock(),
            end_time=Mock(),
            count=0,
            resolution_seconds=60
        )
        mock_response.metadata.start_time.ToDatetime.return_value = start_time
        mock_response.metadata.end_time.ToDatetime.return_value = end_time
        
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.rest_gateway.grpc_client') as mock_grpc:
                mock_stub = AsyncMock()
                mock_stub.GetMetricsHistory.return_value = mock_response
                mock_grpc.get_metrics_stub.return_value = mock_stub
                
                response = client.get(
                    f"/api/v1/metrics/system/history"
                    f"?start_time={start_time.isoformat()}"
                    f"&end_time={end_time.isoformat()}"
                    f"&resolution=60",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "metadata" in data
    
    @pytest.mark.asyncio
    async def test_metrics_history_validation(self, client, auth_headers):
        """Test metrics history validation"""
        # Test invalid time range
        start_time = datetime.utcnow()
        end_time = datetime.utcnow() - timedelta(hours=1)  # End before start
        
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            
            response = client.get(
                f"/api/v1/metrics/system/history"
                f"?start_time={start_time.isoformat()}"
                f"&end_time={end_time.isoformat()}",
                headers=auth_headers
            )
            
            assert response.status_code == 400
            assert "End time must be after start time" in response.json()["detail"]
    
    def test_sse_endpoint(self, client, auth_headers):
        """Test Server-Sent Events endpoint"""
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            
            # SSE endpoints return streaming responses
            # TestClient doesn't handle streaming well, so we just verify it doesn't error
            with client as c:
                response = c.get(
                    "/api/v1/metrics/system/sse",
                    headers=auth_headers,
                    stream=True
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestGrpcClientManager:
    """Test gRPC client manager"""
    
    @pytest.mark.asyncio
    async def test_get_metrics_stub(self):
        """Test getting metrics stub"""
        manager = GrpcClientManager()
        
        with patch('grpc.aio.insecure_channel') as mock_channel:
            mock_channel.return_value = Mock()
            
            stub = await manager.get_metrics_stub()
            
            assert stub is not None
            # Verify channel was created
            mock_channel.assert_called_once()
            
            # Get again - should reuse
            stub2 = await manager.get_metrics_stub()
            assert stub is stub2
            
            # Only one channel should be created
            assert mock_channel.call_count == 1
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_secure_channel_production(self):
        """Test secure channel creation in production"""
        manager = GrpcClientManager()
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            with patch('services.proxy_service.src.api.rest_gateway.settings') as mock_settings:
                mock_settings.ENVIRONMENT = 'production'
                mock_settings.METRICS_SERVICE_URL = 'localhost:50051'
                
                with patch('grpc.ssl_channel_credentials') as mock_creds:
                    with patch('grpc.aio.secure_channel') as mock_secure_channel:
                        mock_creds.return_value = Mock()
                        mock_secure_channel.return_value = Mock()
                        
                        stub = await manager.get_metrics_stub()
                        
                        # Verify secure channel was used
                        mock_creds.assert_called_once()
                        mock_secure_channel.assert_called_once()
        
        await manager.close()


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_grpc_error_handling(self, client, auth_headers):
        """Test handling of gRPC errors"""
        import grpc
        
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = True
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.rest_gateway.grpc_client') as mock_grpc:
                # Mock gRPC error
                mock_stub = AsyncMock()
                mock_error = grpc.RpcError()
                mock_error.code = Mock(return_value=grpc.StatusCode.UNAVAILABLE)
                mock_stub.GetSystemMetrics.side_effect = mock_error
                mock_grpc.get_metrics_stub.return_value = mock_stub
                
                response = client.get(
                    "/api/v1/metrics/system",
                    headers=auth_headers
                )
                
                assert response.status_code == 500
                assert "Service unavailable" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_permission_denied(self, client, auth_headers):
        """Test permission denied scenario"""
        with patch('services.proxy_service.src.api.rest_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user-123"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.has_permission.return_value = False  # No permission
            
            response = client.get(
                "/api/v1/metrics/system",
                headers=auth_headers
            )
            
            assert response.status_code == 403
            assert "Insufficient permissions" in response.json()["detail"]