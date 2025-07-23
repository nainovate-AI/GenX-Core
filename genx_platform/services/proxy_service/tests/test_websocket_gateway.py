# services/proxy_service/tests/test_websocket_gateway.py
"""
Test suite for WebSocket Gateway
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket
from fastapi.testclient import TestClient
import jwt

from ..src.api.websocket_gateway import (
    WebSocketConnection,
    WebSocketManager,
    WSMessageType,
    handle_websocket_connection,
    handle_message,
    proto_to_dict
)
from ..src.generated import metrics_service_pb2
from ..src.core.config import get_settings

settings = get_settings()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket"""
    ws = Mock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    ws.client = Mock(host="127.0.0.1")
    return ws


@pytest.fixture
def mock_token():
    """Create mock JWT token"""
    payload = {
        "user_id": "test-user-123",
        "exp": 9999999999,  # Far future
        "iat": 1234567890,
        "nbf": 1234567890,
        "iss": settings.APP_NAME,
        "jti": "test-token-id",
        "type": "access"
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
async def ws_connection(mock_websocket):
    """Create WebSocket connection"""
    conn = WebSocketConnection(mock_websocket, "test-conn-123")
    yield conn
    # Cleanup
    await conn.close()


@pytest.fixture
async def ws_manager():
    """Create WebSocket manager"""
    manager = WebSocketManager()
    await manager.initialize()
    yield manager
    await manager.shutdown()


class TestWebSocketConnection:
    """Test WebSocket connection handling"""
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, ws_connection, mock_token):
        """Test successful authentication"""
        with patch('services.proxy_service.src.api.websocket_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {
                "user_id": "test-user-123",
                "type": "access"
            }
            
            result = await ws_connection.authenticate(mock_token)
            
            assert result is True
            assert ws_connection.authenticated is True
            assert ws_connection.user_id == "test-user-123"
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, ws_connection):
        """Test authentication failure"""
        with patch('services.proxy_service.src.api.websocket_gateway.security_manager') as mock_security:
            mock_security.verify_token.side_effect = Exception("Invalid token")
            
            result = await ws_connection.authenticate("bad-token")
            
            assert result is False
            assert ws_connection.authenticated is False
            assert ws_connection.user_id is None
    
    @pytest.mark.asyncio
    async def test_close_connection(self, ws_connection):
        """Test closing connection"""
        # Add some mock tasks
        mock_task = Mock()
        mock_task.cancel = Mock()
        ws_connection._tasks.add(mock_task)
        
        await ws_connection.close(code=1000, reason="Test close")
        
        # Verify task was cancelled
        mock_task.cancel.assert_called_once()
        
        # Verify websocket close was called
        ws_connection.websocket.close.assert_called_once_with(
            code=1000,
            reason="Test close"
        )


class TestWebSocketManager:
    """Test WebSocket manager"""
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, ws_manager, mock_websocket):
        """Test connecting and disconnecting"""
        # Connect
        connection = await ws_manager.connect(mock_websocket)
        
        assert connection.connection_id in ws_manager.connections
        assert len(ws_manager.connections) == 1
        
        # Disconnect
        await ws_manager.disconnect(connection)
        
        assert connection.connection_id not in ws_manager.connections
        assert len(ws_manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive(self, ws_manager, mock_websocket):
        """Test cleanup of inactive connections"""
        # Create connection with old heartbeat
        connection = await ws_manager.connect(mock_websocket)
        connection.last_heartbeat = 0  # Very old timestamp
        
        # Wait for cleanup (with shorter interval for testing)
        with patch('services.proxy_service.src.api.websocket_gateway.asyncio.sleep', 
                  new_callable=AsyncMock) as mock_sleep:
            # Make cleanup run once
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            
            # Run cleanup
            await ws_manager._cleanup_inactive()
        
        # Connection should be removed
        assert len(ws_manager.connections) == 0


class TestMessageHandling:
    """Test message handling"""
    
    @pytest.mark.asyncio
    async def test_handle_authenticate(self, ws_connection, mock_token):
        """Test authentication message handling"""
        with patch('services.proxy_service.src.api.websocket_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {
                "user_id": "test-user-123",
                "type": "access"
            }
            mock_security.is_token_blacklisted.return_value = False
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.websocket_gateway.send_message') as mock_send:
                message = {
                    "type": WSMessageType.AUTHENTICATE,
                    "token": mock_token
                }
                
                await handle_message(ws_connection, message)
                
                # Verify authentication
                assert ws_connection.authenticated is True
                
                # Verify response was sent
                mock_send.assert_called_once()
                sent_message = mock_send.call_args[0][1]
                assert sent_message["type"] == WSMessageType.AUTHENTICATED
                assert sent_message["user_id"] == "test-user-123"
    
    @pytest.mark.asyncio
    async def test_handle_subscribe(self, ws_connection):
        """Test subscription message handling"""
        # Authenticate first
        ws_connection.authenticated = True
        ws_connection.user_id = "test-user"
        
        with patch('services.proxy_service.src.api.websocket_gateway.stream_metrics') as mock_stream:
            with patch('services.proxy_service.src.api.websocket_gateway.send_message') as mock_send:
                message = {
                    "type": WSMessageType.SUBSCRIBE,
                    "metric_types": ["cpu", "memory"],
                    "interval": 5
                }
                
                await handle_message(ws_connection, message)
                
                # Verify subscription was created
                assert len(ws_connection.subscriptions) == 1
                
                # Verify streaming task was created
                assert mock_stream.called
                
                # Verify confirmation was sent
                mock_send.assert_called_once()
                sent_message = mock_send.call_args[0][1]
                assert sent_message["type"] == "subscribed"
    
    @pytest.mark.asyncio
    async def test_handle_request_metrics(self, ws_connection):
        """Test one-time metrics request"""
        ws_connection.authenticated = True
        
        mock_metrics = Mock()
        mock_metrics.cpu.usage_percent = 50.0
        
        with patch('services.proxy_service.src.api.websocket_gateway.grpc_client') as mock_grpc:
            # Mock gRPC response
            mock_stub = AsyncMock()
            mock_response = Mock()
            mock_response.metrics = mock_metrics
            mock_response.source = "fresh"
            mock_stub.GetSystemMetrics.return_value = mock_response
            mock_grpc.get_metrics_stub.return_value = mock_stub
            
            with patch('services.proxy_service.src.api.websocket_gateway.send_message') as mock_send:
                with patch('services.proxy_service.src.api.websocket_gateway.proto_to_dict') as mock_convert:
                    mock_convert.return_value = {"cpu": {"usage_percent": 50.0}}
                    
                    message = {
                        "type": WSMessageType.REQUEST_METRICS,
                        "metric_types": ["cpu"],
                        "request_id": "req-123"
                    }
                    
                    await handle_message(ws_connection, message)
                    
                    # Verify gRPC call
                    mock_stub.GetSystemMetrics.assert_called_once()
                    
                    # Verify response
                    mock_send.assert_called_once()
                    sent_message = mock_send.call_args[0][1]
                    assert sent_message["type"] == WSMessageType.METRICS_UPDATE
                    assert sent_message["request_id"] == "req-123"
                    assert sent_message["metrics"]["cpu"]["usage_percent"] == 50.0
    
    @pytest.mark.asyncio
    async def test_handle_ping(self, ws_connection):
        """Test ping message handling"""
        with patch('services.proxy_service.src.api.websocket_gateway.send_message') as mock_send:
            message = {"type": WSMessageType.PING}
            
            await handle_message(ws_connection, message)
            
            # Verify pong was sent
            mock_send.assert_called_once()
            sent_message = mock_send.call_args[0][1]
            assert sent_message["type"] == WSMessageType.PONG
    
    @pytest.mark.asyncio
    async def test_unauthenticated_request(self, ws_connection):
        """Test handling request without authentication"""
        ws_connection.authenticated = False
        
        with patch('services.proxy_service.src.api.websocket_gateway.send_error') as mock_error:
            message = {
                "type": WSMessageType.SUBSCRIBE,
                "metric_types": ["cpu"]
            }
            
            await handle_message(ws_connection, message)
            
            # Should send error
            mock_error.assert_called_once()
            error_args = mock_error.call_args[0]
            assert "Not authenticated" in error_args[1]


class TestProtoConversion:
    """Test protobuf conversion"""
    
    def test_proto_to_dict_cpu(self):
        """Test CPU metrics conversion"""
        proto = Mock()
        proto.cpu = Mock()
        proto.cpu.usage_percent = 75.5
        proto.cpu.per_core_percent = [70.0, 80.0]
        proto.cpu.frequency = Mock(current_mhz=2500.0, min_mhz=800.0, max_mhz=3600.0)
        proto.cpu.load_average = Mock(one_minute=1.2, five_minutes=1.5, fifteen_minutes=1.8)
        proto.cpu.count = 4
        proto.cpu.count_logical = 8
        
        # Mock HasField
        proto.HasField = Mock(side_effect=lambda field: field == 'cpu')
        proto.gpu = []
        
        result = proto_to_dict(proto)
        
        assert result['cpu']['usage_percent'] == 75.5
        assert result['cpu']['per_core_percent'] == [70.0, 80.0]
        assert result['cpu']['frequency']['current'] == 2500.0
        assert result['cpu']['load_average']['1min'] == 1.2
    
    def test_proto_to_dict_gpu(self):
        """Test GPU metrics conversion"""
        proto = Mock()
        
        # Mock GPU
        gpu_mock = Mock()
        gpu_mock.id = 0
        gpu_mock.name = "RTX 3080"
        gpu_mock.load_percent = 85.0
        gpu_mock.memory = Mock(
            total_bytes=10737418240,
            used_bytes=8589934592,
            free_bytes=2147483648,
            percent=80.0
        )
        gpu_mock.temperature_celsius = 70.0
        gpu_mock.uuid = "GPU-123"
        gpu_mock.driver_version = "530.30"
        
        proto.gpu = [gpu_mock]
        proto.HasField = Mock(return_value=False)
        
        result = proto_to_dict(proto)
        
        assert len(result['gpu']) == 1
        assert result['gpu'][0]['name'] == "RTX 3080"
        assert result['gpu'][0]['load_percent'] == 85.0
        assert result['gpu'][0]['memory']['percent'] == 80.0


class TestWebSocketIntegration:
    """Integration tests for WebSocket"""
    
    @pytest.mark.asyncio
    async def test_full_websocket_flow(self, mock_websocket, mock_token):
        """Test complete WebSocket flow"""
        # Setup mocks
        messages_received = []
        
        async def mock_receive():
            # Simulate client messages
            messages = [
                {"type": "authenticate", "token": mock_token},
                {"type": "subscribe", "metric_types": ["cpu"], "interval": 2},
                {"type": "ping"},
                WebSocketDisconnect()  # Disconnect after ping
            ]
            for msg in messages:
                if isinstance(msg, Exception):
                    raise msg
                yield msg
        
        mock_websocket.receive_json.side_effect = mock_receive()
        mock_websocket.send_json.side_effect = lambda msg: messages_received.append(msg)
        
        with patch('services.proxy_service.src.api.websocket_gateway.security_manager') as mock_security:
            mock_security.verify_token.return_value = {"user_id": "test-user", "type": "access"}
            mock_security.is_token_blacklisted.return_value = False
            mock_security.log_access = AsyncMock()
            
            with patch('services.proxy_service.src.api.websocket_gateway.stream_metrics'):
                # Run connection handler
                await handle_websocket_connection(mock_websocket)
        
        # Verify flow
        assert len(messages_received) >= 3
        
        # Check ready message
        assert messages_received[0]["type"] == "ready"
        
        # Check authenticated message
        auth_msg = next((m for m in messages_received if m["type"] == "authenticated"), None)
        assert auth_msg is not None
        
        # Check subscribed message
        sub_msg = next((m for m in messages_received if m["type"] == "subscribed"), None)
        assert sub_msg is not None