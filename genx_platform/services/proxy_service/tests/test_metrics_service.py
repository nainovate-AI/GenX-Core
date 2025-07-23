# services/proxy_service/tests/test_metrics_service.py
"""
Test suite for Metrics gRPC Service
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from ..src.generated import metrics_service_pb2
from ..src.generated import metrics_service_pb2_grpc
from ..src.grpc_services.metrics_service import MetricsService
from ..src.collectors.metrics_collector import MetricsCollector


@pytest.fixture
async def metrics_service():
    """Create metrics service instance"""
    service = MetricsService()
    await service.initialize()
    yield service
    # Cleanup
    if service.collector._collection_task:
        await service.collector.shutdown()


@pytest.fixture
def mock_metrics_data():
    """Mock metrics data for testing"""
    return {
        'cpu': {
            'usage_percent': 45.5,
            'per_core_percent': [40.0, 50.0, 45.0, 48.0],
            'frequency': {
                'current': 2400.0,
                'min': 800.0,
                'max': 3600.0
            },
            'load_average': {
                '1min': 1.5,
                '5min': 1.8,
                '15min': 2.0
            },
            'count': 4,
            'count_logical': 8
        },
        'memory': {
            'total': 17179869184,  # 16GB
            'available': 8589934592,  # 8GB
            'used': 8589934592,  # 8GB
            'free': 4294967296,  # 4GB
            'percent': 50.0,
            'swap': {
                'total': 2147483648,  # 2GB
                'used': 0,
                'free': 2147483648,
                'percent': 0.0
            }
        },
        'gpu': [{
            'id': 0,
            'name': 'NVIDIA GeForce RTX 3080',
            'load': 75.0,
            'memory': {
                'total': 10240.0,  # MB
                'used': 7680.0,
                'free': 2560.0,
                'percent': 75.0
            },
            'temperature': 65.0,
            'uuid': 'GPU-12345678',
            'driver': '530.30.02'
        }],
        'disk': {
            'usage': {
                'total': 1073741824000,  # 1TB
                'used': 536870912000,  # 500GB
                'free': 536870912000,  # 500GB
                'percent': 50.0
            },
            'io': {
                'read_count': 1000000,
                'write_count': 500000,
                'read_bytes': 10737418240,
                'write_bytes': 5368709120
            },
            'partitions': [{
                'device': '/dev/sda1',
                'mountpoint': '/',
                'fstype': 'ext4',
                'total': 1073741824000,
                'used': 536870912000,
                'free': 536870912000,
                'percent': 50.0
            }],
            'model_storage': {
                'path': '/models',
                'total': 536870912000,  # 500GB
                'used': 107374182400,  # 100GB
                'free': 429496729600,  # 400GB
                'percent': 20.0
            }
        },
        'network': {
            'io': {
                'bytes_sent': 1073741824,
                'bytes_recv': 2147483648,
                'packets_sent': 1000000,
                'packets_recv': 2000000,
                'errin': 0,
                'errout': 0,
                'dropin': 0,
                'dropout': 0
            },
            'connections': {
                'ESTABLISHED': 50,
                'TIME_WAIT': 10,
                'CLOSE_WAIT': 5
            }
        }
    }


class TestMetricsService:
    """Test cases for MetricsService"""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_all(self, metrics_service, mock_metrics_data):
        """Test getting all system metrics"""
        # Mock the collector
        with patch.object(
            metrics_service.collector, 
            'collect_metrics',
            new_callable=AsyncMock,
            return_value=mock_metrics_data
        ):
            # Create request
            request = metrics_service_pb2.GetSystemMetricsRequest(
                metric_types=[metrics_service_pb2.METRIC_TYPE_ALL],
                force_refresh=True,
                request_id="test-request-1"
            )
            
            # Mock context
            context = Mock()
            
            # Call service
            response = await metrics_service.GetSystemMetrics(request, context)
            
            # Assertions
            assert response.source == "fresh"
            assert response.metrics.cpu.usage_percent == 45.5
            assert len(response.metrics.cpu.per_core_percent) == 4
            assert response.metrics.memory.percent == 50.0
            assert len(response.metrics.gpu) == 1
            assert response.metrics.gpu[0].name == 'NVIDIA GeForce RTX 3080'
            assert response.metrics.disk.usage.percent == 50.0
            assert response.metrics.network.io.bytes_sent == 1073741824
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_specific(self, metrics_service, mock_metrics_data):
        """Test getting specific metrics only"""
        # Mock the collector to return only CPU data
        cpu_only_data = {'cpu': mock_metrics_data['cpu']}
        
        with patch.object(
            metrics_service.collector,
            'collect_metrics',
            new_callable=AsyncMock,
            return_value=cpu_only_data
        ):
            # Request only CPU metrics
            request = metrics_service_pb2.GetSystemMetricsRequest(
                metric_types=[metrics_service_pb2.METRIC_TYPE_CPU],
                force_refresh=True,
                request_id="test-request-2"
            )
            
            context = Mock()
            response = await metrics_service.GetSystemMetrics(request, context)
            
            # Should have CPU data
            assert response.metrics.cpu.usage_percent == 45.5
            
            # Should not have other metrics
            assert not response.metrics.HasField('memory')
            assert len(response.metrics.gpu) == 0
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_cached(self, metrics_service, mock_metrics_data):
        """Test getting cached metrics"""
        # Mock cached metrics
        with patch.object(
            metrics_service.collector,
            'get_cached_metrics',
            new_callable=AsyncMock,
            return_value=mock_metrics_data
        ):
            # Request without force refresh
            request = metrics_service_pb2.GetSystemMetricsRequest(
                metric_types=[metrics_service_pb2.METRIC_TYPE_ALL],
                force_refresh=False,
                request_id="test-request-3"
            )
            
            context = Mock()
            response = await metrics_service.GetSystemMetrics(request, context)
            
            # Should return cached data
            assert response.source == "cache"
            assert response.metrics.cpu.usage_percent == 45.5
    
    @pytest.mark.asyncio
    async def test_stream_system_metrics(self, metrics_service, mock_metrics_data):
        """Test streaming metrics"""
        # Mock the collector
        call_count = 0
        
        async def mock_collect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return slightly different data each time
            data = mock_metrics_data.copy()
            data['cpu']['usage_percent'] = 45.5 + call_count
            return data
        
        with patch.object(
            metrics_service.collector,
            'collect_metrics',
            new_callable=AsyncMock,
            side_effect=mock_collect
        ):
            # Create request
            request = metrics_service_pb2.StreamSystemMetricsRequest(
                interval_seconds=1,
                metric_types=[metrics_service_pb2.METRIC_TYPE_CPU],
                request_id="test-stream-1"
            )
            
            # Mock context
            context = Mock()
            context.is_active.return_value = True
            
            # Collect streamed updates
            updates = []
            stream = metrics_service.StreamSystemMetrics(request, context)
            
            # Get first 3 updates
            async for update in stream:
                updates.append(update)
                if len(updates) >= 3:
                    context.is_active.return_value = False
            
            # Verify updates
            assert len(updates) >= 3
            
            # First should be INITIAL
            assert updates[0].type == metrics_service_pb2.SystemMetricsUpdate.INITIAL
            assert updates[0].metrics.cpu.usage_percent == 46.5
            
            # Rest should be PERIODIC
            assert updates[1].type == metrics_service_pb2.SystemMetricsUpdate.PERIODIC
            assert updates[1].metrics.cpu.usage_percent == 47.5
    
    @pytest.mark.asyncio
    async def test_get_metrics_history(self, metrics_service, mock_metrics_data):
        """Test getting historical metrics"""
        # Mock historical data
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        historical_data = [
            {
                'timestamp': start_time + timedelta(minutes=i*10),
                'metrics': mock_metrics_data
            }
            for i in range(6)  # 6 data points over 1 hour
        ]
        
        with patch.object(
            metrics_service.collector,
            'get_historical_metrics',
            new_callable=AsyncMock,
            return_value=historical_data
        ):
            # Create request
            request = metrics_service_pb2.GetMetricsHistoryRequest(
                metric_types=[metrics_service_pb2.METRIC_TYPE_ALL],
                resolution_seconds=600,  # 10 minutes
                request_id="test-history-1"
            )
            request.start_time.FromDatetime(start_time)
            request.end_time.FromDatetime(end_time)
            
            context = Mock()
            response = await metrics_service.GetMetricsHistory(request, context)
            
            # Verify response
            assert len(response.data_points) == 6
            assert response.metadata.count == 6
            assert response.metadata.resolution_seconds == 600
            
            # Check first data point
            first_point = response.data_points[0]
            assert first_point.metrics.cpu.usage_percent == 45.5
    
    @pytest.mark.asyncio
    async def test_refresh_metrics(self, metrics_service, mock_metrics_data):
        """Test force refresh metrics"""
        with patch.object(
            metrics_service.collector,
            'collect_metrics',
            new_callable=AsyncMock,
            return_value=mock_metrics_data
        ):
            # Create request
            request = metrics_service_pb2.RefreshMetricsRequest(
                request_id="test-refresh-1"
            )
            
            context = Mock()
            response = await metrics_service.RefreshMetrics(request, context)
            
            # Verify response
            assert response.success is True
            assert response.metrics.cpu.usage_percent == 45.5
            
            # Verify force refresh was called
            metrics_service.collector.collect_metrics.assert_called_with(
                [metrics_service_pb2.METRIC_TYPE_ALL],
                force_refresh=True
            )
    
    @pytest.mark.asyncio
    async def test_error_handling(self, metrics_service):
        """Test error handling in service"""
        # Mock collector to raise exception
        with patch.object(
            metrics_service.collector,
            'collect_metrics',
            new_callable=AsyncMock,
            side_effect=Exception("Test error")
        ):
            request = metrics_service_pb2.GetSystemMetricsRequest(
                force_refresh=True,
                request_id="test-error-1"
            )
            
            context = Mock()
            
            # Should call abort on context
            await metrics_service.GetSystemMetrics(request, context)
            context.abort.assert_called_once()
            
            # Check abort was called with INTERNAL status
            abort_call = context.abort.call_args
            assert abort_call[0][0] == grpc.StatusCode.INTERNAL
            assert "Test error" in abort_call[0][1]
    
    @pytest.mark.asyncio
    async def test_history_validation(self, metrics_service):
        """Test metrics history validation"""
        # Test invalid time range (end before start)
        request = metrics_service_pb2.GetMetricsHistoryRequest(
            request_id="test-validation-1"
        )
        request.start_time.FromDatetime(datetime.utcnow())
        request.end_time.FromDatetime(datetime.utcnow() - timedelta(hours=1))
        
        context = Mock()
        await metrics_service.GetMetricsHistory(request, context)
        
        # Should abort with INVALID_ARGUMENT
        context.abort.assert_called_once()
        abort_call = context.abort.call_args
        assert abort_call[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert "End time must be after start time" in abort_call[0][1]
    
    @pytest.mark.asyncio
    async def test_proto_conversion(self, metrics_service, mock_metrics_data):
        """Test conversion from dict to protobuf"""
        proto_metrics = metrics_service._convert_to_proto(mock_metrics_data)
        
        # CPU assertions
        assert proto_metrics.cpu.usage_percent == 45.5
        assert list(proto_metrics.cpu.per_core_percent) == [40.0, 50.0, 45.0, 48.0]
        assert proto_metrics.cpu.frequency.current_mhz == 2400.0
        assert proto_metrics.cpu.load_average.one_minute == 1.5
        
        # Memory assertions
        assert proto_metrics.memory.total_bytes == 17179869184
        assert proto_metrics.memory.percent == 50.0
        assert proto_metrics.memory.swap.total_bytes == 2147483648
        
        # GPU assertions
        assert len(proto_metrics.gpu) == 1
        gpu = proto_metrics.gpu[0]
        assert gpu.id == 0
        assert gpu.name == 'NVIDIA GeForce RTX 3080'
        assert gpu.load_percent == 75.0
        assert gpu.memory.total_bytes == 10737418240  # Converted from MB to bytes
        
        # Disk assertions
        assert proto_metrics.disk.usage.percent == 50.0
        assert len(proto_metrics.disk.partitions) == 1
        assert proto_metrics.disk.model_storage.percent == 20.0
        
        # Network assertions
        assert proto_metrics.network.io.bytes_sent == 1073741824
        assert proto_metrics.network.connection_states['ESTABLISHED'] == 50


@pytest.fixture
async def grpc_server():
    """Create and start gRPC server for integration tests"""
    server = grpc.aio.server()
    service = MetricsService()
    await service.initialize()
    
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(service, server)
    
    port = server.add_insecure_port('[::]:0')
    await server.start()
    
    yield f'localhost:{port}', service
    
    await server.stop(grace=5)
    await service.collector.shutdown()


class TestMetricsServiceIntegration:
    """Integration tests for Metrics Service"""
    
    @pytest.mark.asyncio
    async def test_grpc_get_metrics(self, grpc_server):
        """Test gRPC call to get metrics"""
        server_address, service = grpc_server
        
        # Mock the collector
        with patch.object(
            service.collector,
            'collect_metrics',
            new_callable=AsyncMock,
            return_value={'cpu': {'usage_percent': 55.0}}
        ):
            async with grpc.aio.insecure_channel(server_address) as channel:
                stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
                
                request = metrics_service_pb2.GetSystemMetricsRequest(
                    metric_types=[metrics_service_pb2.METRIC_TYPE_CPU],
                    force_refresh=True
                )
                
                response = await stub.GetSystemMetrics(request)
                
                assert response.source == "fresh"
                assert response.metrics.cpu.usage_percent == 55.0