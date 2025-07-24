#!/usr/bin/env python3
"""
Test client for Metrics Service
Tests all gRPC endpoints
"""
import asyncio
import grpc
import sys
import os
from datetime import datetime, timedelta

# Add genx_platform to path
current_file = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file)
metrics_root = os.path.dirname(scripts_dir)
microservices_dir = os.path.dirname(metrics_root)
genx_components = os.path.dirname(microservices_dir)
genx_platform = os.path.dirname(genx_components)
sys.path.insert(0, genx_platform)

# Import generated protobuf files
from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import metrics_service_pb2, metrics_service_pb2_grpc


class MetricsServiceTester:
    def __init__(self, host="localhost", port=50056):
        self.address = f"{host}:{port}"
        print(f"Connecting to Metrics Service at {self.address}")
    
    async def test_get_system_metrics(self):
        """Test GetSystemMetrics RPC"""
        print("\n=== Testing GetSystemMetrics ===")
        
        async with grpc.aio.insecure_channel(self.address) as channel:
            stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
            
            # Create request
            request = metrics_service_pb2.GetSystemMetricsRequest()
            request.metadata.request_id = "test-001"
            request.metadata.user_id = "test-user"
            request.metadata.timestamp.GetCurrentTime()
            
            # Test 1: Get all metrics
            print("Test 1: Getting all metrics...")
            request.metric_types.append(metrics_service_pb2.METRIC_TYPE_ALL)
            request.force_refresh = False
            
            try:
                response = await stub.GetSystemMetrics(request)
                print(f"✓ Success! Source: {response.source}")
                print(f"  CPU Usage: {response.metrics.cpu.usage_percent:.1f}%")
                print(f"  Memory Usage: {response.metrics.memory.percent:.1f}%")
                print(f"  Disk Usage: {response.metrics.disk.usage.percent:.1f}%")
                
                if response.metrics.gpu:
                    print(f"  GPU Count: {len(response.metrics.gpu)}")
                    for gpu in response.metrics.gpu:
                        print(f"    GPU {gpu.id}: {gpu.load_percent:.1f}% ({gpu.name})")
                else:
                    print("  GPU: Not available")
                    
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
            
            # Test 2: Force refresh
            print("\nTest 2: Force refresh...")
            request.force_refresh = True
            
            try:
                response = await stub.GetSystemMetrics(request)
                print(f"✓ Success! Source: {response.source}")
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
            
            # Test 3: Get specific metrics
            print("\nTest 3: Getting CPU and Memory only...")
            request.ClearField('metric_types')
            request.metric_types.extend([
                metrics_service_pb2.METRIC_TYPE_CPU,
                metrics_service_pb2.METRIC_TYPE_MEMORY
            ])
            request.force_refresh = False
            
            try:
                response = await stub.GetSystemMetrics(request)
                print(f"✓ Success!")
                print(f"  CPU Load Average: {response.metrics.cpu.load_average.one_minute:.2f}")
                print(f"  Memory Available: {response.metrics.memory.available_bytes / (1024**3):.1f} GB")
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
        
        return True
    
    async def test_stream_system_metrics(self):
        """Test StreamSystemMetrics RPC"""
        print("\n=== Testing StreamSystemMetrics ===")
        
        async with grpc.aio.insecure_channel(self.address) as channel:
            stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
            
            # Create request
            request = metrics_service_pb2.StreamSystemMetricsRequest()
            request.metadata.request_id = "test-stream-001"
            request.metadata.user_id = "test-user"
            request.metadata.timestamp.GetCurrentTime()
            
            request.interval_seconds = 2  # Update every 2 seconds
            request.max_duration_seconds = 10  # Stream for 10 seconds
            request.metric_types.append(metrics_service_pb2.METRIC_TYPE_ALL)
            request.include_alerts = True
            
            print(f"Streaming metrics for {request.max_duration_seconds} seconds...")
            
            try:
                update_count = 0
                async for update in stub.StreamSystemMetrics(request):
                    update_count += 1
                    print(f"\nUpdate {update_count} - Type: {update.type}")
                    print(f"  CPU: {update.metrics.cpu.usage_percent:.1f}%")
                    print(f"  Memory: {update.metrics.memory.percent:.1f}%")
                    
                    if update.alerts:
                        print("  Alerts:")
                        for alert in update.alerts:
                            print(f"    - {alert.severity}: {alert.message}")
                
                print(f"\n✓ Stream completed successfully with {update_count} updates")
                return True
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
    
    async def test_refresh_metrics(self):
        """Test RefreshMetrics RPC"""
        print("\n=== Testing RefreshMetrics ===")
        
        async with grpc.aio.insecure_channel(self.address) as channel:
            stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
            
            # Create request
            request = metrics_service_pb2.RefreshMetricsRequest()
            request.metadata.request_id = "test-refresh-001"
            request.metadata.user_id = "test-user"
            request.metadata.timestamp.GetCurrentTime()
            
            try:
                response = await stub.RefreshMetrics(request)
                if response.success:
                    print("✓ Metrics refreshed successfully!")
                    print(f"  CPU Usage: {response.metrics.cpu.usage_percent:.1f}%")
                    print(f"  Memory Usage: {response.metrics.memory.percent:.1f}%")
                else:
                    print("✗ Refresh failed")
                    if response.error:
                        print(f"  Error: {response.error.message}")
                
                return response.success
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
    
    async def test_get_resource_summary(self):
        """Test GetResourceSummary RPC"""
        print("\n=== Testing GetResourceSummary ===")
        
        async with grpc.aio.insecure_channel(self.address) as channel:
            stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
            
            # Create request
            request = metrics_service_pb2.GetResourceSummaryRequest()
            request.metadata.request_id = "test-summary-001"
            request.metadata.user_id = "test-user"
            request.metadata.timestamp.GetCurrentTime()
            request.include_trend = True
            
            try:
                response = await stub.GetResourceSummary(request)
                print("✓ Resource summary retrieved!")
                
                # Print status for each resource
                for resource_type in ['cpu', 'memory', 'gpu', 'disk']:
                    status = getattr(response, f"{resource_type}_status", None)
                    if status and status.status:  # Check if field exists and is set
                        status_name = metrics_service_pb2.ResourceStatus.HealthStatus.Name(status.status)
                        print(f"  {resource_type.upper()}: {status_name} - {status.usage_percent:.1f}%")
                
                # Overall health
                health_name = metrics_service_pb2.SystemHealth.Name(response.overall_health)
                print(f"  Overall Health: {health_name}")
                
                return True
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
    
    async def test_health_check(self):
        """Test gRPC Health Check"""
        print("\n=== Testing Health Check ===")
        
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        
        async with grpc.aio.insecure_channel(self.address) as channel:
            health_stub = health_pb2_grpc.HealthStub(channel)
            
            # Check overall health
            request = health_pb2.HealthCheckRequest()
            
            try:
                response = await health_stub.Check(request)
                status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
                print(f"✓ Service health: {status}")
                
                # Check specific service
                request.service = "genx.metrics.v1.MetricsService"
                response = await health_stub.Check(request)
                status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
                print(f"✓ MetricsService health: {status}")
                
                return True
                
            except grpc.RpcError as e:
                print(f"✗ Error: {e.code()} - {e.details()}")
                return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print(f"\n{'='*50}")
        print(f"Testing Metrics Service at {self.address}")
        print(f"{'='*50}")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Get System Metrics", self.test_get_system_metrics),
            ("Refresh Metrics", self.test_refresh_metrics),
            ("Get Resource Summary", self.test_get_resource_summary),
            ("Stream System Metrics", self.test_stream_system_metrics),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n✗ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print(f"\n{'='*50}")
        print("Test Summary:")
        print(f"{'='*50}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        return passed == total


async def main():
    # Parse command line arguments
    host = os.environ.get("METRICS_SERVICE_HOST", "localhost")
    port = int(os.environ.get("METRICS_SERVICE_PORT", "50056"))
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # Run tests
    tester = MetricsServiceTester(host, port)
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())