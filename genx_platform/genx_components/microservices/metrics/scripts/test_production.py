#!/usr/bin/env python3
# genx_platform/genx_components/microservices/metrics/scripts/test_production.py
"""
Test script to verify production deployment
"""
import grpc
import sys
import time
import asyncio
import os
from pathlib import Path

# Add genx_platform to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    metrics_service_pb2,
    metrics_service_pb2_grpc,
)


async def test_metrics_service():
    """Test metrics service functionality"""
    print("Testing Metrics Service...")
    
    # Determine if TLS is enabled
    tls_enabled = os.environ.get('GRPC_TLS_ENABLED', 'true').lower() == 'true'
    
    if tls_enabled and Path('/certs/ca.crt').exists():
        print("✓ Using TLS connection")
        # Create secure channel
        with open('/certs/ca.crt', 'rb') as f:
            ca_cert = f.read()
        with open('/certs/client.crt', 'rb') as f:
            client_cert = f.read()
        with open('/certs/client.key', 'rb') as f:
            client_key = f.read()
        
        credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )
        channel = grpc.aio.secure_channel('localhost:50056', credentials)
    else:
        print("✓ Using insecure connection (development mode)")
        channel = grpc.aio.insecure_channel('localhost:50056')
    
    try:
        # Test connection
        await channel.channel_ready()
        print("✓ Connected to metrics service")
        
        # Create stub
        stub = metrics_service_pb2_grpc.MetricsServiceStub(channel)
        
        # Prepare auth metadata if needed
        metadata = []
        if os.environ.get('ENABLE_AUTH', 'true').lower() == 'true':
            auth_token = os.environ.get('AUTH_TOKEN', 'default-token')
            metadata.append(('x-auth-token', auth_token))
            print("✓ Authentication configured")
        
        # Test GetSystemMetrics
        request = metrics_service_pb2.GetSystemMetricsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-prod-1",
                user_id="test-user"
            )
        )
        
        response = await stub.GetSystemMetrics(request, metadata=metadata)
        
        print("✓ GetSystemMetrics successful")
        print(f"  - CPU Usage: {response.metrics.cpu.usage_percent:.1f}%")
        print(f"  - Memory Usage: {response.metrics.memory.percent:.1f}%")
        print(f"  - Source: {response.source}")
        
        # Test streaming
        stream_request = metrics_service_pb2.StreamSystemMetricsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-stream-1",
                user_id="test-user"
            ),
            interval_seconds=2,
            max_duration_seconds=6
        )
        
        print("\n✓ Testing streaming metrics (6 seconds)...")
        update_count = 0
        async for update in stub.StreamSystemMetrics(stream_request, metadata=metadata):
            update_count += 1
            print(f"  - Update {update_count}: CPU={update.metrics.cpu.usage_percent:.1f}%")
            if update_count >= 3:
                break
        
        print(f"✓ Streaming successful - received {update_count} updates")
        
        # Test health check
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        health_stub = health_pb2_grpc.HealthStub(channel)
        health_response = await health_stub.Check(
            health_pb2.HealthCheckRequest(service="")
        )
        print(f"✓ Health check: {health_response.status}")
        
        return True
        
    except grpc.RpcError as e:
        print(f"✗ gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        await channel.close()


async def test_monitoring_stack():
    """Test monitoring components"""
    print("\nTesting Monitoring Stack...")
    
    import aiohttp
    
    tests = [
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("Grafana", "http://localhost:3001/api/health"),
        ("Jaeger", "http://localhost:16686/"),
        ("Consul", "http://localhost:8500/v1/status/leader"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for name, url in tests:
            try:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        print(f"✓ {name} is healthy")
                    else:
                        print(f"✗ {name} returned status {response.status}")
            except Exception as e:
                print(f"✗ {name} is not accessible: {e}")


async def main():
    """Run all tests"""
    print("=== GenX Metrics Service Production Tests ===\n")
    
    # Wait for services to be ready
    print("Waiting for services to start...")
    await asyncio.sleep(5)
    
    # Test metrics service
    metrics_ok = await test_metrics_service()
    
    # Test monitoring stack if available
    if os.environ.get('ENVIRONMENT') == 'production':
        await test_monitoring_stack()
    
    print("\n=== Test Summary ===")
    if metrics_ok:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)