# services/proxy_service/test_client.py

#!/usr/bin/env python3
"""Test client for Proxy Service"""

import asyncio
import grpc
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from genx_components.common.grpc import common_pb2
from services.proxy_service.src.grpc import (
    proxy_service_pb2,
    proxy_service_pb2_grpc
)


async def test_proxy_service():
    """Test the proxy service"""
    channel = grpc.aio.insecure_channel('localhost:8080')
    stub = proxy_service_pb2_grpc.ProxyServiceStub(channel)
    
    print("Testing Proxy Service...")
    print("-" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    health_response = await stub.HealthCheck(proxy_service_pb2.HealthCheckRequest())
    print(f"Status: {health_response.status}")
    print(f"Services: {dict(health_response.services)}")
    
    # Test 2: List Models
    print("\n2. Testing List Models...")
    list_request = proxy_service_pb2.ListModelsRequest(
        metadata=common_pb2.RequestMetadata(
            request_id="test-list-1",
            user_id="test-user"
        )
    )
    list_response = await stub.ListModels(list_request)
    print(f"Found {len(list_response.models)} models:")
    for model in list_response.models:
        print(f"  - {model.name} ({model.status}) - {model.backend}")
    
    # Test 3: Generate Text
    print("\n3. Testing Generate Text...")
    gen_request = proxy_service_pb2.GenerateTextRequest(
        metadata=common_pb2.RequestMetadata(
            request_id="test-gen-1",
            user_id="test-user"
        ),
        model_name="gpt2",
        prompt="Hello, world!",
        max_tokens=50
    )
    gen_response = await stub.GenerateText(gen_request)
    print(f"Generated: {gen_response.generated_text}")
    print(f"Model used: {gen_response.model_used}")
    print(f"Tokens: {gen_response.token_usage.total_tokens}")
    
    await channel.close()
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_proxy_service())