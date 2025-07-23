# services/proxy_service/test_integration.py

#!/usr/bin/env python3
"""Integration test for Proxy Service with LLM Service"""

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


async def test_integration():
    """Test proxy service with real LLM backend"""
    channel = grpc.aio.insecure_channel('localhost:8080')
    stub = proxy_service_pb2_grpc.ProxyServiceStub(channel)
    
    print("Testing Proxy Service Integration with LLM Service...")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        health_response = await stub.HealthCheck(proxy_service_pb2.HealthCheckRequest())
        print(f"✅ Proxy Status: {health_response.status}")
        for service, status in health_response.services.items():
            print(f"   - {service}: {status}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: List Models
    print("\n2. Testing List Models...")
    try:
        list_request = proxy_service_pb2.ListModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-list-integration",
                user_id="test-user"
            )
        )
        list_response = await stub.ListModels(list_request)
        print(f"✅ Found {len(list_response.models)} models from LLM service:")
        for model in list_response.models:
            print(f"   - {model.name} ({model.status}) - Backend: {model.backend}")
    except Exception as e:
        print(f"❌ List models failed: {e}")
    
    # Test 3: Generate Text with Real Model
    print("\n3. Testing Real Text Generation...")
    prompts = [
        ("The future of AI is", None),  # Use default model
        ("Once upon a time", "gpt2"),   # Specify model
        ("Python programming is", "gpt2")
    ]
    
    for prompt, model in prompts:
        try:
            print(f"\n   Prompt: '{prompt}' | Model: {model or 'default'}")
            gen_request = proxy_service_pb2.GenerateTextRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id=f"test-gen-{hash(prompt)}",
                    user_id="test-user"
                ),
                model_name=model if model else "",
                prompt=prompt,
                max_tokens=30,
                temperature=0.7
            )
            
            gen_response = await stub.GenerateText(gen_request)
            print(f"   ✅ Generated: {gen_response.generated_text[:100]}...")
            print(f"   Model used: {gen_response.model_used}")
            print(f"   Tokens: {gen_response.token_usage.total_tokens} "
                  f"(prompt: {gen_response.token_usage.prompt_tokens}, "
                  f"completion: {gen_response.token_usage.completion_tokens})")
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
    
    # Test 4: Test Error Handling
    print("\n4. Testing Error Handling...")
    try:
        # Try with non-existent model
        gen_request = proxy_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-error",
                user_id="test-user"
            ),
            model_name="non-existent-model",
            prompt="This should fail",
            max_tokens=10
        )
        gen_response = await stub.GenerateText(gen_request)
        print(f"Response: {gen_response.generated_text}")
    except grpc.RpcError as e:
        print(f"✅ Error handling works: {e.code()} - {e.details()}")
    
    await channel.close()
    print("\n" + "=" * 60)
    print("✅ Integration tests completed!")


if __name__ == "__main__":
    asyncio.run(test_integration())
    