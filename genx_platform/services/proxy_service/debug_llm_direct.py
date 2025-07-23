# services/proxy_service/debug_llm_direct.py

#!/usr/bin/env python3
"""Debug LLM service directly"""

import asyncio
import grpc
import sys
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)


async def debug_llm():
    """Debug LLM service"""
    print("Debugging LLM service...")
    
    # First, let's check if the service is running
    try:
        channel = grpc.aio.insecure_channel('localhost:50053')
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        # Just check if we can connect
        print("1. Testing basic connectivity...")
        
        # Use GetModel to check a specific model
        get_model_req = llm_service_pb2.GetModelRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="debug-1",
                user_id="test"
            ),
            model_id="gpt2"
        )
        
        print("2. Checking model info...")
        try:
            model_resp = await stub.GetModel(get_model_req, timeout=5)
            if model_resp.HasField("model"):
                print(f"✅ Model found: {model_resp.model.model_id}")
            else:
                print("❌ Model not found")
        except Exception as e:
            print(f"GetModel error: {e}")
        
        # Try ValidatePrompt - this shouldn't trigger model loading
        print("\n3. Testing ValidatePrompt...")
        validate_req = llm_service_pb2.ValidatePromptRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="debug-2",
                user_id="test"
            ),
            prompt="Test prompt",
            count_tokens=True
        )
        
        try:
            validate_resp = await stub.ValidatePrompt(validate_req, timeout=5)
            print(f"Validate response: valid={validate_resp.is_valid}")
        except Exception as e:
            print(f"ValidatePrompt error: {e}")
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        if 'channel' in locals():
            await channel.close()


if __name__ == "__main__":
    asyncio.run(debug_llm())