# services/proxy_service/force_load_model.py

#!/usr/bin/env python3
"""Force load a model in LLM service"""

import asyncio
import grpc
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)


async def force_load_model():
    """Force load a model using the Load API"""
    channel = grpc.aio.insecure_channel('localhost:50053')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    print("Force loading model in LLM service...")
    
    # Check if LLM service has LoadModel method
    # If not, we'll need to trigger loading differently
    
    # First, let's see what methods are available
    print("\n1. Checking available methods...")
    print("Available methods:", dir(stub))
    
    # Try to load by generating with specific model
    print("\n2. Attempting to load gpt2 by generating text...")
    gen_req = llm_service_pb2.GenerateRequest(
        metadata=common_pb2.RequestMetadata(
            request_id="force-load",
            user_id="test"
        ),
        prompt="Test",
        model_id="gpt2",  # Specify model explicitly
        config=llm_service_pb2.GenerationConfig(
            max_tokens=1
        )
    )
    
    try:
        print("   Sending request (this will load the model)...")
        print("   Please wait, first-time loading can take 30-60 seconds...")
        
        # Use a very long timeout for first load
        response = await stub.Generate(gen_req, timeout=120)
        
        print(f"\n✅ Success! Model loaded and generated: '{response.text}'")
        print(f"   Model ID: {response.model_id}")
        
        # Now list models to confirm
        print("\n3. Confirming model is loaded...")
        list_req = llm_service_pb2.ListModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="confirm-load",
                user_id="test"
            )
        )
        
        list_resp = await stub.ListModels(list_req, timeout=5)
        print(f"   Models loaded: {len(list_resp.models)}")
        for model in list_resp.models:
            print(f"   - {model.model_id}")
            
    except grpc.RpcError as e:
        print(f"\n❌ Failed: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    await channel.close()


if __name__ == "__main__":
    asyncio.run(force_load_model())