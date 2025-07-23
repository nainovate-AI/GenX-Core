# services/proxy_service/load_model_test.py

#!/usr/bin/env python3
"""Load a model in LLM service for testing"""

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


async def load_model():
    """Load a model in the LLM service"""
    channel = grpc.aio.insecure_channel('localhost:50053')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    print("Loading model in LLM service...")
    
    # First, list models to see what's available
    list_req = llm_service_pb2.ListModelsRequest(
        metadata=common_pb2.RequestMetadata(
            request_id="load-test-list",
            user_id="test"
        )
    )
    
    list_resp = await stub.ListModels(list_req)
    print(f"Currently loaded models: {len(list_resp.models)}")
    for model in list_resp.models:
        print(f"  - {model.model_id}")
    
    # Try to generate text to trigger model loading
    print("\nTriggering model load by generating text...")
    gen_req = llm_service_pb2.GenerateRequest(
        metadata=common_pb2.RequestMetadata(
            request_id="load-test-gen",
            user_id="test"
        ),
        prompt="Hello world",
        config=llm_service_pb2.GenerationConfig(
            max_tokens=10
        )
    )
    
    try:
        gen_resp = await stub.Generate(gen_req)
        print(f"✅ Model loaded and generated: {gen_resp.text}")
        print(f"Model used: {gen_resp.model_id}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    await channel.close()


if __name__ == "__main__":
    asyncio.run(load_model())  