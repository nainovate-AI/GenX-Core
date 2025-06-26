#!/usr/bin/env python3
"""
genx_platform/genx_components/microservices/llm/test_client.py
Test client for LLM service
"""
import asyncio
import sys
import os
import grpc
import time
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)


async def test_llm_service():
    """Test the LLM service"""
    # Connect to the service
    channel = grpc.aio.insecure_channel('localhost:50053')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    try:
        print("Testing LLM Service...")
        print("-" * 50)
        
        # Test 1: List models
        print("\n1. Testing ListModels...")
        list_request = llm_service_pb2.ListModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-list-1",
                user_id="test-user"
            )
        )
        
        list_response = await stub.ListModels(list_request)
        print(f"Available models: {[model.model_id for model in list_response.models]}")
        print(f"Default model: {list_response.default_model_id}")
        
        # Test 2: Get model info
        print("\n2. Testing GetModel...")
        get_request = llm_service_pb2.GetModelRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-get-1",
                user_id="test-user"
            ),
            model_id="distilgpt2"  # or whatever model you're using
        )
        
        try:
            get_response = await stub.GetModel(get_request)
            if get_response.model:
                print(f"Model: {get_response.model.model_id}")
                print(f"Provider: {get_response.model.provider}")
                print(f"Status: Loaded={get_response.status.is_loaded}, Available={get_response.status.is_available}")
        except grpc.RpcError as e:
            print(f"GetModel error: {e.details()}")
        
        # Test 3: Validate prompt
        print("\n3. Testing ValidatePrompt...")
        validate_request = llm_service_pb2.ValidatePromptRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-validate-1",
                user_id="test-user"
            ),
            prompt="Hello, how are you?",
            model_id="distilgpt2",
            count_tokens=True
        )
        
        try:
            validate_response = await stub.ValidatePrompt(validate_request)
            print(f"Valid: {validate_response.is_valid}")
            if validate_response.HasField('token_count'):
                print(f"Token count: {validate_response.token_count}")
        except grpc.RpcError as e:
            print(f"ValidatePrompt error: {e.details()}")
        
        # Test 4: Generate text
        print("\n4. Testing Generate...")
        generate_request = llm_service_pb2.GenerateRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-generate-1",
                user_id="test-user"
            ),
            prompt="Once upon a time, in a land far away",
            model_id="distilgpt2",
            config=llm_service_pb2.GenerationConfig(
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        print("Sending generation request...")
        print(f"Prompt: '{generate_request.prompt}'")
        start_time = time.time()
        
        try:
            generate_response = await stub.Generate(generate_request)
            duration = time.time() - start_time
            
            print(f"\nGenerated text: {generate_response.text}")
            print(f"Model used: {generate_response.model_id}")
            print(f"Tokens used: prompt={generate_response.usage.prompt_tokens}, "
                  f"completion={generate_response.usage.completion_tokens}, "
                  f"total={generate_response.usage.total_tokens}")
            print(f"Generation time: {duration:.2f}s")
            print(f"Finish reason: {generate_response.finish_reason}")
        except grpc.RpcError as e:
            print(f"Generate error: {e.details()}")
        
        # Test 5: Stream generation
        print("\n5. Testing StreamGenerate...")
        stream_request = llm_service_pb2.GenerateRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-stream-1",
                user_id="test-user"
            ),
            prompt="The future of artificial intelligence is",
            model_id="distilgpt2",
            config=llm_service_pb2.GenerationConfig(
                max_tokens=30,
                temperature=0.8
            )
        )
        
        print(f"Prompt: '{stream_request.prompt}'")
        print("Streaming response: ", end="", flush=True)
        
        try:
            token_count = 0
            async for response in stub.StreamGenerate(stream_request):
                if response.delta:
                    print(response.delta, end="", flush=True)
                    token_count += 1
                elif response.is_final and response.HasField('usage'):
                    print(f"\n\nStreaming complete!")
                    print(f"Total tokens: {response.usage.total_tokens}")
                    print(f"Finish reason: {response.finish_reason if response.HasField('finish_reason') else 'N/A'}")
        except grpc.RpcError as e:
            print(f"\nStreamGenerate error: {e.details()}")
        
        # Test 6: Test with system prompt
        print("\n6. Testing with system prompt...")
        system_request = llm_service_pb2.GenerateRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-system-1",
                user_id="test-user"
            ),
            prompt="What is machine learning?",
            system_prompt="You are a helpful AI assistant. Explain things simply.",
            model_id="distilgpt2",
            config=llm_service_pb2.GenerationConfig(
                max_tokens=50,
                temperature=0.7
            )
        )
        
        try:
            system_response = await stub.Generate(system_request)
            print(f"Response: {system_response.text}")
        except grpc.RpcError as e:
            print(f"System prompt test error: {e.details()}")
        
        # Test 7: Test with conversation history
        print("\n7. Testing with conversation history...")
        conversation_request = llm_service_pb2.GenerateRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="test-conversation-1",
                user_id="test-user"
            ),
            prompt="What about Python?",
            model_id="distilgpt2",
            messages=[
                llm_service_pb2.Message(
                    role="user",
                    content="What are the best programming languages?"
                ),
                llm_service_pb2.Message(
                    role="assistant",
                    content="Some popular programming languages are JavaScript, Python, and Java."
                )
            ],
            config=llm_service_pb2.GenerationConfig(
                max_tokens=50,
                temperature=0.7
            )
        )
        
        try:
            conversation_response = await stub.Generate(conversation_request)
            print(f"Response: {conversation_response.text}")
        except grpc.RpcError as e:
            print(f"Conversation test error: {e.details()}")
        
        print("\n" + "="*50)
        print("✅ All tests completed!")
        
    except grpc.RpcError as e:
        print(f"\n❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await channel.close()


if __name__ == "__main__":
    print("LLM Service Test Client")
    print("=" * 50)
    print("Make sure the LLM service is running on localhost:50053")
    print("=" * 50)
    
    asyncio.run(test_llm_service())