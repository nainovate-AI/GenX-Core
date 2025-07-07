#!/usr/bin/env python3
"""
Enhanced test client for LLM service with new APIs
Demonstrates dynamic model loading and text generation
"""
import asyncio
import sys
import os
import grpc
import time
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)


async def test_model_management():
    """Test model loading and management APIs"""
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    print("\n" + "="*60)
    print("Testing Model Management APIs")
    print("="*60)
    
    try:
        # Test 1: Load a model with specific backend and device
        print("\n1. Loading GPT-2 with Transformers backend on CPU...")
        load_request = llm_service_pb2.LoadModelRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="load-test-1",
                user_id="test-user"
            ),
            model_name="gpt2",
            backend="transformers",
            device="cpu",
            options=llm_service_pb2.ModelLoadingOptions(
                load_in_8bit=False,
                trust_remote_code=False
            )
        )
        
        load_response = await stub.LoadModel(load_request)
        if load_response.success:
            print(f"✅ Model loaded successfully!")
            print(f"   Model ID: {load_response.model_id}")
            print(f"   Backend: {load_response.model_info.backend}")
            print(f"   Device: {load_response.model_info.device}")
            gpt2_model_id = load_response.model_id
        else:
            print(f"❌ Failed to load model: {load_response.error.message}")
            return
        
        # Test 2: Try loading another model
        print("\n2. Loading DistilGPT-2 with auto device selection...")
        load_request2 = llm_service_pb2.LoadModelRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="load-test-2",
                user_id="test-user"
            ),
            model_name="distilgpt2",
            backend="transformers",
            device="auto"  # Let system choose best device
        )
        
        load_response2 = await stub.LoadModel(load_request2)
        if load_response2.success:
            print(f"✅ Model loaded successfully!")
            print(f"   Model ID: {load_response2.model_id}")
            print(f"   Device: {load_response2.model_info.device}")
            distilgpt2_model_id = load_response2.model_id
        
        # Test 3: Get loaded models
        print("\n3. Getting list of loaded models...")
        get_models_request = llm_service_pb2.GetLoadedModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="get-models-1",
                user_id="test-user"
            ),
            include_stats=True
        )
        
        models_response = await stub.GetLoadedModels(get_models_request)
        print(f"✅ Found {len(models_response.models)} loaded models:")
        
        for model in models_response.models:
            print(f"\n   Model: {model.model_name}")
            print(f"   - ID: {model.model_id}")
            print(f"   - Backend: {model.backend}")
            print(f"   - Device: {model.device}")
            print(f"   - Status: Loaded={model.status.is_loaded}, Available={model.status.is_available}")
            if model.HasField('stats'):
                print(f"   - Requests: {model.stats.total_requests}")
                print(f"   - Tokens Generated: {model.stats.total_tokens_generated}")
            if model.HasField('capabilities'):
                print(f"   - Max Context: {model.capabilities.max_context_length}")
                print(f"   - Features: {', '.join(model.capabilities.features)}")
        
        # System info
        print("\n   System Resources:")
        print(f"   - CPU Cores: {models_response.system_info.cpu.cores}")
        print(f"   - CPU Usage: {models_response.system_info.cpu.utilization_percent:.1f}%")
        print(f"   - RAM: {models_response.system_info.memory.used_ram_mb}MB / {models_response.system_info.memory.total_ram_mb}MB")
        
        if models_response.system_info.gpus:
            print(f"   - GPUs: {len(models_response.system_info.gpus)}")
            for gpu in models_response.system_info.gpus:
                print(f"     * {gpu.name}: {gpu.used_memory_mb}MB / {gpu.total_memory_mb}MB")
        
        return gpt2_model_id
        
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


async def test_text_generation(model_id: str):
    """Test text generation APIs"""
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    print("\n" + "="*60)
    print("Testing Text Generation APIs")
    print("="*60)
    
    try:
        user_id = "test-user-123"
        
        # Test 1: Simple generation without history
        print("\n1. Testing simple text generation...")
        generate_request = llm_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="gen-test-1",
                user_id=user_id
            ),
            user_id=user_id,
            prompt="What is machine learning?",
            prompt_config=llm_service_pb2.PromptConfig(
                system_prompt="You are a helpful AI assistant. Explain things clearly and concisely."
            ),
            use_history=False,
            streaming=False,
            model_id=model_id,
            generation_config=llm_service_pb2.TextGenerationConfig(
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        print(f"Prompt: {generate_request.prompt}")
        start_time = time.time()
        
        response = await stub.GenerateText(generate_request)
        
        print(f"\n✅ Generated Response:")
        print(f"   {response.generated_text}")
        print(f"\n   Model: {response.model_id}")
        print(f"   Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
        print(f"   Time: {response.stats.total_time_ms}ms")
        print(f"   Speed: {response.stats.tokens_per_second:.1f} tokens/sec")
        
        # Test 2: Generation with history
        print("\n2. Testing generation with conversation history...")
        
        # First message
        generate_request2 = llm_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="gen-test-2",
                user_id=user_id
            ),
            user_id=user_id,
            prompt="My name is Alice and I work in healthcare.",
            use_history=True,  # Enable history
            streaming=False,
            model_id=model_id,
            generation_config=llm_service_pb2.TextGenerationConfig(
                max_tokens=50,
                temperature=0.7
            )
        )
        
        response2 = await stub.GenerateText(generate_request2)
        print(f"User: {generate_request2.prompt}")
        print(f"Assistant: {response2.generated_text}")
        
        # Follow-up with history
        generate_request3 = llm_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="gen-test-3",
                user_id=user_id
            ),
            user_id=user_id,
            prompt="What's my name and profession?",
            use_history=True,  # Will remember previous conversation
            streaming=False,
            model_id=model_id,
            generation_config=llm_service_pb2.TextGenerationConfig(
                max_tokens=50,
                temperature=0.7
            )
        )
        
        response3 = await stub.GenerateText(generate_request3)
        print(f"\nUser: {generate_request3.prompt}")
        print(f"Assistant: {response3.generated_text}")
        print("(Should remember Alice and healthcare from history)")
        
        # Test 3: Streaming generation
        print("\n3. Testing streaming text generation...")
        stream_request = llm_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="stream-test-1",
                user_id=user_id
            ),
            user_id=user_id,
            prompt="Write a short story about a robot learning to paint.",
            prompt_config=llm_service_pb2.PromptConfig(
                system_prompt="You are a creative storyteller.",
                format_instructions="Keep it under 100 words."
            ),
            use_history=False,
            streaming=True,  # Request streaming
            model_id=model_id,
            generation_config=llm_service_pb2.TextGenerationConfig(
                max_tokens=150,
                temperature=0.8,
                top_p=0.95
            )
        )
        
        print(f"Prompt: {stream_request.prompt}")
        print("Streaming response: ", end="", flush=True)
        
        total_tokens = 0
        async for response in stub.StreamGenerateText(stream_request):
            if response.HasField('metadata'):
                # First message with metadata
                continue
            elif response.text_chunk:
                print(response.text_chunk, end="", flush=True)
            elif response.is_final:
                print(f"\n\n✅ Streaming complete!")
                print(f"   Total tokens: {response.usage.total_tokens}")
                print(f"   Speed: {response.stats.tokens_per_second:.1f} tokens/sec")
        
        # Test 4: Using prompt examples (few-shot)
        print("\n4. Testing with examples (few-shot learning)...")
        few_shot_request = llm_service_pb2.GenerateTextRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="few-shot-1",
                user_id=user_id
            ),
            user_id=user_id,
            prompt="The weather is beautiful today.",
            prompt_config=llm_service_pb2.PromptConfig(
                system_prompt="Translate English to French.",
                examples=[
                    llm_service_pb2.Example(
                        input="Hello, how are you?",
                        output="Bonjour, comment allez-vous?"
                    ),
                    llm_service_pb2.Example(
                        input="Thank you very much.",
                        output="Merci beaucoup."
                    )
                ]
            ),
            use_history=False,
            streaming=False,
            model_id=model_id,
            generation_config=llm_service_pb2.TextGenerationConfig(
                max_tokens=30,
                temperature=0.3  # Lower temperature for translation
            )
        )
        
        response4 = await stub.GenerateText(few_shot_request)
        print(f"English: {few_shot_request.prompt}")
        print(f"French: {response4.generated_text}")
        
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_model_cleanup(model_id: str):
    """Test model unloading"""
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    print("\n" + "="*60)
    print("Testing Model Cleanup")
    print("="*60)
    
    try:
        print(f"\nUnloading model {model_id}...")
        unload_request = llm_service_pb2.UnloadModelRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="unload-1",
                user_id="test-user"
            ),
            model_id=model_id
        )
        
        unload_response = await stub.UnloadModel(unload_request)
        if unload_response.success:
            print("✅ Model unloaded successfully!")
        else:
            print(f"❌ Failed to unload: {unload_response.error.message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Enhanced LLM Service Test Client")
    print("="*60)
    print("Make sure the LLM service is running on localhost:50051")
    
    # Wait for service to be ready
    await asyncio.sleep(2)
    
    # Test model management
    model_id = await test_model_management()
    
    if model_id:
        # Test text generation with the loaded model
        await test_text_generation(model_id)
        
        # Clean up
        await test_model_cleanup(model_id)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())