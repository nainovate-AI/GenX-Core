#!/usr/bin/env python3
"""
Test TGI Backend with Model Info and Loading
"""
import asyncio
import sys
import os
import json
from pathlib import Path
import aiohttp

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Force TGI backend
os.environ['BACKEND_TYPE'] = 'tgi'
os.environ['TGI_SERVER_URL'] = 'http://localhost:8080'
os.environ['TGI_EXTERNAL_SERVER'] = 'true'  # Assume server is already running


async def check_tgi_server_status(server_url: str = "http://localhost:8080"):
    """Check TGI server status and get model info"""
    print("\n=== TGI Server Status ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Check health endpoint
            async with session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    print("✅ TGI Server is healthy")
                else:
                    print(f"❌ TGI Server health check failed: {response.status}")
                    return None
            
            # Get server info
            async with session.get(f"{server_url}/info") as response:
                if response.status == 200:
                    info = await response.json()
                    print("\n📊 TGI Server Information:")
                    print(f"   Model ID: {info.get('model_id', 'Unknown')}")
                    print(f"   Model Type: {info.get('model_type', 'Unknown')}")
                    print(f"   Model SHA: {info.get('sha', 'Unknown')[:8]}...")
                    print(f"   Docker Label: {info.get('docker_label', 'Unknown')}")
                    print(f"   Max Concurrent Requests: {info.get('max_concurrent_requests', 'Unknown')}")
                    print(f"   Max Input Length: {info.get('max_input_length', 'Unknown')}")
                    print(f"   Max Total Tokens: {info.get('max_total_tokens', 'Unknown')}")
                    print(f"   Max Batch Total Tokens: {info.get('max_batch_total_tokens', 'Unknown')}")
                    print(f"   Waiting Served Ratio: {info.get('waiting_served_ratio', 'Unknown')}")
                    print(f"   Max Waiting Tokens: {info.get('max_waiting_tokens', 'Unknown')}")
                    return info
                else:
                    print(f"❌ Failed to get server info: {response.status}")
                    return None
                    
        except aiohttp.ClientError as e:
            print(f"❌ Cannot connect to TGI server at {server_url}: {e}")
            return None


async def test_model_loading_simulation():
    """
    Simulate model loading process (TGI loads model at startup)
    This shows how to verify the loaded model
    """
    print("\n=== Model Loading Verification ===")
    
    server_url = "http://localhost:8080"
    expected_model = "gpt2"  # The model we expect TGI to have loaded
    
    async with aiohttp.ClientSession() as session:
        try:
            # Get current loaded model info
            async with session.get(f"{server_url}/info") as response:
                if response.status == 200:
                    info = await response.json()
                    loaded_model = info.get('model_id', '')
                    
                    if expected_model in loaded_model:
                        print(f"✅ Expected model '{expected_model}' is loaded")
                    else:
                        print(f"⚠️  Different model loaded: {loaded_model}")
                        print(f"   Expected: {expected_model}")
                        print("\n   Note: TGI loads a single model at startup.")
                        print("   To change models, restart TGI with different --model-id")
                    
                    return info
                    
        except Exception as e:
            print(f"❌ Error checking loaded model: {e}")
            return None


async def test_tgi_backend():
    """Test TGI backend with comprehensive model info"""
    backend = None
    try:
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        print("🚀 Starting TGI Backend Test...")
        print("=" * 60)
        
        # Step 1: Check server status and get model info
        server_info = await check_tgi_server_status()
        if not server_info:
            print("\n❌ TGI server is not running!")
            print("\nTo start TGI server with gpt2:")
            print("docker run -d --name tgi-server -p 8080:80 \\")
            print("  ghcr.io/huggingface/text-generation-inference:latest \\")
            print("  --model-id gpt2 --max-input-length 1024")
            return
        
        # Step 2: Verify model loading
        model_info = await test_model_loading_simulation()
        
        # Step 3: Create TGI backend with the loaded model
        print("\n=== TGI Backend Initialization ===")
        
        # Use the actual loaded model from server info
        loaded_model_id = server_info.get('model_id', 'gpt2')
        print(f"Creating TGI backend for model: {loaded_model_id}")
        
        backend = TGIBackend(
            loaded_model_id,
            server_url="http://localhost:8080",
            external_server=True
        )
        
        # Initialize backend
        print("Initializing backend...")
        success = await backend.initialize()
        if not success:
            print("❌ Failed to initialize TGI backend!")
            return
        
        print("✅ TGI backend initialized successfully")
        
        # Step 4: Test model capabilities
        print("\n=== Model Capabilities Test ===")
        
        # Get backend model info
        backend_model_info = backend.get_model_info()
        if backend_model_info:
            print(f"Model ID: {backend_model_info.model_id}")
            print(f"Provider: {backend_model_info.provider}")
            print(f"Capabilities: {', '.join(backend_model_info.capabilities)}")
        
        # Step 5: Test generation with different prompts
        print("\n=== Generation Tests ===")
        
        test_prompts = [
            {
                "prompt": "The quick brown fox",
                "config": GenerationConfig(max_tokens=20, temperature=0.5),
                "description": "Complete a well-known phrase"
            },
            {
                "prompt": "def fibonacci(n):",
                "config": GenerationConfig(max_tokens=50, temperature=0.2),
                "description": "Generate Python code"
            },
            {
                "prompt": "Once upon a time in a distant galaxy",
                "config": GenerationConfig(max_tokens=40, temperature=0.8),
                "description": "Creative story generation"
            }
        ]
        
        for i, test in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {test['description']}")
            print(f"Prompt: '{test['prompt']}'")
            
            result = await backend.generate(
                prompt=test['prompt'],
                config=test['config']
            )
            
            print(f"Generated: {result['text'][:100]}...")
            print(f"Tokens used: {result['tokens_used']['total_tokens']}")
            print(f"Finish reason: {result['finish_reason']}")
        
        # Step 6: Test streaming with token counting
        print("\n=== Streaming Test with Token Counting ===")
        print("Prompt: 'The future of artificial intelligence'")
        print("Streaming response: ", end="", flush=True)
        
        token_count = 0
        start_time = asyncio.get_event_loop().time()
        
        async for chunk in backend.stream_generate(
            prompt="The future of artificial intelligence",
            config=GenerationConfig(max_tokens=50, temperature=0.7)
        ):
            print(chunk, end="", flush=True)
            token_count += 1
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        print(f"\n\nStreaming stats:")
        print(f"  Tokens generated: {token_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Tokens/second: {token_count/duration:.1f}")
        
        # Step 7: Test prompt validation
        print("\n=== Prompt Validation Test ===")
        
        # Test with different prompt lengths
        test_lengths = [10, 100, 1000]
        for length in test_lengths:
            test_prompt = "word " * length
            validation = await backend.validate_prompt(test_prompt)
            print(f"Prompt length {length} words: Valid={validation['is_valid']}, "
                  f"Token count={validation.get('token_count', 'N/A')}")
        
        # Step 8: Final health check
        print("\n=== Final Health Check ===")
        health = await backend.health_check()
        print(f"Backend status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
        
        print("\n✅ All TGI backend tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if backend:
            await backend.cleanup()
            print("\n🧹 Backend cleaned up")


async def test_model_switching_info():
    """
    Show how to work with different models in TGI
    (Note: TGI requires restart to switch models)
    """
    print("\n=== Model Switching Information ===")
    print("TGI loads a single model at startup and cannot switch models dynamically.")
    print("\nTo use different models with TGI:")
    print("1. Stop the current TGI container:")
    print("   docker stop tgi-server")
    print("\n2. Start TGI with a different model:")
    print("   # Example with Mistral-7B:")
    print("   docker run -d --name tgi-mistral --gpus all -p 8081:80 \\")
    print("     ghcr.io/huggingface/text-generation-inference:latest \\")
    print("     --model-id mistralai/Mistral-7B-v0.1")
    print("\n3. Run multiple TGI instances on different ports for multiple models")


async def main():
    """Main test runner"""
    # Run comprehensive TGI backend tests
    await test_tgi_backend()
    
    # Show model switching information
    await test_model_switching_info()
    
    print("\n" + "="*60)
    print("🎉 TGI Backend Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())