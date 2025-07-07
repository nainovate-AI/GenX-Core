# genx_platform/genx_components/microservices/llm/tests/test_tgi_backend.py
#!/usr/bin/env python3
"""
Test TGI Backend
"""
import asyncio
import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Force TGI backend
os.environ['BACKEND_TYPE'] = 'tgi'
os.environ['TGI_SERVER_URL'] = 'http://localhost:8080'
os.environ['TGI_EXTERNAL_SERVER'] = 'true'  # Assume server is already running


async def test_tgi_backend():
    """Test TGI backend directly"""
    try:
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        print("Testing TGI Backend...")
        print("-" * 50)
        
        # Create backend
        print("1. Creating TGI backend...")
        backend = TGIBackend(
            "mistralai/Mistral-7B-Instruct-v0.2",
            server_url="http://localhost:8080",
            external_server=True
        )
        
        # Initialize
        print("2. Initializing backend...")
        success = await backend.initialize()
        if not success:
            print("Failed to initialize! Make sure TGI server is running.")
            print("Start it with: docker-compose -f docker-compose.tgi.yml up -d")
            return
        
        # Check health
        print("\n3. Health check:")
        health = await backend.health_check()
        for key, value in health.items():
            print(f"   {key}: {value}")
        
        # Test generation
        print("\n4. Testing generation...")
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        result = await backend.generate(
            prompt="Explain quantum computing in simple terms:",
            config=config
        )
        
        print(f"   Generated: {result['text']}")
        print(f"   Tokens: {result['tokens_used']}")
        print(f"   Finish reason: {result['finish_reason']}")
        
        # Test streaming
        print("\n5. Testing streaming...")
        print("   Streaming: ", end="", flush=True)
        
        async for chunk in backend.stream_generate(
            prompt="Write a haiku about artificial intelligence:",
            config=GenerationConfig(max_tokens=50, temperature=0.8)
        ):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Cleanup
        await backend.cleanup()
        print("\n✅ TGI backend test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tgi_backend())