#!/usr/bin/env python3
"""
Test MLX Backend
"""
import asyncio
import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Force MLX backend
os.environ['BACKEND_TYPE'] = 'mlx'
os.environ['DEFAULT_MODEL_ID'] = 'mlx-community/gpt2-mlx'


async def test_mlx_backend():
    """Test MLX backend directly"""
    try:
        from genx_components.microservices.llm.src.backends.mlx_backend import MLXBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        print("Testing MLX Backend...")
        print("-" * 50)
        
        # Create backend
        print("1. Creating MLX backend...")
        backend = MLXBackend("phi-4")  # Will map to mlx-community/gpt2-mlx
        
        # Initialize
        print("2. Initializing model...")
        success = await backend.initialize()
        if not success:
            print("Failed to initialize!")
            return
        
        # Check health
        print("\n3. Health check:")
        health = await backend.health_check()
        for key, value in health.items():
            print(f"   {key}: {value}")
        
        # Test generation
        print("\n4. Testing generation...")
        config = GenerationConfig(
            max_tokens=30,
            temperature=0.7
        )
        
        result = await backend.generate(
            prompt="The future of AI is",
            config=config
        )
        
        print(f"   Generated: {result['text']}")
        print(f"   Tokens: {result['tokens_used']}")
        
        # Test streaming
        print("\n5. Testing streaming...")
        print("   Streaming: ", end="", flush=True)
        
        async for chunk in backend.stream_generate(
            prompt="Once upon a time",
            config=GenerationConfig(max_tokens=20)
        ):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Cleanup
        await backend.cleanup()
        print("\n✅ MLX backend test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mlx_backend())