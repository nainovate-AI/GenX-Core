#!/usr/bin/env python3
"""
Test TGI Backend with Multi-Instance Support
"""
import asyncio
import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Force TGI backend
os.environ['BACKEND_TYPE'] = 'tgi'


async def test_multi_model_tgi():
    """Test loading multiple models with TGI"""
    backends = {}
    
    try:
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        print("🚀 Testing Multi-Model TGI Support")
        print("=" * 60)
        
        # Test models - start with small ones
        test_models = [
            "gpt2",
            "distilgpt2",
            # Add more models as needed
        ]
        
        # Step 1: Load multiple models
        print("\n=== Loading Multiple Models ===")
        for model_id in test_models:
            print(f"\n📦 Loading {model_id}...")
            backend = TGIBackend(model_id)
            
            success = await backend.initialize()
            if success:
                backends[model_id] = backend
                print(f"✅ {model_id} loaded successfully")
                
                # Get instance info
                health = await backend.health_check()
                print(f"   Port: {health.get('port', 'N/A')}")
                print(f"   Status: {health.get('status', 'N/A')}")
            else:
                print(f"❌ Failed to load {model_id}")
        
        # Step 2: Show all running instances
        print("\n=== All TGI Instances ===")
        all_instances = await TGIBackend.get_all_instances_info()
        for model_id, info in all_instances.items():
            print(f"\n{model_id}:")
            print(f"  Port: {info['port']}")
            print(f"  URL: {info['server_url']}")
            print(f"  Container: {info['container_name']}")
            print(f"  Ready: {info['is_ready']}")
            print(f"  Uptime: {info['uptime']:.1f}s")
        
        # Step 3: Test generation with each model
        print("\n=== Testing Generation with Each Model ===")
        prompt = "The future of AI is"
        
        for model_id, backend in backends.items():
            print(f"\n🤖 Generating with {model_id}:")
            
            result = await backend.generate(
                prompt=prompt,
                config=GenerationConfig(max_tokens=30, temperature=0.7)
            )
            
            print(f"Response: {result['text']}")
            print(f"Tokens: {result['tokens_used']['total_tokens']}")
        
        # Step 4: Test concurrent generation
        print("\n=== Testing Concurrent Generation ===")
        
        async def generate_concurrent(backend, model_id, prompt_num):
            prompt = f"Example {prompt_num}: Tell me about"
            result = await backend.generate(
                prompt=prompt,
                config=GenerationConfig(max_tokens=20, temperature=0.5)
            )
            return f"{model_id}: {result['text'][:50]}..."
        
        # Generate concurrently with all models
        tasks = []
        for i, (model_id, backend) in enumerate(backends.items()):
            tasks.append(generate_concurrent(backend, model_id, i))
        
        results = await asyncio.gather(*tasks)
        for result in results:
            print(f"  {result}")
        
        print("\n✅ Multi-model TGI test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup backends
        for backend in backends.values():
            await backend.cleanup()
        
        # Show final state
        print("\n=== Final Instance State ===")
        try:
            all_instances = await TGIBackend.get_all_instances_info()
            print(f"Active instances: {len(all_instances)}")
            for model_id in all_instances:
                print(f"  - {model_id}")
        except:
            pass


async def test_instance_reuse():
    """Test that instances are reused when same model is requested"""
    print("\n" + "="*60)
    print("🔄 Testing Instance Reuse")
    print("="*60)
    
    from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
    
    # Create first backend
    backend1 = TGIBackend("gpt2")
    await backend1.initialize()
    health1 = await backend1.health_check()
    port1 = health1.get('port')
    
    print(f"First instance on port: {port1}")
    
    # Create second backend with same model
    backend2 = TGIBackend("gpt2")
    await backend2.initialize()
    health2 = await backend2.health_check()
    port2 = health2.get('port')
    
    print(f"Second instance on port: {port2}")
    
    if port1 == port2:
        print("✅ Instance reused correctly!")
    else:
        print("❌ New instance created instead of reusing!")
    
    # Cleanup
    await backend1.cleanup()
    await backend2.cleanup()


async def main():
    """Run all tests"""
    # Test multi-model support
    await test_multi_model_tgi()
    
    # Test instance reuse
    await test_instance_reuse()
    
    print("\n" + "="*60)
    print("🎉 All TGI Multi-Instance Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())