#!/usr/bin/env python3
"""
Comprehensive test for TGI multi-instance support
"""
import asyncio
import sys
import os
from pathlib import Path
import docker
import time

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Set environment
os.environ['BACKEND_TYPE'] = 'tgi'


class TGIMultiInstanceTester:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.test_results = []
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        print(f"\n{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def check_containers(self, expected_count: int = None):
        """Check running TGI containers"""
        containers = self.docker_client.containers.list(
            filters={"name": "tgi-"}
        )
        print(f"\n📦 Running TGI containers: {len(containers)}")
        for container in containers:
            print(f"   - {container.name} (Status: {container.status})")
        
        if expected_count is not None:
            return len(containers) == expected_count
        return containers
    
    async def test_single_model_loading(self):
        """Test 1: Load a single model (or reuse existing)"""
        print("\n" + "="*60)
        print("Test 1: Single Model Loading/Reuse")
        print("="*60)
        
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        backend = None
        try:
            # Check initial state
            initial_containers = self.check_containers()
            initial_count = len(initial_containers)
            
            # Check if gpt2 is already running
            gpt2_running = any("gpt2" in c.name for c in initial_containers)
            
            # Load GPT-2
            print(f"\n🔄 Loading gpt2... (existing: {gpt2_running})")
            backend = TGIBackend("gpt2")
            success = await backend.initialize()
            
            if not success:
                self.log_result("Single Model Loading/Reuse", False, "Failed to initialize")
                return None
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Check container state
            current_containers = self.check_containers()
            current_count = len(current_containers)
            
            # Test generation
            result = await backend.generate(
                prompt="Hello world",
                config=GenerationConfig(max_tokens=10)
            )
            
            if result and result.get('text'):
                if gpt2_running:
                    # Should reuse existing container
                    if current_count == initial_count:
                        self.log_result("Single Model Loading/Reuse", True, 
                                      f"Reused existing container. Generated: {result['text'][:30]}...")
                    else:
                        self.log_result("Single Model Loading/Reuse", False, 
                                      f"Expected reuse but container count changed: {initial_count} -> {current_count}")
                else:
                    # Should create new container
                    if current_count == initial_count + 1:
                        self.log_result("Single Model Loading/Reuse", True, 
                                      f"Created new container. Generated: {result['text'][:30]}...")
                    else:
                        self.log_result("Single Model Loading/Reuse", False, 
                                      f"Expected new container but count: {initial_count} -> {current_count}")
                return backend
            else:
                self.log_result("Single Model Loading/Reuse", False, "Generation failed")
                return None
                
        except Exception as e:
            self.log_result("Single Model Loading/Reuse", False, str(e))
            return None
        finally:
            if backend:
                await backend.cleanup()
    
    async def test_instance_reuse(self):
        """Test 2: Verify instance reuse for same model"""
        print("\n" + "="*60)
        print("Test 2: Instance Reuse")
        print("="*60)
        
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        
        backend1 = None
        backend2 = None
        try:
            # Create first backend
            print("\n🔄 Creating first backend for gpt2...")
            backend1 = TGIBackend("gpt2")
            await backend1.initialize()
            health1 = await backend1.health_check()
            port1 = health1.get('port', 0)
            
            # Count containers
            container_count = len(self.check_containers())
            
            # Create second backend with same model
            print("\n🔄 Creating second backend for same model (gpt2)...")
            backend2 = TGIBackend("gpt2")
            await backend2.initialize()
            health2 = await backend2.health_check()
            port2 = health2.get('port', 0)
            
            # Check no new container created
            new_container_count = len(self.check_containers())
            
            if port1 == port2 and container_count == new_container_count:
                self.log_result("Instance Reuse", True, 
                              f"Same port ({port1}) and container count ({container_count})")
            else:
                self.log_result("Instance Reuse", False, 
                              f"Port1: {port1}, Port2: {port2}, "
                              f"Containers: {container_count} -> {new_container_count}")
                
        except Exception as e:
            self.log_result("Instance Reuse", False, str(e))
        finally:
            if backend1:
                await backend1.cleanup()
            if backend2:
                await backend2.cleanup()
    
    async def test_multiple_models(self):
        """Test 3: Load multiple different models"""
        print("\n" + "="*60)
        print("Test 3: Multiple Models")
        print("="*60)
        
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        backends = {}
        # Use models that are more likely to exist
        models = ["gpt2", "gpt2-medium"]  # Both should work with TGI
        
        try:
            initial_containers = len(self.check_containers())
            
            # Count how many are already running
            existing_models = 0
            containers = self.docker_client.containers.list(filters={"name": "tgi-"})
            for model in models:
                if any(model in c.name for c in containers):
                    existing_models += 1
            
            # Load multiple models
            for model_id in models:
                print(f"\n🔄 Loading {model_id}...")
                backend = TGIBackend(model_id)
                success = await backend.initialize()
                
                if success:
                    backends[model_id] = backend
                    health = await backend.health_check()
                    print(f"   ✅ Loaded on port {health.get('port', 'unknown')}")
                else:
                    print(f"   ❌ Failed to load {model_id}")
            
            # Calculate expected containers
            expected_new = len(models) - existing_models
            expected_total = initial_containers + expected_new
            actual_containers = len(self.check_containers())
            
            if len(backends) == len(models):
                # Test generation with each
                all_generated = True
                for model_id, backend in backends.items():
                    result = await backend.generate(
                        prompt="Test",
                        config=GenerationConfig(max_tokens=5)
                    )
                    if not result or not result.get('text'):
                        all_generated = False
                        break
                
                if all_generated:
                    self.log_result("Multiple Models", True,
                                  f"Loaded {len(backends)} models successfully")
                else:
                    self.log_result("Multiple Models", False,
                                  "Models loaded but generation failed")
            else:
                self.log_result("Multiple Models", False,
                              f"Only loaded {len(backends)} of {len(models)} models")
                
        except Exception as e:
            self.log_result("Multiple Models", False, str(e))
        finally:
            # Cleanup
            for backend in backends.values():
                await backend.cleanup()
    
    async def test_concurrent_requests(self):
        """Test 4: Concurrent requests to different models"""
        print("\n" + "="*60)
        print("Test 4: Concurrent Requests")
        print("="*60)
        
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        from genx_components.microservices.llm.src.backends.base import GenerationConfig
        
        backend1 = None
        backend2 = None
        try:
            # Use same model for both to ensure it works
            backend1 = TGIBackend("gpt2")
            await backend1.initialize()
            
            # Use the same backend for concurrent requests
            backend2 = backend1  # Reuse same instance
            
            # Define concurrent generation tasks
            async def generate_text(backend, model_name, index):
                start = time.time()
                try:
                    result = await backend.generate(
                        prompt=f"Test prompt {index}:",
                        config=GenerationConfig(max_tokens=20)
                    )
                    duration = time.time() - start
                    return {
                        "model": model_name,
                        "index": index,
                        "text": result['text'][:50] if result else "Failed",
                        "duration": duration
                    }
                except Exception as e:
                    return e
            
            # Run concurrent requests
            print("\n🔄 Sending 10 concurrent requests...")
            tasks = []
            for i in range(10):
                tasks.append(generate_text(backend1, "gpt2", i))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, dict))
            failed = sum(1 for r in results if isinstance(r, Exception))
            
            print(f"\n📊 Results: {successful} successful, {failed} failed")
            
            if successful == len(tasks):
                avg_duration = sum(r['duration'] for r in results if isinstance(r, dict)) / successful
                self.log_result("Concurrent Requests", True,
                              f"All {len(tasks)} requests succeeded, avg time: {avg_duration:.2f}s")
            else:
                self.log_result("Concurrent Requests", False,
                              f"{failed} requests failed")
                
        except Exception as e:
            self.log_result("Concurrent Requests", False, str(e))
        finally:
            if backend1 and backend1 != backend2:
                await backend1.cleanup()
            if backend2:
                await backend2.cleanup()
    
    async def test_instance_info(self):
        """Test 5: Get instance information"""
        print("\n" + "="*60)
        print("Test 5: Instance Information")
        print("="*60)
        
        from genx_components.microservices.llm.src.backends.tgi_backend import TGIBackend
        
        backend = None
        try:
            # Load a model
            backend = TGIBackend("gpt2")
            await backend.initialize()
            
            # Get all instances info
            instances = await TGIBackend.get_all_instances_info()
            
            print("\n📊 All TGI Instances:")
            for model_id, info in instances.items():
                print(f"\n{model_id}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            
            if instances and len(instances) > 0:
                self.log_result("Instance Information", True,
                              f"Found {len(instances)} instances")
            else:
                self.log_result("Instance Information", False,
                              "No instances found")
                
        except Exception as e:
            self.log_result("Instance Information", False, str(e))
        finally:
            if backend:
                await backend.cleanup()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {result['test']}")
            if result['details'] and not result['passed']:
                print(f"   → {result['details']}")
        
        return passed == total


async def main():
    """Run all tests"""
    print("🚀 TGI Multi-Instance Test Suite")
    print("="*60)
    
    tester = TGIMultiInstanceTester()
    
    # Show initial state
    print("\n📊 Initial Docker State:")
    tester.check_containers()
    
    # Run tests
    await tester.test_single_model_loading()
    await tester.test_instance_reuse()
    await tester.test_multiple_models()
    await tester.test_concurrent_requests()
    await tester.test_instance_info()
    
    # Show final state
    print("\n📊 Final Docker State:")
    tester.check_containers()
    
    # Print summary
    all_passed = tester.print_summary()
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")
    
    # Cleanup option
    print("\n🧹 Cleanup Option:")
    print("To stop all TGI containers, run:")
    print("docker ps -a | grep tgi- | awk '{print $1}' | xargs -r docker stop")
    print("docker ps -a | grep tgi- | awk '{print $1}' | xargs -r docker rm")


if __name__ == "__main__":
    asyncio.run(main())