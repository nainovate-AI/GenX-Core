#!/usr/bin/env python3
"""
Enhanced test client for LLM service with TGI multi-instance support
Tests complete flow: gRPC → LLM Service → Model Manager → TGI Backend → TGI Instances
"""
import asyncio
import sys
import os
import grpc
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)


class LLMServiceTester:
    def __init__(self, server_url: str = 'localhost:50051'):
        self.server_url = server_url
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
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
    
    async def test_service_health(self):
        """Test 1: Basic service health check"""
        print("\n" + "="*60)
        print("Test 1: Service Health Check")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # List models to check service is responding
            list_request = llm_service_pb2.ListModelsRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="health-check-1",
                    user_id="test-user"
                )
            )
            
            response = await stub.ListModels(list_request)
            if response.models:
                self.log_test("Service Health Check", True, 
                            f"Service is healthy, {len(response.models)} models available")
            else:
                self.log_test("Service Health Check", True, 
                            "Service is healthy, no models loaded yet")
                            
        except grpc.RpcError as e:
            self.log_test("Service Health Check", False, f"gRPC error: {e.details()}")
        finally:
            await channel.close()
    
    async def test_tgi_model_loading(self):
        """Test 2: Load models with TGI backend"""
        print("\n" + "="*60)
        print("Test 2: TGI Model Loading")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # Test loading GPT-2 with TGI
            print("\n🔄 Loading gpt2 with TGI backend...")
            load_request = llm_service_pb2.LoadModelRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="load-tgi-1",
                    user_id="test-user"
                ),
                model_name="gpt2",
                backend="tgi",
                device="auto"
            )
            
            load_response = await stub.LoadModel(load_request)
            
            if load_response.success:
                print(f"✅ Model loaded successfully!")
                print(f"   Model ID: {load_response.model_id}")
                print(f"   Backend: {load_response.model_info.backend}")
                print(f"   Port: {load_response.model_info.device}")  # TGI shows port info in device
                
                # Test generation with loaded model
                generate_request = llm_service_pb2.GenerateRequest(
                    metadata=common_pb2.RequestMetadata(
                        request_id="gen-tgi-1",
                        user_id="test-user"
                    ),
                    prompt="The future of AI is",
                    model_id=load_response.model_id,
                    config=llm_service_pb2.GenerationConfig(
                        max_tokens=20,
                        temperature=0.7
                    )
                )
                
                gen_response = await stub.Generate(generate_request)
                if gen_response.text:
                    self.log_test("TGI Model Loading", True, 
                                f"Model loaded and generated: {gen_response.text[:50]}...")
                else:
                    self.log_test("TGI Model Loading", False, "Model loaded but generation failed")
                    
                return load_response.model_id
            else:
                self.log_test("TGI Model Loading", False, 
                            f"Failed to load model: {load_response.error.message}")
                return None
                
        except grpc.RpcError as e:
            self.log_test("TGI Model Loading", False, f"gRPC error: {e.details()}")
            return None
        finally:
            await channel.close()
    
    async def test_tgi_instance_reuse(self):
        """Test 3: Test TGI instance reuse"""
        print("\n" + "="*60)
        print("Test 3: TGI Instance Reuse")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # Get initial loaded models
            initial_models = await self._get_loaded_models(stub)
            initial_count = len(initial_models)
            print(f"Initial models loaded: {initial_count}")
            
            # Try loading gpt2 again (should reuse)
            print("\n🔄 Loading gpt2 again (should reuse existing)...")
            load_request = llm_service_pb2.LoadModelRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="load-reuse-1",
                    user_id="test-user"
                ),
                model_name="gpt2",
                backend="tgi",
                device="auto"
            )
            
            load_response = await stub.LoadModel(load_request)
            
            # Get models after second load
            final_models = await self._get_loaded_models(stub)
            final_count = len(final_models)
            
            # Check if instance was reused
            if load_response.success and final_count == initial_count:
                # Find TGI instances
                tgi_instances = [m for m in final_models if m['backend'] == 'tgi']
                gpt2_instances = [m for m in tgi_instances if 'gpt2' in m['model_name']]
                
                if len(gpt2_instances) == 1:
                    self.log_test("TGI Instance Reuse", True, 
                                "Same instance reused, no new container created")
                else:
                    self.log_test("TGI Instance Reuse", False, 
                                f"Expected 1 gpt2 instance, found {len(gpt2_instances)}")
            else:
                self.log_test("TGI Instance Reuse", False, 
                            f"Model count changed: {initial_count} → {final_count}")
                            
        except Exception as e:
            self.log_test("TGI Instance Reuse", False, str(e))
        finally:
            await channel.close()
    
    async def test_multiple_tgi_models(self):
        """Test 4: Load multiple models with TGI"""
        print("\n" + "="*60)
        print("Test 4: Multiple TGI Models")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        loaded_models = []
        models_to_load = ["gpt2", "gpt2-medium"]  # Different models
        
        try:
            # Load multiple models
            for model_name in models_to_load:
                print(f"\n🔄 Loading {model_name} with TGI...")
                load_request = llm_service_pb2.LoadModelRequest(
                    metadata=common_pb2.RequestMetadata(
                        request_id=f"load-multi-{model_name}",
                        user_id="test-user"
                    ),
                    model_name=model_name,
                    backend="tgi",
                    device="auto"
                )
                
                load_response = await stub.LoadModel(load_request)
                
                if load_response.success:
                    loaded_models.append({
                        'model_id': load_response.model_id,
                        'model_name': model_name,
                        'port': load_response.model_info.device  # Port info
                    })
                    print(f"✅ {model_name} loaded as {load_response.model_id}")
                else:
                    print(f"❌ Failed to load {model_name}: {load_response.error.message}")
            
            # Test generation with each model
            successful_generations = 0
            for model_info in loaded_models:
                print(f"\n📝 Testing generation with {model_info['model_name']}...")
                
                gen_request = llm_service_pb2.GenerateRequest(
                    metadata=common_pb2.RequestMetadata(
                        request_id=f"gen-multi-{model_info['model_name']}",
                        user_id="test-user"
                    ),
                    prompt=f"Tell me about {model_info['model_name']}:",
                    model_id=model_info['model_id'],
                    config=llm_service_pb2.GenerationConfig(
                        max_tokens=20,
                        temperature=0.5
                    )
                )
                
                gen_response = await stub.Generate(gen_request)
                if gen_response.text:
                    successful_generations += 1
                    print(f"✅ Generated: {gen_response.text[:50]}...")
            
            # Check results
            if len(loaded_models) == len(models_to_load) and successful_generations == len(loaded_models):
                self.log_test("Multiple TGI Models", True, 
                            f"All {len(models_to_load)} models loaded and working")
            else:
                self.log_test("Multiple TGI Models", False, 
                            f"Loaded {len(loaded_models)}/{len(models_to_load)}, "
                            f"successful generations: {successful_generations}")
                            
        except Exception as e:
            self.log_test("Multiple TGI Models", False, str(e))
        finally:
            await channel.close()
    
    async def test_concurrent_tgi_requests(self):
        """Test 5: Concurrent requests to different TGI models"""
        print("\n" + "="*60)
        print("Test 5: Concurrent TGI Requests")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # Get loaded models
            loaded_models = await self._get_loaded_models(stub)
            tgi_models = [m for m in loaded_models if m['backend'] == 'tgi']
            
            if len(tgi_models) < 1:
                self.log_test("Concurrent TGI Requests", False, "No TGI models loaded")
                return
            
            # Create concurrent requests
            async def generate_concurrent(model_id: str, index: int):
                try:
                    gen_request = llm_service_pb2.GenerateRequest(
                        metadata=common_pb2.RequestMetadata(
                            request_id=f"concurrent-{index}",
                            user_id="test-user"
                        ),
                        prompt=f"Request {index}: Hello",
                        model_id=model_id,
                        config=llm_service_pb2.GenerationConfig(
                            max_tokens=10,
                            temperature=0.5
                        )
                    )
                    
                    start_time = time.time()
                    response = await stub.Generate(gen_request)
                    duration = time.time() - start_time
                    
                    return {
                        'success': bool(response.text),
                        'duration': duration,
                        'model_id': model_id
                    }
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            # Send concurrent requests
            print(f"\n🔄 Sending 20 concurrent requests across {len(tgi_models)} TGI models...")
            tasks = []
            for i in range(20):
                model = tgi_models[i % len(tgi_models)]
                tasks.append(generate_concurrent(model['model_id'], i))
            
            results = await asyncio.gather(*tasks)
            
            # Analyze results
            successful = sum(1 for r in results if r.get('success', False))
            avg_duration = sum(r.get('duration', 0) for r in results if r.get('success', False)) / max(successful, 1)
            
            print(f"\n📊 Results: {successful}/20 successful")
            print(f"   Average response time: {avg_duration:.2f}s")
            
            if successful == len(results):
                self.log_test("Concurrent TGI Requests", True, 
                            f"All requests successful, avg time: {avg_duration:.2f}s")
            else:
                self.log_test("Concurrent TGI Requests", False, 
                            f"Only {successful}/{len(results)} requests succeeded")
                            
        except Exception as e:
            self.log_test("Concurrent TGI Requests", False, str(e))
        finally:
            await channel.close()
    
    async def test_streaming_with_tgi(self):
        """Test 6: Streaming generation with TGI"""
        print("\n" + "="*60)
        print("Test 6: Streaming with TGI")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # Get a TGI model
            loaded_models = await self._get_loaded_models(stub)
            tgi_models = [m for m in loaded_models if m['backend'] == 'tgi']
            
            if not tgi_models:
                self.log_test("Streaming with TGI", False, "No TGI models loaded")
                return
            
            model_id = tgi_models[0]['model_id']
            
            # Test streaming
            print(f"\n🔄 Testing streaming with {model_id}...")
            stream_request = llm_service_pb2.GenerateRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="stream-tgi-1",
                    user_id="test-user"
                ),
                prompt="Write a short story about a robot:",
                model_id=model_id,
                config=llm_service_pb2.GenerationConfig(
                    max_tokens=50,
                    temperature=0.8
                )
            )
            
            print("Streaming response: ", end="", flush=True)
            token_count = 0
            start_time = time.time()
            
            async for response in stub.StreamGenerate(stream_request):
                if response.delta:
                    print(response.delta, end="", flush=True)
                    token_count += 1
            
            duration = time.time() - start_time
            print(f"\n\nTokens: {token_count}, Time: {duration:.2f}s, Speed: {token_count/duration:.1f} tokens/s")
            
            if token_count > 0:
                self.log_test("Streaming with TGI", True, 
                            f"Streamed {token_count} tokens at {token_count/duration:.1f} tokens/s")
            else:
                self.log_test("Streaming with TGI", False, "No tokens streamed")
                
        except Exception as e:
            self.log_test("Streaming with TGI", False, str(e))
        finally:
            await channel.close()
    
    async def test_model_unloading(self):
        """Test 7: Model unloading"""
        print("\n" + "="*60)
        print("Test 7: Model Unloading")
        print("="*60)
        
        channel = grpc.aio.insecure_channel(self.server_url)
        stub = llm_service_pb2_grpc.LLMServiceStub(channel)
        
        try:
            # Get current models
            loaded_models = await self._get_loaded_models(stub)
            if not loaded_models:
                self.log_test("Model Unloading", False, "No models to unload")
                return
            
            # Pick a model to unload
            model_to_unload = loaded_models[0]
            print(f"\n🔄 Unloading {model_to_unload['model_id']}...")
            
            unload_request = llm_service_pb2.UnloadModelRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="unload-1",
                    user_id="test-user"
                ),
                model_id=model_to_unload['model_id']
            )
            
            unload_response = await stub.UnloadModel(unload_request)
            
            if unload_response.success:
                # Verify model is gone
                new_models = await self._get_loaded_models(stub)
                if not any(m['model_id'] == model_to_unload['model_id'] for m in new_models):
                    self.log_test("Model Unloading", True, 
                                f"Successfully unloaded {model_to_unload['model_id']}")
                else:
                    self.log_test("Model Unloading", False, 
                                "Model still appears in loaded models")
            else:
                self.log_test("Model Unloading", False, 
                            f"Failed to unload: {unload_response.error.message}")
                            
        except Exception as e:
            self.log_test("Model Unloading", False, str(e))
        finally:
            await channel.close()
    
    async def _get_loaded_models(self, stub) -> List[Dict[str, Any]]:
        """Helper to get loaded models info"""
        request = llm_service_pb2.GetLoadedModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id="get-models",
                user_id="test-user"
            ),
            include_stats=True
        )
        
        response = await stub.GetLoadedModels(request)
        models = []
        for model in response.models:
            models.append({
                'model_id': model.model_id,
                'model_name': model.model_name,
                'backend': model.backend,
                'device': model.device,
                'is_ready': model.status.is_available
            })
        return models
    
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
    """Run all LLM service tests with TGI backend"""
    print("🚀 LLM Service Test Suite - TGI Multi-Instance")
    print("="*60)
    print("Make sure the LLM service is running on localhost:50051")
    print("="*60)
    
    # Wait a bit for service to be ready
    await asyncio.sleep(2)
    
    tester = LLMServiceTester()
    
    # Run all tests
    await tester.test_service_health()
    
    # Load first model
    first_model_id = await tester.test_tgi_model_loading()
    
    # Test reuse
    await tester.test_tgi_instance_reuse()
    
    # Test multiple models
    await tester.test_multiple_tgi_models()
    
    # Test concurrent requests
    await tester.test_concurrent_tgi_requests()
    
    # Test streaming
    await tester.test_streaming_with_tgi()
    
    # Test unloading
    await tester.test_model_unloading()
    
    # Print summary
    all_passed = tester.print_summary()
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")
    
    # Show loaded models at the end
    print("\n📊 Final Model State:")
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = llm_service_pb2_grpc.LLMServiceStub(channel)
    
    try:
        models = await tester._get_loaded_models(stub)
        print(f"Models still loaded: {len(models)}")
        for model in models:
            print(f"  - {model['model_id']} ({model['backend']})")
    except:
        pass
    finally:
        await channel.close()


if __name__ == "__main__":
    # Ensure the LLM service is configured to allow TGI backend
    os.environ['BACKEND_TYPE'] = ''  # Allow auto-selection
    os.environ['AUTO_SELECT_BACKEND'] = 'true'
    
    asyncio.run(main())