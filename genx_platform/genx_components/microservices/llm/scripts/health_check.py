#!/usr/bin/env python3
"""Health check script for LLM service"""
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/app')

def check_health():
    try:
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        
        port = os.environ.get("SERVICE_PORT", "50053")
        service_name = os.environ.get("SERVICE_NAME", "llm-service")
        
        channel = grpc.insecure_channel(f'localhost:{port}')
        stub = health_pb2_grpc.HealthStub(channel)
        request = health_pb2.HealthCheckRequest(service=service_name)
        response = stub.Check(request, timeout=5)
        
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            print(f"Service {service_name} is healthy")
            return True
        else:
            print(f"Service {service_name} unhealthy: {response.status}")
            return False
            
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    healthy = check_health()
    sys.exit(0 if healthy else 1)