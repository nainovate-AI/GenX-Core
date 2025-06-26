#!/usr/bin/env python3
"""Health check script for LLM service"""
import grpc
import sys
from grpc_health.v1 import health_pb2, health_pb2_grpc

def main():
    try:
        channel = grpc.insecure_channel('localhost:50053')
        stub = health_pb2_grpc.HealthStub(channel)
        request = health_pb2.HealthCheckRequest(service='llm-service')
        response = stub.Check(request, timeout=5)
        
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            sys.exit(0)
        else:
            print(f"Service not serving. Status: {response.status}")
            sys.exit(1)
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()