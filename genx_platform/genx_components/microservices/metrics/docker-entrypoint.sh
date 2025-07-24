#!/usr/bin/env python3
"""
Docker entrypoint for Metrics Service
Ensures proto files are generated before starting
"""
import os
import sys
import subprocess

def generate_protos():
    """Generate protobuf files if they don't exist"""
    proto_file = "/genx_platform/genx_components/common/grpc/common_pb2.py"
    
    if not os.path.exists(proto_file):
        print("Generating protobuf files...")
        os.chdir("/genx_platform")
        
        # Generate common protos
        subprocess.run([
            sys.executable, "-m", "grpc_tools.protoc",
            "-I./protos",
            "--python_out=./genx_components/common/grpc",
            "--grpc_python_out=./genx_components/common/grpc",
            "./protos/common.proto"
        ], check=True)
        
        # Generate service protos
        subprocess.run([
            sys.executable, "-m", "grpc_tools.protoc",
            "-I./protos",
            "--python_out=./genx_components/microservices/grpc",
            "--grpc_python_out=./genx_components/microservices/grpc",
            "./protos/common.proto",
            "./protos/metrics_service.proto"
        ], check=True)
        
        # Fix imports
        print("Fixing imports...")
        import_fixes = [
            ("sed", "-i", "s/^import common_pb2/from . import common_pb2/g",
             "/genx_platform/genx_components/microservices/grpc/metrics_service_pb2_grpc.py"),
        ]
        
        for cmd in import_fixes:
            subprocess.run(cmd, check=True)
        
        # Create __init__.py files
        open("/genx_platform/genx_components/common/grpc/__init__.py", "a").close()
        open("/genx_platform/genx_components/microservices/grpc/__init__.py", "a").close()
        
        print("Protobuf generation complete")

def main():
    """Main entrypoint"""
    # Generate protos if needed
    generate_protos()
    
    # Change to metrics directory
    os.chdir("/genx_platform/genx_components/microservices/metrics")
    
    # Start the service
    os.execvp(sys.executable, [sys.executable, "-m", "src.main"])

if __name__ == "__main__":
    main()