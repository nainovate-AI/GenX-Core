#!/bin/bash

# Generate gRPC stubs from proto files
echo "Generating gRPC stubs..."
python -m grpc_tools.protoc -Iprotos --python_out=protos --grpc_python_out=protos protos/core_service.proto
python -m grpc_tools.protoc -Iprotos --python_out=protos --grpc_python_out=protos protos/mcp.proto
python -m grpc_tools.protoc -Iprotos --python_out=protos --grpc_python_out=protos protos/vector_service.proto
python -m grpc_tools.protoc -Iprotos --python_out=protos --grpc_python_out=protos protos/retrieval_service.proto

# Create __init__.py to make protos/ a package
touch protos/__init__.py

# Patch core_service_pb2_grpc.py to use relative import
sed -i '' 's/import core_service_pb2 as core__service__pb2/from . import core_service_pb2 as core__service__pb2/' protos/core_service_pb2_grpc.py

# Patch mcp_pb2_grpc.py to use relative import
sed -i '' 's/import mcp_pb2 as mcp__pb2/from . import mcp_pb2 as mcp__pb2/' protos/mcp_pb2_grpc.py

# Patch vector_service_pb2_grpc.py to use relative import
sed -i '' 's/import vector_service_pb2 as vector__service__pb2/from . import vector_service_pb2 as vector__service__pb2/' protos/vector_service_pb2_grpc.py

# Patch retrieval_service_pb2_grpc.py to use relative import
sed -i '' 's/import retrieval_service_pb2 as retrieval__service__pb2/from . import retrieval_service_pb2 as retrieval__service__pb2/' protos/retrieval_service_pb2_grpc.py

# Verify stubs were generated and patched
if [ -f protos/core_service_pb2.py ] && [ -f protos/core_service_pb2_grpc.py ] && \
   [ -f protos/mcp_pb2.py ] && [ -f protos/mcp_pb2_grpc.py ] && \
   [ -f protos/vector_service_pb2.py ] && [ -f protos/vector_service_pb2_grpc.py ] && \
   [ -f protos/retrieval_service_pb2.py ] && [ -f protos/retrieval_service_pb2_grpc.py ] && \
   [ -f protos/__init__.py ]; then
    echo "gRPC stubs generated and patched successfully."
else
    echo "Error: gRPC stubs generation or patching failed."
    exit 1
fi