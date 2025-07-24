#!/usr/bin/env python3
"""
Fix imports in generated protobuf files
Run this after generating proto files
"""
import os
import re

def fix_imports_in_file(filepath):
    """Fix relative imports in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common_pb2 imports
    content = re.sub(
        r'^import common_pb2 as common__pb2',
        'from . import common_pb2 as common__pb2',
        content,
        flags=re.MULTILINE
    )
    
    # Fix metrics_service_pb2 imports
    content = re.sub(
        r'^import metrics_service_pb2 as metrics__service__pb2',
        'from . import metrics_service_pb2 as metrics__service__pb2',
        content,
        flags=re.MULTILINE
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed imports in {os.path.basename(filepath)}")

def fix_directory(directory):
    """Fix imports in all *_pb2_grpc.py files in a directory"""
    for filename in os.listdir(directory):
        if filename.endswith('_pb2_grpc.py'):
            filepath = os.path.join(directory, filename)
            fix_imports_in_file(filepath)

if __name__ == "__main__":
    # Fix imports in all generated directories
    directories = [
        r"genx_components\common\grpc",
        r"genx_components\microservices\grpc",
        r"genx_components\microservices\metrics\src\generated"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nFixing imports in {directory}")
            fix_directory(directory)
        else:
            print(f"Directory not found: {directory}")
    
    print("\nImport fixes completed!")