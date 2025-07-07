#!/usr/bin/env python3
"""
Generate Python code from Protocol Buffer definitions with correct imports
"""
import os
import subprocess
import sys
from pathlib import Path

def generate_protos():
    """Generate Python code from proto files"""
    # Get project root
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "protos"
    
    # Output directories
    output_dirs = {
        "common": project_root / "genx_components" / "common" / "grpc",
        "services": project_root / "genx_components" / "microservices" / "grpc"
    }
    
    # Create output directories and __init__.py files
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
        init_file = output_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            print(f"Created {init_file}")
    
    # Find all proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print("No proto files found!")
        return False
    
    print(f"Found {len(proto_files)} proto files")
    
    # Generate Python code for each proto
    for proto_file in proto_files:
        print(f"\nGenerating code for {proto_file.name}")
        
        # Determine output directory
        if "common" in proto_file.name:
            output_dir = output_dirs["common"]
        else:
            output_dir = output_dirs["services"]
        
        # Generate Python code
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Generated {proto_file.stem}_pb2.py and {proto_file.stem}_pb2_grpc.py")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to generate code for {proto_file.name}")
            print(f"    Error: {e.stderr}")
            return False
    
    # Fix imports in generated files
    print("\nFixing imports in generated files...")
    fix_imports_comprehensive(output_dirs)
    
    print("\n✅ Proto generation complete!")
    return True

def fix_imports_comprehensive(output_dirs):
    """Fix all imports in generated proto files"""
    
    # Fix common files
    common_dir = output_dirs["common"]
    if common_dir.exists():
        # Fix common_pb2_grpc.py if it exists
        common_grpc = common_dir / "common_pb2_grpc.py"
        if common_grpc.exists():
            content = common_grpc.read_text()
            content = content.replace(
                "import common_pb2 as common__pb2",
                "from genx_components.common.grpc import common_pb2 as common__pb2"
            )
            common_grpc.write_text(content)
            print(f"  ✓ Fixed imports in common_pb2_grpc.py")
    
    # Fix service files
    services_dir = output_dirs["services"]
    if services_dir.exists():
        # Process all _pb2.py files
        for pb2_file in services_dir.glob("*_pb2.py"):
            content = pb2_file.read_text()
            original_content = content
            
            # Fix common imports (both styles)
            content = content.replace(
                "import common_pb2 as common__pb2",
                "from genx_components.common.grpc import common_pb2 as common__pb2"
            )
            content = content.replace(
                "from . import common_pb2 as common__pb2",
                "from genx_components.common.grpc import common_pb2 as common__pb2"
            )
            
            if content != original_content:
                pb2_file.write_text(content)
                print(f"  ✓ Fixed imports in {pb2_file.name}")
        
        # Process all _pb2_grpc.py files
        for grpc_file in services_dir.glob("*_pb2_grpc.py"):
            content = grpc_file.read_text()
            original_content = content
            
            # Get the base name (e.g., "llm_service" from "llm_service_pb2_grpc.py")
            base_name = grpc_file.stem.replace("_pb2_grpc", "")
            
            # Fix the import to use full path
            old_import = f"import {base_name}_pb2 as"
            new_import = f"from genx_components.microservices.grpc import {base_name}_pb2 as"
            content = content.replace(old_import, new_import)
            
            # Also fix if there's no 'as' clause
            old_import_simple = f"import {base_name}_pb2\n"
            new_import_simple = f"from genx_components.microservices.grpc import {base_name}_pb2\n"
            content = content.replace(old_import_simple, new_import_simple)
            
            # Fix common imports too
            content = content.replace(
                "import common_pb2 as common__pb2",
                "from genx_components.common.grpc import common_pb2 as common__pb2"
            )
            
            if content != original_content:
                grpc_file.write_text(content)
                print(f"  ✓ Fixed imports in {grpc_file.name}")

def create_all_init_files():
    """Create all necessary __init__.py files"""
    project_root = Path(__file__).parent.parent
    
    dirs_needing_init = [
        "genx_components",
        "genx_components/common",
        "genx_components/common/grpc",
        "genx_components/microservices",
        "genx_components/microservices/grpc",
        "genx_components/microservices/llm",
        "genx_components/microservices/llm/src",
        "genx_components/microservices/llm/src/backends",
        "genx_components/microservices/llm/src/core",
        "genx_components/microservices/llm/src/service",
        "genx_components/microservices/llm/src/models",
    ]
    
    for dir_path in dirs_needing_init:
        full_path = project_root / dir_path
        if full_path.exists():
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
                print(f"  ✓ Created {init_file}")

if __name__ == "__main__":
    # Ensure grpcio-tools is installed
    try:
        import grpc_tools
    except ImportError:
        print("Error: grpcio-tools not installed. Run: pip install grpcio-tools")
        sys.exit(1)
    
    # Create all __init__.py files first
    print("Creating __init__.py files...")
    create_all_init_files()
    
    # Generate protos
    success = generate_protos()
    sys.exit(0 if success else 1)