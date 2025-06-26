#!/usr/bin/env python3
"""
Generate Python code from Protocol Buffer definitions
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
    
    # Create output directories
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create __init__.py files
        (output_dir / "__init__.py").touch()
    
    # Find all proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print("No proto files found!")
        return False
    
    print(f"Found {len(proto_files)} proto files")
    
    # Generate Python code for each proto
    for proto_file in proto_files:
        print(f"Generating code for {proto_file.name}")
        
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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Generated {proto_file.stem}_pb2.py and {proto_file.stem}_pb2_grpc.py")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to generate code for {proto_file.name}")
            print(f"    Error: {e.stderr}")
            return False
    
    # Fix imports in generated files
    print("\nFixing imports in generated files...")
    fix_imports(output_dirs)
    
    print("\n✅ Proto generation complete!")
    return True

def fix_imports(output_dirs):
    """Fix relative imports in generated proto files"""
    for output_dir in output_dirs.values():
        for py_file in output_dir.glob("*_pb2*.py"):
            content = py_file.read_text()
            original_content = content
            
            # Fix imports based on file location
            if "grpc/services" in str(py_file) or "microservices/grpc" in str(py_file):
                # Service files in microservices/grpc
                
                # Fix common_pb2 imports
                content = content.replace(
                    "import common_pb2",
                    "from genx_components.common.grpc import common_pb2"
                )
                
                # Fix other service imports (e.g., llm_service_pb2 in llm_service_pb2_grpc.py)
                if "_grpc.py" in py_file.name:
                    base_name = py_file.name.replace("_pb2_grpc.py", "")
                    content = content.replace(
                        f"import {base_name}_pb2 as",
                        f"from . import {base_name}_pb2 as"
                    )
            else:
                # Common files in common/grpc
                content = content.replace(
                    "import common_pb2",
                    "from . import common_pb2"
                )
            
            # Write back if changed
            if content != original_content:
                py_file.write_text(content)
                print(f"  ✓ Fixed imports in {py_file.name}")

if __name__ == "__main__":
    # Ensure grpcio-tools is installed
    try:
        import grpc_tools
    except ImportError:
        print("Error: grpcio-tools not installed. Run: pip install grpcio-tools")
        sys.exit(1)
    
    success = generate_protos()
    sys.exit(0 if success else 1)