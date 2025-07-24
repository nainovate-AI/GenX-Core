#!/usr/bin/env python3
"""
Run script for Metrics Service
Sets up proper Python path for GenX platform
"""
import os
import sys
import subprocess

# Get the genx_platform directory
current_file = os.path.abspath(__file__)
metrics_dir = os.path.dirname(current_file)
microservices_dir = os.path.dirname(metrics_dir)
genx_components_dir = os.path.dirname(microservices_dir)
genx_platform_dir = os.path.dirname(genx_components_dir)

# Set PYTHONPATH to include genx_platform
env = os.environ.copy()
env['PYTHONPATH'] = genx_platform_dir

# Set default environment variables if not already set
env.setdefault('SERVICE_NAME', 'metrics-service')
env.setdefault('SERVICE_PORT', '50056')
env.setdefault('ENVIRONMENT', 'development')
env.setdefault('DEBUG', 'true')
env.setdefault('TELEMETRY_ENABLED', 'false')  # Disable telemetry for local dev
env.setdefault('GRPC_MAX_WORKERS', '4')

print(f"Starting Metrics Service...")
print(f"PYTHONPATH={genx_platform_dir}")
print(f"Working directory: {metrics_dir}")
print(f"Port: {env['SERVICE_PORT']}")

# Run the service
try:
    subprocess.run(
        [sys.executable, '-m', 'src.main'],
        cwd=metrics_dir,
        env=env,
        check=True
    )
except KeyboardInterrupt:
    print("\nService stopped by user")
except subprocess.CalledProcessError as e:
    print(f"Service failed with error: {e}")
    sys.exit(1)