#!/bin/bash

# Setup test environment for GenX Platform
echo "Setting up test environment..."

# Ensure the script is run from the genx-platform directory
cd "$(dirname "$0")/.."

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install grpcio grpcio-tools pytest pytest-asyncio
pip install -r core_services/data_preparation/requirements.txt

# Generate gRPC stubs
./scripts/generate_protos.sh

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Ensure pytest.ini exists
if [ ! -f pytest.ini ]; then
    cat > pytest.ini << EOL
[pytest]
asyncio_mode = auto
markers =
    asyncio: mark a test as an async test
EOL
fi

echo "Test environment setup complete."
