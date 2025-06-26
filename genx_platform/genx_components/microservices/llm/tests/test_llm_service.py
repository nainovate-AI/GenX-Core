#!/usr/bin/env python3
"""
genx_platform/genx_components/microservices/llm/test_llm_service.py
Test LLM Service
"""
import asyncio
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

async def test_llm_service():
    """Test if LLM service can be created"""
    try:
        print("Testing LLM service initialization...")
        
        # Set required environment variable
        os.environ['SERVICE_NAME'] = 'llm-service'
        
        # Import and create service
        from genx_components.microservices.llm.src.main import LLMMicroservice
        
        service = LLMMicroservice()
        print("✓ LLM service created successfully")
        
        # Test configuration
        print(f"✓ Service name: {service.config.service_name}")
        print(f"✓ Service port: {service.config.service_port}")
        print(f"✓ Default model: {service.config.default_model_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_service())
    sys.exit(0 if success else 1)