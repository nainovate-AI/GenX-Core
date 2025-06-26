#!/usr/bin/env python3
"""
Basic test to ensure the service can start
"""
import asyncio
import sys
import os
from typing import Dict


# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

async def test_basic_imports():
    """Test basic imports and service creation"""
    try:
        print("Testing basic imports...")
        
        # Test imports
        from genx_components.common.config import BaseServiceConfig
        print("✓ Config import successful")
        
        from genx_components.common.base_service import GenxMicroservice
        print("✓ Base service import successful")
        
        # Create a simple test service
        class TestService(GenxMicroservice):
            async def initialize(self):
                print("✓ Test service initialized")
                
            async def add_grpc_services(self, server):
                pass
                
            async def cleanup(self):
                pass
                
            async def check_health(self) -> Dict:
                return {"status": "healthy"}
        
        # Create config
        os.environ['SERVICE_NAME'] = 'test-service'
        config = BaseServiceConfig()
        
        # Create service instance
        service = TestService(config)
        print("✓ Service instance created")
        
        print("\nAll basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_imports())
    sys.exit(0 if success else 1)