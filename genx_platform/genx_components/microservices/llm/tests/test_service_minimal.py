#!/usr/bin/env python3
"""
Test service without model loading
"""
import asyncio
import sys
import os
from pathlib import Path

# Load environment
from dotenv import load_dotenv

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

# Load .env
env_path = Path(current_dir) / '.env'
load_dotenv(env_path)

# Set minimal config to skip model loading
os.environ['DEFAULT_MODEL_ID'] = 'gpt2'
os.environ['TELEMETRY_ENABLED'] = 'false'

async def test_minimal():
    """Test minimal service creation"""
    try:
        from genx_components.microservices.llm.src.models.config import LLMServiceConfig
        from genx_components.microservices.llm.src.core.model_manager import ModelManager
        from genx_components.microservices.llm.src.service.llm_service import LLMService
        
        print("Creating config...")
        config = LLMServiceConfig()
        print(f"✓ Config created: {config.service_name}")
        
        print("\nCreating model manager...")
        model_manager = ModelManager({
            'default_model_id': config.default_model_id,
            'auto_select_backend': True,
        })
        print("✓ Model manager created")
        
        print("\nCreating LLM service...")
        llm_service = LLMService(
            model_manager=model_manager,
            telemetry=None
        )
        print(f"✓ LLM service created: {llm_service}")
        
        # Check that service has required methods
        required_methods = ['Generate', 'StreamGenerate', 'ListModels', 'GetModel', 'ValidatePrompt']
        for method in required_methods:
            if hasattr(llm_service, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal())
    sys.exit(0 if success else 1)