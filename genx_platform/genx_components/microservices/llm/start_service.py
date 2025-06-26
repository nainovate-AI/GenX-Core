#!/usr/bin/env python3
"""
genx_platform/genx_components/microservices/llm/start_service.py
Start LLM Service with proper error handling
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

# Load .env file
env_path = Path(current_dir) / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded .env from {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}")

# Import after path is set
from genx_components.microservices.llm.src.main import main

async def start_with_monitoring():
    """Start the service with monitoring"""
    try:
        logger.info("Starting LLM microservice...")
        logger.info(f"Python path: {sys.path[0]}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Check environment
        env_vars = ['SERVICE_NAME', 'SERVICE_PORT', 'DEFAULT_MODEL_ID']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            logger.info(f"{var}: {value}")
        
        # Start the service
        await main()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(start_with_monitoring())