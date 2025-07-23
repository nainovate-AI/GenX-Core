# services/proxy_service/start_service.py

#!/usr/bin/env python3
"""Start the Proxy Service"""

import asyncio
import sys
import os
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from src.main import ProxyService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    logger.info("Starting Proxy Service...")
    
    service = ProxyService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())