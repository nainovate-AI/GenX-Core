"""
Base Collector Abstract Class
All metrics collectors should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
from contextlib import contextmanager
import sys
import os

# Add paths for imports
current_file = os.path.abspath(__file__)
collectors_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(collectors_dir)
sys.path.insert(0, src_dir)

from genx_components.microservices.metrics.src.utils.logger import setup_logging

logger = setup_logging(__name__)

class BaseCollector(ABC):
    """
    Abstract base class for all metrics collectors.
    Provides a structure for collecting metrics and handling common functionality.
    """

    def __init__(self, name: str):
        self.name = name
        self._initialized = False
        self._last_collection_time = 0
        self._last_collection_date: Optional[Dict[str, Any]] = None
        self._error_count = 0
        self._success_count = 0

    async def initialize(self) -> None:
        """
        Initialize the collector.
        This method should be overridden by subclasses to perform any necessary setup.
        """
        if self._initialized:
            logger.warning(f"{self.name} collector is already initialized.")
            return
        
        try:
            await self._initialize()
            self._initialized = True
            logger.info(f"{self.name} collector initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing {self.name} collector: {e}")
            raise
    
    @abstractmethod
    async def _initialize(self) -> None:
        """
        Collector-specific initialization logic.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def _collect(self) -> Dict[str, Any]:
        """
        Collect metrics - must be implemented by subclasses.
        Should return a dictionary of collected metrics.
        """
        pass

    async def collect(self) -> Dict[str, Any]:
        """
        Public method to collect metrics with error handling and timing.
        Should be called by external code to trigger metric collection.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()

        try:
            # Collect metrics
            data = await self._collect()

            # Update statistics
            self._last_collection_time = time.time()
            self._last_collection_data = data
            self._success_count += 1

            # Add metadata
            data['_metadata'] = {
                'collector_name': self.name,
                'collection_time': self._last_collection_time,
                'duration_ms': int((time.time() - start_time) * 1000)
            }

            logger.debug(
                f"{self.name} collection successful",
                duration_ms = int((time.time() - start_time) * 1000)
            )

            return data
        
        except Exception as e:
            self._error_count += 1
            logger.error(
                f"{self.name} collection failed",
                error=str(e),
                error_count=self._error_count
            )

            # Return last know good data if available
            if self._last_collection_data:
                logger.info(f"Returning last known good data for {self.name} collector.")
                return self._last_collection_data
            
            # Return error metrics
            return self._get_error_metrics(str(e))
        
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """
        Return error metrics when collection fails.
        Override this method for custom error responses.
        """
        return {
            'error': True,
            'error_message': error_message,
            'collector': self.name,
            '_metadata': {
                'collector': self.name,
                'error': True,
                'timestamp': time.time(),
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the collector.
        Returns a dictionary with initialization status and counts.
        """
        return {
            'name': self.name,
            'initialized': self._initialized,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'last_collection_time': self._last_collection_time,
            'error_rate': self._error_count / max(1, self._success_count + self._error_count),
        }
    
    @contextmanager
    def _time_operation(self, operation: str):
        """
        Context manager to time operations within the collector.
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            logger.debug(
                f"{self.name}.{operation} completed",
                duration_ms=int(duration * 1000)
            )
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value
        """
        return numerator / denominator if denominator != 0 else default
    
    def _bytes_to_gb(self, bytes_value: int) -> float:
        """
        Convert bytes to gigabytes.
        """
        return round(bytes_value / (1024 ** 3), 2) 
    
    def _bytes_to_mb(self, bytes_value: int) -> float:
        """
        Convert bytes to megabytes.
        """
        return round(bytes_value / (1024 ** 2), 2)