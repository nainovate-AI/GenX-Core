"""
Simple in-memory cache for metrics
Production version would use Redis
"""
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .logger import setup_logging

logger = setup_logging(__name__)


class CacheEntry:
    """Single cache entry with TTL"""
    
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.ttl_seconds = ttl_seconds
        self.created_at = time.time()
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def time_to_live(self) -> float:
        """Get remaining TTL in seconds"""
        remaining = self.ttl_seconds - (time.time() - self.created_at)
        return max(0, remaining)


class MetricsCache:
    """
    Simple in-memory cache with TTL support
    Thread-safe using asyncio locks
    """
    
    def __init__(self, ttl_seconds: int = 30):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        
        logger.info(f"Metrics cache initialized with {ttl_seconds}s TTL")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            if entry.is_expired():
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key]
                return None
            
            logger.debug(
                f"Cache hit for key: {key}, TTL: {entry.time_to_live():.1f}s"
            )
            return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl_seconds or self._ttl_seconds
        
        async with self._lock:
            self._cache[key] = CacheEntry(value, ttl)
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache deleted for key: {key}")
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared, removed {count} entries")
    
    async def size(self) -> int:
        """Get number of entries in cache"""
        async with self._lock:
            return len(self._cache)
    
    async def keys(self) -> list:
        """Get all cache keys"""
        async with self._lock:
            return list(self._cache.keys())
    
    async def _cleanup_expired(self) -> None:
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                        
            except asyncio.CancelledError:
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values() 
                if entry.is_expired()
            )
            
            avg_ttl = 0
            if self._cache:
                avg_ttl = sum(
                    entry.time_to_live() 
                    for entry in self._cache.values()
                ) / len(self._cache)
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "average_ttl": avg_ttl,
                "default_ttl": self._ttl_seconds
            }
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()