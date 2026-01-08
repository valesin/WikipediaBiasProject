"""
Thread-safe rate limiter using sliding window approach.

This module provides a thread-safe rate limiter that ensures no more than
a specified number of requests per second, using a sliding window timestamp approach.
"""

import threading
import time
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Thread-safe rate limiter using sliding window of timestamps.
    
    This limiter ensures that no more than max_per_second requests
    are allowed in any 1-second window. It's safe for use with
    concurrent threads.
    """
    
    def __init__(self, max_per_second: int = 180):
        """
        Initialize the rate limiter.
        
        Args:
            max_per_second: Maximum number of requests allowed per second
        """
        self.max_per_second = max_per_second
        self.timestamps = deque()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request, waiting if necessary.
        
        This method will block until a permit is available or timeout expires.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            True if permit acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                cutoff = now - 1.0
                
                # Remove timestamps older than 1 second
                while self.timestamps and self.timestamps[0] < cutoff:
                    self.timestamps.popleft()
                
                # If under limit, grant permit
                if len(self.timestamps) < self.max_per_second:
                    self.timestamps.append(now)
                    return True
                
                # Calculate how long to wait
                sleep_time = self.timestamps[0] - cutoff
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning("Rate limiter acquire timeout")
                    return False
                sleep_time = min(sleep_time, timeout - elapsed)
            
            # Sleep without holding the lock
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dict with current request count and max allowed
        """
        with self.lock:
            now = time.time()
            cutoff = now - 1.0
            
            # Count active timestamps
            active_count = sum(1 for ts in self.timestamps if ts > cutoff)
            
            return {
                "current_rate": active_count,
                "max_rate": self.max_per_second,
                "utilization": active_count / self.max_per_second if self.max_per_second > 0 else 0,
            }
