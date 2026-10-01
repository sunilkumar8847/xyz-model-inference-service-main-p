# src/adapters/circuit_breaker.py
"""Circuit breaker for resilience."""

import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,
        recovery_timeout_s: int = 30,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime = None
        self.opened_at: datetime = None
    
    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        If circuit is OPEN, raises without calling function.
        If HALF_OPEN, allows test call.
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit {self.name} entering HALF_OPEN")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit {self.name} recovered, CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        failure_rate = self.failure_count / (self.failure_count + self.success_count + 1)
        
        if failure_rate >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.opened_at = datetime.now()
            logger.warning(
                f"Circuit {self.name} OPEN (failure rate: {failure_rate:.1%})"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.opened_at:
            return True
        
        elapsed = (datetime.now() - self.opened_at).total_seconds()
        return elapsed >= self.recovery_timeout_s


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass