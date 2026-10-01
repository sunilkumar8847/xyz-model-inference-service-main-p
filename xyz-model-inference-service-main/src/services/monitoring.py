# src/services/monitoring.py
"""Monitoring and metrics collection service."""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from src.infrastructure.observability import (
    inference_latency,
    inference_errors,
    model_accuracy,
    cache_hit_rate,
    http_requests_total
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Centralized monitoring and metrics service.
    
    Responsibilities:
    - Collect and aggregate metrics
    - Track SLO compliance
    - Detect anomalies
    - Generate alerts
    """
    
    def __init__(self):
        self._request_count = 0
        self._error_count = 0
        self._latencies = []
        self._start_time = datetime.now()
        self._slo_violations = defaultdict(int)
        
        # SLO thresholds
        self.SLO_P95_LATENCY_MS = 200
        self.SLO_ERROR_RATE = 0.01  # 1%
        self.SLO_AVAILABILITY = 0.99999  # 99.999%
    
    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        status: str,
        error: Optional[Exception] = None
    ) -> None:
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint name
            latency_ms: Request latency in milliseconds
            status: Request status (success, error, timeout)
            error: Exception if error occurred
        """
        self._request_count += 1
        self._latencies.append(latency_ms)
        
        # Record Prometheus metrics
        http_requests_total.labels(endpoint=endpoint, status=status).inc()
        inference_latency.observe(latency_ms)
        
        if status == "error":
            self._error_count += 1
            error_type = type(error).__name__ if error else "unknown"
            inference_errors.labels(error_type=error_type).inc()
        
        # Check SLO violations
        if latency_ms > self.SLO_P95_LATENCY_MS:
            self._slo_violations['latency'] += 1
            logger.warning(
                f"SLO violation: Latency {latency_ms}ms > {self.SLO_P95_LATENCY_MS}ms"
            )
    
    def record_model_accuracy(
        self,
        model_version: str,
        f1_score: float
    ) -> None:
        """Record model accuracy metric."""
        model_accuracy.labels(model_version=model_version).set(f1_score)
        
        # Check if accuracy below threshold
        if f1_score < 0.92:
            self._slo_violations['accuracy'] += 1
            logger.warning(
                f"SLO violation: Model {model_version} accuracy {f1_score} < 0.92"
            )
    
    def record_cache_stats(
        self,
        hits: int,
        misses: int
    ) -> None:
        """Record cache hit rate."""
        total = hits + misses
        hit_rate_pct = (hits / total * 100) if total > 0 else 0
        cache_hit_rate.set(hit_rate_pct)
        
        # Check if cache hit rate below threshold
        if hit_rate_pct < 90:
            self._slo_violations['cache'] += 1
            logger.warning(f"Cache hit rate {hit_rate_pct}% < 90%")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get aggregated metrics summary.
        
        Returns:
            Dict with current metrics
        """
        error_rate = (self._error_count / self._request_count) if self._request_count > 0 else 0
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        latency_stats = {}
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            latency_stats = {
                "min": min(self._latencies),
                "max": max(self._latencies),
                "avg": sum(self._latencies) / len(self._latencies),
                "p50": sorted_latencies[int(len(sorted_latencies) * 0.50)],
                "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
                "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)]
}
            return {
        "requests": {
            "total": self._request_count,
            "successful": self._request_count - self._error_count,
            "failed": self._error_count,
            "error_rate": round(error_rate, 4)
        },
        "latency": latency_stats,
        "uptime_seconds": round(uptime, 2),
        "slo_compliance": self._check_slo_compliance(),
        "slo_violations": dict(self._slo_violations)
    }

def _check_slo_compliance(self) -> Dict[str, bool]:
    """Check if service is meeting SLOs."""
    error_rate = (self._error_count / self._request_count) if self._request_count > 0 else 0
    
    p95_latency = 0
    if self._latencies:
        sorted_latencies = sorted(self._latencies)
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    
    return {
        "latency_slo": p95_latency <= self.SLO_P95_LATENCY_MS,
        "error_rate_slo": error_rate <= self.SLO_ERROR_RATE,
        "availability_slo": error_rate <= (1 - self.SLO_AVAILABILITY)
    }

def reset_stats(self) -> None:
    """Reset monitoring statistics."""
    self._request_count = 0
    self._error_count = 0
    self._latencies = []
    self._slo_violations.clear()
    self._start_time = datetime.now()
    logger.info("Monitoring statistics reset")