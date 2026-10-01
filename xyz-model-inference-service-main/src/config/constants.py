# src/config/constants.py
"""Application constants."""

from enum import Enum

# Service Constants
SERVICE_NAME = "model-inference-service"
SERVICE_VERSION = "3.0.0"
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Inference Constants
DEFAULT_FEATURE_DIMENSION = 50
DEFAULT_CONFIDENCE_INTERVAL_WIDTH = 0.04  # ±2%

# Timeout Constants (milliseconds)
TIMEOUT_ULTRA_FAST = 50  # Cache lookups, JWT validation
TIMEOUT_FAST = 200  # Database reads, API calls, matching
TIMEOUT_NORMAL = 500  # Database writes, schema validation
TIMEOUT_LONG = 30_000  # Batch operations, cross-cell queries

# Model Decision Thresholds
THRESHOLD_AUTO_MATCH = 0.95
THRESHOLD_LIKELY_MATCH = 0.85
THRESHOLD_UNCERTAIN = 0.50
THRESHOLD_LIKELY_NON = 0.15

# Cache Constants
CACHE_TTL_MODEL_METADATA = 300  # 5 minutes
CACHE_TTL_FEATURES = 60  # 1 minute
CACHE_TTL_MATCH_RESULTS = 3600  # 1 hour

# Health Check Constants
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 5  # seconds

# Circuit Breaker Constants
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 0.5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30  # seconds
CIRCUIT_BREAKER_SAMPLE_SIZE = 100

# Metrics Constants
HISTOGRAM_BUCKETS = (10, 20, 50, 100, 200, 500, 1000)
LATENCY_PERCENTILES = [0.50, 0.95, 0.99]
