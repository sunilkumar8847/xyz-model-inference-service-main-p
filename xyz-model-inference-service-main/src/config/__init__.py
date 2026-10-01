# src/config/__init__.py
"""Configuration module."""

from src.config.settings import settings, Settings
from src.config.constants import (
    SERVICE_NAME,
    SERVICE_VERSION,
    API_VERSION,
    TIMEOUT_FAST,
    TIMEOUT_NORMAL,
    THRESHOLD_AUTO_MATCH,
    THRESHOLD_LIKELY_MATCH,
)

__all__ = [
    "settings",
    "Settings",
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "API_VERSION",
    "TIMEOUT_FAST",
    "TIMEOUT_NORMAL",
    "THRESHOLD_AUTO_MATCH",
    "THRESHOLD_LIKELY_MATCH",
]