# src/__init__.py
"""XYZ MDM 3.0 - Model Inference Service.

Production-grade AI inference microservice for entity matching.
"""

__version__ = "3.0.0"
__author__ = "XYZ MDM Engineering Team"
__license__ = "Proprietary"

from src.config.settings import settings

__all__ = ["settings", "__version__"]