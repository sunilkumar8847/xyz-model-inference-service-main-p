# src/config/settings.py
"""Configuration management for model-inference-service."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class TritonSettings(BaseSettings):
    """Triton Inference Server configuration."""
    model_config = SettingsConfigDict(extra="ignore")

    host: str = Field(default="localhost", env="TRITON_HOST")
    port: int = Field(default=8000, env="TRITON_PORT")
    grpc_port: int = Field(default=8001, env="TRITON_GRPC_PORT")
    timeout_ms: int = Field(default=200, env="TRITON_TIMEOUT_MS")
    max_retries: int = Field(default=3, env="TRITON_MAX_RETRIES")
    health_check_interval_s: int = Field(default=30, env="TRITON_HEALTH_CHECK_INTERVAL")


class FeatureStoreSettings(BaseSettings):
    """Feature Store configuration."""
    model_config = SettingsConfigDict(extra="ignore")

    url: str = Field(default="http://localhost:8115", env="FEATURE_STORE_URL")
    timeout_s: int = Field(default=10, env="FEATURE_STORE_TIMEOUT_S")
    cache_ttl_s: int = Field(default=300, env="FEATURE_STORE_CACHE_TTL_S")


class MLflowSettings(BaseSettings):
    """MLflow model registry configuration."""
    model_config = SettingsConfigDict(extra="ignore")

    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    registry_uri: str = Field(default="http://localhost:5000", env="MLFLOW_REGISTRY_URI")
    model_name: str = Field(default="customer_matcher", env="MLFLOW_MODEL_NAME")
    refresh_interval_s: int = Field(default=300, env="MLFLOW_REFRESH_INTERVAL_S")


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    model_config = SettingsConfigDict(extra="ignore")

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    ttl_s: int = Field(default=3600, env="REDIS_CACHE_TTL_S")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")


class Settings(BaseSettings):
    """Main application settings.

    Loads configuration in order (later files override earlier):
      1. .env          — base defaults (committed, non-sensitive)
      2. .env.local    — local overrides (git-ignored, secrets)
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),  # .env.local overrides .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service identity
    service_name: str = "model-inference-service"
    version: str = "3.0.0"
    environment: str = Field(default="local", env="ENVIRONMENT")

    # Core configuration
    host: str = Field(default="0.0.0.0", env="SERVICE_HOST")
    port: int = Field(default=8090, env="SERVICE_PORT")

    # Sub-components (optional — None if not configured)
    feature_store: Optional[FeatureStoreSettings] = None
    triton: Optional[TritonSettings] = None
    redis: Optional[RedisSettings] = None
    mlflow: Optional[MLflowSettings] = None

    # Model inference defaults
    default_model_version: str = Field(default="latest", env="DEFAULT_MODEL_VERSION")
    inference_timeout_ms: int = Field(default=200, env="INFERENCE_TIMEOUT_MS")
    enable_shap: bool = Field(default=True, env="ENABLE_SHAP")
    fallback_enabled: bool = Field(default=True, env="FALLBACK_ENABLED")

    # A/B testing
    ab_test_enabled: bool = Field(default=False, env="AB_TEST_ENABLED")
    champion_version: str = Field(default="v5", env="CHAMPION_MODEL_VERSION")
    challenger_version: Optional[str] = Field(default=None, env="CHALLENGER_MODEL_VERSION")
    challenger_traffic_pct: int = Field(default=10, env="CHALLENGER_TRAFFIC_PCT")

    # Monitoring
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    enable_opentelemetry: bool = Field(default=False, env="ENABLE_OPENTELEMETRY")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    @field_validator('challenger_traffic_pct')
    @classmethod
    def validate_traffic_pct(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('traffic percentage must be 0-100')
        return v


# Global settings instance
settings = Settings()