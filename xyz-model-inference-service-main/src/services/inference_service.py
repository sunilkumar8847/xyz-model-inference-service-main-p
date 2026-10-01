# src/services/inference_service.py
"""Core inference logic: feature extraction, model scoring, decision logic."""

import asyncio
import logging
import time
from typing import Optional, Tuple
from datetime import datetime

from src.config.settings import settings
from src.domain.models import (
    FeatureVector, MatchScore, PredictionResult, 
    ExplanationResult, InferenceDecision, FallbackReason
)
from src.adapters.triton_client import TritonClient
from src.adapters.feature_store_client import FeatureStoreClient
from src.services.model_manager import ModelManager
from src.services.fallback_service import FallbackService
from src.adapters.circuit_breaker import CircuitBreaker
from src.infrastructure.observability import inference_latency, inference_errors, fallback_rate

logger = logging.getLogger(__name__)


class InferenceService:
    """High-level inference orchestration."""
    
    def __init__(
        self,
        triton_client: TritonClient,
        feature_store_client: FeatureStoreClient,
        model_manager: ModelManager,
        fallback_service: FallbackService
    ):
        self.triton = triton_client
        self.feature_store = feature_store_client
        self.model_manager = model_manager
        self.fallback = fallback_service
        self._triton_cb = CircuitBreaker(
            name="triton",
            failure_threshold=0.5,
            recovery_timeout_s=30,
        )
    
    async def predict_match(
        self,
        entity_id_1: str,
        entity_id_2: str,
        model_version: Optional[str] = None,
        include_shap: bool = True,
        timeout_ms: Optional[int] = None
    ) -> PredictionResult:
        """
        Predict if two entities are duplicates (matches).
        
        Args:
            entity_id_1: First entity ID
            entity_id_2: Second entity ID
            model_version: Specific model version (or use A/B test routing)
            include_shap: Include SHAP explanations
            timeout_ms: Inference timeout in milliseconds
        
        Returns:
            PredictionResult with score, decision, explanation
        """
        start_time = time.time()
        timeout_ms = timeout_ms or settings.inference_timeout_ms
        
        try:
            # 1. Get features from feature store
            features = await self.feature_store.get_features(
                entity_id_1, entity_id_2
            )
            
            # 2. Select model version (A/B test routing)
            model_version = self.model_manager.get_model_version()
            
            # 3. Call Triton inference via circuit breaker
            score_result = await self._triton_cb.call(
                self.triton.predict,
                features=features.features,
                model_version=model_version,
                timeout_ms=timeout_ms,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 4. Calculate match decision
            decision = self._classify_decision(score_result.score)
            
            # 5. Get SHAP explanation (optional, fast path)
            explanation = None
            if include_shap and elapsed_ms < (timeout_ms * 0.7):
                explanation = await self.triton.explain(
                    features=features.features,
                    model_version=model_version,
                    field_names=features.feature_names
                )
            
            result = PredictionResult(
                score=score_result.score,
                confidence_interval=(
                    score_result.confidence_lower,
                    score_result.confidence_upper
                ),
                decision=decision,
                model_version=model_version,
                inference_ms=int(elapsed_ms),
                explanation=explanation
            )
            
            inference_latency.observe(elapsed_ms)
            logger.debug(f"Match prediction: {entity_id_1} vs {entity_id_2} = {score_result.score:.3f}")
            
            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"Inference timeout for {entity_id_1} vs {entity_id_2}")
            inference_errors.labels(error_type="timeout").inc()
            fallback_rate.labels(reason="timeout").inc()
            if settings.fallback_enabled:
                return await self.fallback.predict_match(
                    entity_id_1, entity_id_2,
                    reason=FallbackReason.TIMEOUT
                )
            raise

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            inference_errors.labels(error_type="error").inc()
            fallback_rate.labels(reason="model_error").inc()
            if settings.fallback_enabled:
                return await self.fallback.predict_match(
                    entity_id_1, entity_id_2,
                    reason=FallbackReason.MODEL_ERROR
                )
            raise
    
    def _classify_decision(self, score: float) -> str:
        """Classify match decision based on score."""
        if score >= 0.95:
            return InferenceDecision.AUTO_MATCH.value
        elif score >= 0.85:
            return InferenceDecision.LIKELY_MATCH.value
        elif score >= 0.50:
            return InferenceDecision.UNCERTAIN.value
        elif score >= 0.15:
            return InferenceDecision.LIKELY_NON.value
        else:
            return InferenceDecision.AUTO_NON_MATCH.value
    
    async def batch_predict(
        self,
        entity_pairs: list[Tuple[str, str]],
        timeout_ms: Optional[int] = None
    ) -> list[PredictionResult]:
        """Batch inference for multiple entity pairs."""
        tasks = [
            self.predict_match(e1, e2, timeout_ms=timeout_ms)
            for e1, e2 in entity_pairs
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)
