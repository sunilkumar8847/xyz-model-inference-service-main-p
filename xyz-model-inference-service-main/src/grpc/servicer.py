"""gRPC servicer — implements ModelInferenceService RPCs."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import grpc

from src.grpc import inference_pb2, inference_pb2_grpc

if TYPE_CHECKING:
    from src.services.inference_service import InferenceService
    from src.adapters.triton_client import TritonClient

logger = logging.getLogger(__name__)

_TENANT_HEADER = "x-verified-tenant-id"
_USER_HEADER = "x-verified-user-id"


def _require_auth(context: grpc.ServicerContext) -> bool:
    """Validate Envoy-injected auth metadata. Aborts with UNAUTHENTICATED if missing."""
    meta = dict(context.invocation_metadata())
    if not meta.get(_TENANT_HEADER) or not meta.get(_USER_HEADER):
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            f"Missing required metadata: {_TENANT_HEADER} and {_USER_HEADER}",
        )
        return False
    return True


class ModelInferenceServicer(inference_pb2_grpc.ModelInferenceServiceServicer):
    """
    Implements all four LLD-specified gRPC RPCs:
      Predict, PredictBatch, GetEmbedding, Explain
    """

    def __init__(self, inference_service: "InferenceService", triton_client: "TritonClient"):
        self._svc = inference_service
        self._triton = triton_client

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    async def Predict(
        self,
        request: inference_pb2.PredictRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PredictResponse:
        if not _require_auth(context):
            return inference_pb2.PredictResponse()

        try:
            result = await self._svc.predict_match(
                entity_id_1=request.entity_id_1,
                entity_id_2=request.entity_id_2,
                model_version=request.model_version or None,
                include_shap=request.include_explanation,
                timeout_ms=request.timeout_ms or None,
            )

            resp = inference_pb2.PredictResponse(
                match_probability=result.score,
                decision=result.decision,
                model_version=result.model_version,
                inference_ms=result.inference_ms,
                confidence_lower=result.confidence_interval[0],
                confidence_upper=result.confidence_interval[1],
                fallback_used=result.fallback_used,
                fallback_reason=result.fallback_reason or "",
            )

            if request.include_explanation and result.explanation:
                resp.explanation.CopyFrom(
                    _explanation_to_proto(result.explanation, result.score, result.model_version)
                )

            return resp

        except asyncio.TimeoutError:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Inference timeout")
        except Exception as exc:
            logger.error("Predict RPC error: %s", exc, exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return inference_pb2.PredictResponse()

    # ------------------------------------------------------------------
    # PredictBatch
    # ------------------------------------------------------------------
    async def PredictBatch(
        self,
        request: inference_pb2.PredictBatchRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PredictBatchResponse:
        if not _require_auth(context):
            return inference_pb2.PredictBatchResponse()

        if not request.pairs:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "pairs list must not be empty")
            return inference_pb2.PredictBatchResponse()

        start = time.monotonic()
        try:
            tasks = [
                self._svc.predict_match(
                    entity_id_1=p.entity_id_1,
                    entity_id_2=p.entity_id_2,
                    model_version=p.model_version or None,
                    include_shap=p.include_explanation,
                    timeout_ms=p.timeout_ms or None,
                )
                for p in request.pairs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            responses = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Batch item error: %s", result)
                    responses.append(
                        inference_pb2.PredictResponse(
                            fallback_used=True,
                            fallback_reason="batch_item_error",
                        )
                    )
                else:
                    responses.append(
                        inference_pb2.PredictResponse(
                            match_probability=result.score,
                            decision=result.decision,
                            model_version=result.model_version,
                            inference_ms=result.inference_ms,
                            confidence_lower=result.confidence_interval[0],
                            confidence_upper=result.confidence_interval[1],
                            fallback_used=result.fallback_used,
                            fallback_reason=result.fallback_reason or "",
                        )
                    )

            total_ms = int((time.monotonic() - start) * 1000)
            return inference_pb2.PredictBatchResponse(
                results=responses,
                total=len(responses),
                total_inference_ms=total_ms,
            )

        except Exception as exc:
            logger.error("PredictBatch RPC error: %s", exc, exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return inference_pb2.PredictBatchResponse()

    # ------------------------------------------------------------------
    # GetEmbedding
    # ------------------------------------------------------------------
    async def GetEmbedding(
        self,
        request: inference_pb2.GetEmbeddingRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.GetEmbeddingResponse:
        if not _require_auth(context):
            return inference_pb2.GetEmbeddingResponse()

        if not request.entity_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "entity_id is required")
            return inference_pb2.GetEmbeddingResponse()

        start = time.monotonic()
        try:
            # Get features from feature store if not provided in request
            if request.features:
                features = list(request.features)
            else:
                fv = await self._svc.feature_store.get_features(request.entity_id, "")
                features = fv.features

            embedding = await self._triton.get_embedding(
                features=features,
                model_version=request.model_version or None,
            )

            inference_ms = int((time.monotonic() - start) * 1000)
            return inference_pb2.GetEmbeddingResponse(
                entity_id=request.entity_id,
                embedding=embedding,
                dimension=len(embedding),
                model_version=request.model_version or "champion",
                inference_ms=inference_ms,
            )

        except Exception as exc:
            logger.error("GetEmbedding RPC error: %s", exc, exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return inference_pb2.GetEmbeddingResponse()

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------
    async def Explain(
        self,
        request: inference_pb2.ExplainRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.ExplainResponse:
        if not _require_auth(context):
            return inference_pb2.ExplainResponse()

        try:
            # Get features if not provided
            if request.features:
                features = list(request.features)
                feature_names = list(request.feature_names)
            else:
                fv = await self._svc.feature_store.get_features(
                    request.entity_id_1, request.entity_id_2
                )
                features = fv.features
                feature_names = fv.feature_names

            explanation = await self._triton.explain(
                features=features,
                model_version=request.model_version or None,
                field_names=feature_names,
            )

            if explanation is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "SHAP model not available")
                return inference_pb2.ExplainResponse()

            # Also get the match probability for context
            score_result = await self._svc.triton.predict(
                features=features,
                model_version=request.model_version or "champion",
                timeout_ms=200,
            )

            return _explanation_to_proto(explanation, score_result.score, request.model_version)

        except Exception as exc:
            logger.error("Explain RPC error: %s", exc, exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

        return inference_pb2.ExplainResponse()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explanation_to_proto(explanation, match_probability: float, model_version: str) -> inference_pb2.ExplainResponse:
    contributions = []
    shap_vals = explanation.get("shap_values", {})
    total_abs = sum(abs(v) for v in shap_vals.values()) or 1.0

    for name, val in shap_vals.items():
        contributions.append(
            inference_pb2.FieldContribution(
                field_name=name,
                shap_value=val,
                contribution_pct=round(abs(val) / total_abs * 100, 2),
            )
        )
    # Sort by absolute attribution descending
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return inference_pb2.ExplainResponse(
        contributions=contributions,
        match_probability=match_probability,
        human_readable=explanation.get("human_readable", ""),
        model_version=model_version or "champion",
    )
