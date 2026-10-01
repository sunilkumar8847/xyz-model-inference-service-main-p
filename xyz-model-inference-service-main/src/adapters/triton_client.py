# src/adapters/triton_client.py
"""Triton Inference Server gRPC client."""

import logging
from typing import List, Optional
import grpc
from tritonclient.grpc import InferenceServerClient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
import numpy as np

from src.config.settings import settings
from src.domain.models import MatchScore

logger = logging.getLogger(__name__)


class TritonClient:
    """Triton Inference Server client wrapper."""
    
    def __init__(self):
        if settings.triton is None:
            # Local/dev mode – Triton not configured
            self.enabled = False
            self._client = None
            self.host = None
            self.port = None
            return
        self.enabled = True
        self.host = settings.triton.host
        self.port = settings.triton.grpc_port
        self.timeout = settings.triton.timeout_ms / 1000.0
        # self._client: Optional[InferenceServerClient] = None
        self._client = None
    
    async def initialize(self) -> None:
        """Connect to Triton server."""
        if not self.enabled:
            # Local/dev mode – skip Triton initialization
            return

        try:
            self._client = InferenceServerClient(
                url=f"{self.host}:{self.port}",
                timeout=settings.triton.timeout_ms / 1000,
            )
            self._client.is_server_live()
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise
        # try:
        #     self._client = InferenceServerClient(
        #         url=f"{self.host}:{self.port}",
        #         verbose=False
        #     )
        #     self._client.is_server_live()
        #     logger.info(f"Connected to Triton at {self.host}:{self.port}")
        # except Exception as e:
        #     logger.error(f"Failed to connect to Triton: {e}")
        #     raise
    
    async def predict(
        self,
        features: List[float],
        model_version: str,
        timeout_ms: int
    ) -> MatchScore:
        """
        Run inference on matching model.
        
        Input: 50-dim feature vector
        Output: Match score [0, 1]
        """
        import asyncio
        
        try:
            # Prepare input tensor
            input_data = np.array([features], dtype=np.float32)
            inputs = [
                grpc.InferInput(
                    name="input",
                    shape=input_data.shape,
                    datatype="FP32"
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            # Run inference with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.infer,
                    model_name=f"customer_matcher_{model_version}",
                    inputs=inputs
                ),
                timeout=timeout_ms / 1000.0
            )
            
            # Extract output (score, confidence_lower, confidence_upper)
            output_data = response.as_numpy("output")
            score = float(output_data[0][0])
            confidence_lower = float(output_data[0][1])
            confidence_upper = float(output_data[0][2])
            
            return MatchScore(
                score=score,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                model_version=model_version,
                inference_ms=0  # Would be filled by caller
            )
        
        except asyncio.TimeoutError:
            logger.warning(f"Triton inference timeout (>{timeout_ms}ms)")
            raise
        
        except Exception as e:
            logger.error(f"Triton inference error: {e}")
            raise
    
    async def explain(
        self,
        features: List[float],
        model_version: str,
        field_names: List[str]
    ) -> Optional[dict]:
        """
        Get SHAP explanations from Triton.

        Calls the `customer_matcher_shap` ensemble model which returns one
        SHAP value per input feature.  Falls back to kernel-SHAP (perturbation)
        when the dedicated SHAP model is unavailable.
        """
        import asyncio

        if not self.enabled or self._client is None:
            return None

        shap_model = f"customer_matcher_shap_{model_version}" if model_version else "customer_matcher_shap"

        try:
            shap_values = await self._call_shap_model(features, shap_model)
        except Exception as exc:
            logger.warning("SHAP model '%s' unavailable (%s); falling back to perturbation", shap_model, exc)
            shap_values = await self._kernel_shap(features, model_version)

        if shap_values is None:
            return None

        total_abs = sum(abs(v) for v in shap_values) or 1.0
        contributions = {
            name: round(float(val), 6)
            for name, val in zip(field_names, shap_values)
        }

        # Top-5 by absolute attribution for the human-readable summary
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top5 = ranked[:5]
        human_readable = "Top drivers: " + ", ".join(
            f"{name} ({'+' if val >= 0 else ''}{val:.3f})" for name, val in top5
        )

        return {
            "shap_values": contributions,
            "human_readable": human_readable,
        }

    async def _call_shap_model(self, features: List[float], shap_model_name: str) -> Optional[List[float]]:
        """Call dedicated Triton SHAP ensemble model."""
        import asyncio

        input_data = np.array([features], dtype=np.float32)
        inputs = [grpc.InferInput(name="input", shape=input_data.shape, datatype="FP32")]
        inputs[0].set_data_from_numpy(input_data)

        response = await asyncio.wait_for(
            asyncio.to_thread(
                self._client.infer,
                model_name=shap_model_name,
                inputs=inputs,
            ),
            timeout=self.timeout,
        )
        shap_output = response.as_numpy("shap_values")
        return shap_output[0].tolist()

    async def _kernel_shap(self, features: List[float], model_version: str) -> Optional[List[float]]:
        """
        Perturbation-based kernel SHAP approximation.
        Masks each feature with its mean (0.0) and measures impact on score.
        Returns marginal contribution per feature.
        """
        import asyncio

        baseline_score = None
        try:
            result = await self.predict(features, model_version, int(self.timeout * 1000))
            baseline_score = result.score
        except Exception:
            return None

        shap_values = []
        for i in range(len(features)):
            masked = list(features)
            masked[i] = 0.0  # mask with feature mean baseline
            try:
                masked_result = await self.predict(masked, model_version, int(self.timeout * 1000))
                shap_values.append(round(baseline_score - masked_result.score, 6))
            except Exception:
                shap_values.append(0.0)

        return shap_values

    async def get_embedding(
        self,
        features: List[float],
        model_version: Optional[str] = None,
    ) -> List[float]:
        """
        Return the 128-dim Transformer embedding for a feature vector.
        Calls the `customer_matcher_embed` Triton model.
        """
        import asyncio

        if not self.enabled or self._client is None:
            return [0.0] * 128

        embed_model = f"customer_matcher_embed_{model_version}" if model_version else "customer_matcher_embed"
        input_data = np.array([features], dtype=np.float32)
        inputs = [grpc.InferInput(name="input", shape=input_data.shape, datatype="FP32")]
        inputs[0].set_data_from_numpy(input_data)

        response = await asyncio.wait_for(
            asyncio.to_thread(
                self._client.infer,
                model_name=embed_model,
                inputs=inputs,
            ),
            timeout=self.timeout,
        )
        embedding = response.as_numpy("embedding")
        return embedding[0].tolist()
    
    async def health_check(self) -> bool:
        """Check if Triton is healthy."""
        try:
            return self._client.is_server_live()
        except Exception:
            return False
