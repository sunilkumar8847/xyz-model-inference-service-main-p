# src/adapters/feature_store_client.py
"""Feature Store (Feast) client."""

import logging
import re
from typing import List, Optional

from src.domain.models import EntityAttributes, FeatureVector

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """Feast feature store client."""
    
    async def get_features(
        self,
        entity_id_1: str,
        entity_id_2: str
    ) -> FeatureVector:
        """
        Get pre-computed feature vector for entity pair.
        
        Returns 50-dim feature vector:
        - 15 string similarity features
        - 5 phonetic features
        - 8 token-based features
        - 10 semantic (embedding) features
        - 7 structural features
        - 5 domain-specific features
        """
        # Placeholder: would fetch from Feast online store (Redis)
        from datetime import datetime
        
        return FeatureVector(
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            features=[0.5] * 50,  # Mock features
            feature_names=[
                f"feature_{i}" for i in range(50)
            ],
            timestamp=datetime.now()
        )

    async def get_entity_attributes(self, entity_id: str) -> Optional[EntityAttributes]:
        """
        Fetch raw entity attributes needed by the fallback rule engine.
        In production this hits the Feast online store (Redis-backed).
        Returns None when the entity is not found.
        """
        try:
            import httpx
            from src.config.settings import settings

            url = f"{settings.feature_store.url}/v1/entities/{entity_id}/attributes"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()

            phone_raw = data.get("phone", "")
            phone_normalised = re.sub(r"\D", "", phone_raw) if phone_raw else None

            return EntityAttributes(
                entity_id=entity_id,
                email=data.get("email"),
                phone=phone_normalised or None,
                full_name=data.get("full_name"),
                address_line=data.get("address_line"),
                city=data.get("city"),
                postal_code=data.get("postal_code"),
            )
        except Exception as exc:
            logger.warning("Could not fetch entity attributes for %s: %s", entity_id, exc)
            return None