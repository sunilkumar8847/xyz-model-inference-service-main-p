"""Rule-based fallback matching when AI inference is unavailable."""
from __future__ import annotations

import logging
import re
from typing import Optional, TYPE_CHECKING

import jellyfish

from src.domain.models import EntityAttributes, FallbackReason, PredictionResult

if TYPE_CHECKING:
    from src.adapters.feature_store_client import FeatureStoreClient

logger = logging.getLogger(__name__)

# Rule tier thresholds (LLD §9.2)
_JW_NAME_HIGH = 0.90
_JW_ADDR_HIGH = 0.80

# Score constants
_SCORE_EMAIL_EXACT = 0.99
_SCORE_PHONE_NAME  = 0.95
_SCORE_NAME_ADDR   = 0.85
_SCORE_NAME_ONLY   = 0.70
_SCORE_NO_DATA     = 0.50   # conservative default when no entity data available


class FallbackService:
    """Rule-based matching fallback when AI inference is unavailable."""

    def __init__(self, feature_store_client: Optional["FeatureStoreClient"] = None):
        self._feature_store = feature_store_client

    async def predict_match(
        self,
        entity_id_1: str,
        entity_id_2: str,
        reason: Optional[FallbackReason] = None,
    ) -> PredictionResult:
        """
        Rule-based matching (LLD §9.2 priority order):
          1. Exact email match              → 0.99
          2. Phone match + name JW ≥ 0.90  → 0.95
          3. Name JW ≥ 0.90 + addr JW ≥ 0.80 → 0.85
          4. Name Jaro-Winkler ≥ 0.90      → 0.70
          5. No data available              → 0.50 (UNCERTAIN)
        """
        attr1, attr2 = await self._fetch_pair(entity_id_1, entity_id_2)
        score = self._apply_rules(attr1, attr2)

        logger.info(
            "Fallback match: %s vs %s = %.3f (reason: %s)",
            entity_id_1, entity_id_2, score,
            reason.value if reason else "unknown",
        )

        return PredictionResult(
            score=score,
            confidence_interval=(max(0.0, score - 0.05), min(1.0, score + 0.05)),
            decision=self._classify_decision(score),
            model_version="fallback_rules_v1",
            inference_ms=5,
            fallback_used=True,
            fallback_reason=reason.value if reason else None,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_pair(
        self, entity_id_1: str, entity_id_2: str
    ) -> tuple[Optional[EntityAttributes], Optional[EntityAttributes]]:
        if self._feature_store is None:
            return None, None
        import asyncio
        attr1, attr2 = await asyncio.gather(
            self._feature_store.get_entity_attributes(entity_id_1),
            self._feature_store.get_entity_attributes(entity_id_2),
        )
        return attr1, attr2

    def _apply_rules(
        self,
        a: Optional[EntityAttributes],
        b: Optional[EntityAttributes],
    ) -> float:
        """Apply LLD §9.2 rule tiers and return the highest matching score."""
        if a is None or b is None:
            return _SCORE_NO_DATA

        # Tier 1 — exact email match
        if a.email and b.email and _normalise_email(a.email) == _normalise_email(b.email):
            return _SCORE_EMAIL_EXACT

        # Tier 2 — phone match + name Jaro-Winkler ≥ 0.90
        name_jw = _jw(a.full_name, b.full_name)
        if _phone_match(a.phone, b.phone) and name_jw >= _JW_NAME_HIGH:
            return _SCORE_PHONE_NAME

        # Tier 3 — name JW ≥ 0.90 + address JW ≥ 0.80
        addr_jw = _jw(_full_address(a), _full_address(b))
        if name_jw >= _JW_NAME_HIGH and addr_jw >= _JW_ADDR_HIGH:
            return _SCORE_NAME_ADDR

        # Tier 4 — name JW ≥ 0.90 alone
        if name_jw >= _JW_NAME_HIGH:
            return _SCORE_NAME_ONLY

        # Below threshold — score proportionally from name similarity
        return round(name_jw * _SCORE_NAME_ONLY, 4)

    def _classify_decision(self, score: float) -> str:
        if score >= 0.95:
            return "AUTO_MATCH"
        elif score >= 0.80:
            return "LIKELY_MATCH"
        elif score >= 0.50:
            return "UNCERTAIN"
        else:
            return "AUTO_NON_MATCH"


# ---------------------------------------------------------------------------
# Pure helpers — no I/O, fully testable
# ---------------------------------------------------------------------------

def _normalise_email(email: str) -> str:
    return email.strip().lower()


def _phone_match(p1: Optional[str], p2: Optional[str]) -> bool:
    """Match last 10 digits of normalised phone strings."""
    if not p1 or not p2:
        return False
    return p1[-10:] == p2[-10:]


def _full_address(a: EntityAttributes) -> str:
    parts = [a.address_line, a.city, a.postal_code]
    return " ".join(p for p in parts if p).lower()


def _jw(s1: Optional[str], s2: Optional[str]) -> float:
    """Jaro-Winkler similarity in [0, 1]; returns 0.0 when either string is absent."""
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(s1.strip().lower(), s2.strip().lower())
