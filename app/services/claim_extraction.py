"""Provider-neutral final-text claim extraction.

Live providers may propose structured claims, but every proposal is validated against an exact span
in the rendered text. The deterministic extractor is always retained as a fail-closed baseline.
"""

from __future__ import annotations

from typing import Protocol

from app.schemas.content import CanonicalClaim, ClaimExtractionContext
from app.services.claim_structure import extract_claims, normalize_text, reconcile_claims


class ClaimExtractorPort(Protocol):
    async def extract_claims(
        self,
        text: str,
        *,
        context: ClaimExtractionContext,
    ) -> tuple[CanonicalClaim, ...]: ...


class DeterministicClaimExtractor:
    async def extract_claims(
        self,
        text: str,
        *,
        context: ClaimExtractionContext,
    ) -> tuple[CanonicalClaim, ...]:
        del context
        return extract_claims(text)


class MockClaimExtractor(DeterministicClaimExtractor):
    """Explicit offline implementation used by mock workflows and tests."""


def validate_provider_claims(
    text: str,
    claims: tuple[CanonicalClaim, ...],
) -> tuple[CanonicalClaim, ...]:
    """Keep only claims grounded in an exact rendered-text span."""

    valid: list[CanonicalClaim] = []
    for claim in claims:
        if claim.start is not None and claim.end is not None:
            if claim.end > len(text) or text[claim.start : claim.end] != claim.exact_text:
                continue
        elif claim.exact_text not in text:
            continue
        if any(
            normalize_text(item.exact_text) == normalize_text(claim.exact_text) for item in valid
        ):
            continue
        valid.append(claim)
    return tuple(valid)


def merge_extractor_results(
    text: str,
    deterministic: tuple[CanonicalClaim, ...],
    provider: tuple[CanonicalClaim, ...] | None,
) -> tuple[CanonicalClaim, ...]:
    if provider is None:
        return deterministic
    return reconcile_claims(text, deterministic, validate_provider_claims(text, provider))
