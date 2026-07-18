"""Provider boundary deliberately exposes no tools or publication capability."""

from __future__ import annotations

from typing import Protocol

from app.schemas.content import GenerationRequest, GenerationResult


class LLMProvider(Protocol):
    """The only operation a content-generation provider may perform."""

    async def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        """Return a validated structured proposal without side effects."""
        ...
