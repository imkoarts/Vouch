from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.image_generation import (
    ImageGenerationDisabledError,
    ImageGenerationService,
)


class FakeImages:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(self.payload).decode("ascii"))]
        )


class FakeDrafts:
    def __init__(self) -> None:
        self.attached: dict[str, Any] | None = None

    def get(self, draft_id: str) -> Any:
        return SimpleNamespace(id=draft_id)

    def current_version(self, _draft: Any) -> Any:
        return SimpleNamespace(rendered_text="Text for an illustration")

    def attach_image_bytes(self, draft_id: str, **kwargs: Any) -> str:
        self.attached = {"draft_id": draft_id, **kwargs}
        return "media/generated.png"


@pytest.mark.asyncio
async def test_image_generation_is_fail_closed_when_disabled() -> None:
    service = ImageGenerationService(
        drafts=FakeDrafts(),  # type: ignore[arg-type]
        client_factory=None,
        enabled=False,
        model="gpt-image-2",
        size="1536x1024",
        quality="medium",
        output_format="png",
    )

    with pytest.raises(ImageGenerationDisabledError, match="disabled"):
        await service.generate_for_draft("draft-1")


@pytest.mark.asyncio
async def test_image_generation_attaches_returned_bytes_to_draft() -> None:
    drafts = FakeDrafts()
    images = FakeImages(b"synthetic image")
    client = SimpleNamespace(images=images)
    service = ImageGenerationService(
        drafts=drafts,  # type: ignore[arg-type]
        client_factory=lambda: client,
        enabled=True,
        model="gpt-image-2",
        size="1536x1024",
        quality="medium",
        output_format="png",
    )

    path = await service.generate_for_draft("draft-1")

    assert path == "media/generated.png"
    assert drafts.attached is not None
    assert drafts.attached["image_bytes"] == b"synthetic image"
    assert drafts.attached["extension"] == "png"
    assert images.calls[0]["model"] == "gpt-image-2"
    assert images.calls[0]["output_format"] == "png"
    assert "response_format" not in images.calls[0]
