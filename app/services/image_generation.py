"""Optional image generation kept behind an explicit feature flag."""

from __future__ import annotations

import base64
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from app.services.draft_service import DraftService


class ImageGenerationDisabledError(PermissionError):
    pass


@dataclass(slots=True)
class ImageGenerationService:
    drafts: DraftService
    client_factory: Callable[[], Any] | None
    enabled: bool
    model: str
    size: str
    quality: str
    output_format: str

    async def generate_for_draft(self, draft_id: str, *, prompt: str | None = None) -> str:
        if not self.enabled or self.client_factory is None:
            raise ImageGenerationDisabledError("Image generation is disabled in config/runtime.yml")
        draft = self.drafts.get(draft_id)
        version = self.drafts.current_version(draft)
        source_text = version.rendered_text
        image_prompt = prompt or (
            "Create a clean editorial illustration for an X post. No logos, no watermarks, "
            "no fake screenshots, and no text unless essential. Visualize this post naturally:\n\n"
            + source_text
        )
        client = self.client_factory()
        try:
            response = await client.images.generate(
                model=self.model,
                prompt=image_prompt,
                n=1,
                size=self.size,
                quality=cast(Literal["low", "medium", "high", "auto"], self.quality),
                output_format=cast(Literal["png", "jpeg", "webp"], self.output_format),
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("The image provider returned no image bytes")
        image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
        return self.drafts.attach_image_bytes(
            draft_id,
            image_bytes=image_bytes,
            extension=self.output_format,
            reason="Generated illustration for Telegram review",
            actor="telegram",
        )
