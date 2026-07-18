from types import SimpleNamespace

import pytest

from app.models.enums import ContentType, FactCheckStatus
from app.services.publishing_service import PublicationGateError, PublishingService


def _quote_service(*, enabled: bool) -> PublishingService:
    service = object.__new__(PublishingService)
    service.enterprise_quote_posts_enabled = enabled
    service.drafts = SimpleNamespace(
        current_version=lambda _draft: SimpleNamespace(
            generation_metadata={
                "source_context": {
                    "strategy": "quote_post",
                    "source_post_id": "2076668264755732487",
                }
            },
            content={
                "parts": ["the wording is doing a lot of work here"],
                "weighted_lengths": [42],
            },
        ),
        artifacts=SimpleNamespace(validate_media_manifest=lambda *_args, **_kwargs: {"files": []}),
    )
    service._successful_prefix = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    service._validate_state = lambda *_args, **_kwargs: SimpleNamespace(id="approval-1")  # type: ignore[method-assign]
    service._estimate = lambda **_kwargs: None  # type: ignore[method-assign]
    return service


def _draft() -> SimpleNamespace:
    return SimpleNamespace(
        id="draft-1",
        current_content_hash="a" * 64,
        content_type=ContentType.SHORT_POST,
        fact_check_status=FactCheckStatus.NOT_REQUIRED,
        media_plan={"type": "none", "required_files": [], "metadata": {}},
        artifact_path="synthetic-draft-1",
        idea_id="idea-1",
    )


def test_quote_context_requires_enterprise_api_access() -> None:
    service = _quote_service(enabled=False)

    with pytest.raises(PublicationGateError, match="X Enterprise access"):
        service._preview_from_account(
            _draft(),
            account_id="account-1",
            account_username="configured_account",
        )


def test_enterprise_quote_context_prepares_quote_publication_preview() -> None:
    service = _quote_service(enabled=True)

    preview = service._preview_from_account(
        _draft(),
        account_id="account-1",
        account_username="configured_account",
    )

    assert preview.action == "quote_post"
    assert preview.quote_post_id == "2076668264755732487"
    assert preview.parts == ("the wording is doing a lot of work here",)
