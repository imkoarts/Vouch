from pathlib import Path

import pytest

from app.domain.enums import ContentType
from app.providers.mock_provider import MockLLMProvider
from app.services.configuration import ConfigurationService
from app.services.draft_service import DraftService, DraftValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _service(*, premium_enabled: bool = True) -> DraftService:
    profile = ConfigurationService(PROJECT_ROOT / "config").load_profile()
    return DraftService(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        MockLLMProvider(),
        profile=profile,
        max_weighted_length=280,
        premium_long_posts_enabled=premium_enabled,
        premium_long_post_max_chars=25_000,
    )


def test_premium_long_post_accepts_more_than_standard_limit() -> None:
    service = _service()

    lengths = service._validate_lengths(("a" * 500,), ContentType.LONG_POST)

    assert lengths == [500]


def test_premium_long_post_rejects_more_than_25000_characters() -> None:
    service = _service()

    with pytest.raises(DraftValidationError, match="Premium long post"):
        service._validate_lengths(("a" * 25_001,), ContentType.LONG_POST)


def test_short_post_and_nonpremium_long_post_keep_standard_limit() -> None:
    premium = _service()
    standard = _service(premium_enabled=False)

    with pytest.raises(DraftValidationError, match="weighted X length"):
        premium._validate_lengths(("a" * 281,), ContentType.SHORT_POST)
    with pytest.raises(DraftValidationError, match="weighted X length"):
        standard._validate_lengths(("a" * 281,), ContentType.LONG_POST)
