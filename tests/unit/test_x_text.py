"""Small documented corpus for the local X weighted-length approximation."""

import pytest

from app.domain.x_text import weighted_length


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain ASCII", 11),
        ("https://example.test/path", 23),
        ("Link https://example.test/path", 28),
        ("🙂", 2),
        ("漢字", 4),
        ("A🙂漢 https://example.test", 29),
        ("https://one.test https://two.test", 47),
    ],
)
def test_weighted_length_reference_corpus(text: str, expected: int) -> None:
    assert weighted_length(text) == expected


def test_bare_domain_is_not_claimed_as_a_transformed_url() -> None:
    assert weighted_length("example.test") == len("example.test")
