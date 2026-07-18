from __future__ import annotations

from app.schemas.content import GenerationVariant
from app.services.editorial_quality import assess_variant


def _variant(text: str) -> GenerationVariant:
    return GenerationVariant(
        label="standalone-v350",
        text=text,
        character_count=len(text),
        similarity_risk=0.0,
        angle_id="source-event-bounded-interpretation",
        angle_type="practical_implication",
        evidence_ids=("evidence-1",),
        confidence="medium",
        factual_claims=(),
        attributed_claims=(),
        uncertainty_markers=(),
    )


def _codes(text: str) -> set[str]:
    return {issue.code for issue in assess_variant(_variant(text)).issues}


def test_v350_rejects_abstract_test_packaging() -> None:
    codes = _codes("The tournament gives the format a narrow test of market legibility.")
    assert {"ABSTRACT_TEST_FRAMING", "ANALYTICAL_PACKAGING"} <= codes


def test_v350_rejects_manufactured_question_and_reveal() -> None:
    codes = _codes(
        "The practical question is not only whether more categories can be listed, "
        "but whether the combination remains legible."
    )
    assert {"MANUFACTURED_QUESTION", "NOT_ONLY_BUT_REVEAL", "STAGED_THESIS"} <= codes


def test_v350_rejects_product_memo_voice() -> None:
    codes = _codes("The broader implication is a new operating model for product adoption.")
    assert "PRODUCT_MEMO_VOICE" in codes


def test_v350_preserves_plain_standalone_observation() -> None:
    codes = _codes(
        "Adding categories will not help if traders cannot tell what each price includes."
    )
    assert (
        not {
            "ABSTRACT_TEST_FRAMING",
            "ANALYTICAL_PACKAGING",
            "MANUFACTURED_QUESTION",
            "NOT_ONLY_BUT_REVEAL",
            "PRODUCT_MEMO_VOICE",
            "MINI_ESSAY_ARC",
        }
        & codes
    )
