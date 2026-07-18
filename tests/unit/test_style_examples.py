from __future__ import annotations

from pathlib import Path

from app.services.style_examples import load_curated_style_examples

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_curated_style_examples_are_mode_specific_and_versioned() -> None:
    examples = load_curated_style_examples(
        PROJECT_ROOT / "config" / "style_examples.yml",
        editorial_intent="comment_on_source",
        max_approved=3,
        max_rejected=2,
    )

    assert examples.approved == (
        "The scores can be exported, but the evaluation cannot be rerun without the test cases.",
    )
    assert len(examples.rejected) == 2
    assert "named officeholder" in examples.rejected[0]
