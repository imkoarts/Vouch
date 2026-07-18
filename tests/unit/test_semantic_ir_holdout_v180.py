from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_reply_shell,
)
from app.services.semantic_extraction import extract_semantic_inspection

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "semantic_ir_holdout_v180.json"


def _cases() -> tuple[dict[str, str], ...]:
    return tuple(json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"])


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_closed_semantic_ir_holdout(case: dict[str, str]) -> None:
    inspection = extract_semantic_inspection(case["source"], case["reply"])
    harm = adjudicate_human_harm(inspection)
    irony = adjudicate_evaluative_irony(inspection)
    shell = adjudicate_reply_shell(inspection)

    match case["kind"]:
        case "sarcasm":
            assert irony.implicit_sarcasm
        case "literal_praise":
            assert irony.positive_evaluation
            assert irony.literal_positive_reading_plausible
            assert not irony.implicit_sarcasm
        case "human_harm":
            assert harm.harm_referent == "human"
            assert harm.literal_harm
        case "nonhuman":
            assert harm.harm_referent == "non_human"
            assert not harm.literal_harm
        case "mixed":
            assert harm.harm_referent == "human"
            assert harm.metaphorical_or_technical_usage
        case "shell":
            assert shell.operator in {"decorated_identity", "repackaging", "reveal"}
            assert shell.generic_reusability
        case "no_harm":
            assert not harm.literal_harm
        case "not_person":
            assert harm.harm_referent != "human"
        case "no_shell":
            assert shell.operator == "none"
        case _:
            raise AssertionError(f"unknown holdout kind: {case['kind']}")
