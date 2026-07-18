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

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "semantic_composition_visible_v181.json"
_CASES = json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_visible_semantic_composition_fixture(case: dict[str, object]) -> None:
    source = str(case["source"])
    reply = str(case["reply"])
    expected = dict(case["expect"])
    inspection = extract_semantic_inspection(source, reply)
    harm = adjudicate_human_harm(inspection)
    irony = adjudicate_evaluative_irony(inspection)
    shell = adjudicate_reply_shell(inspection)

    observed: dict[str, object] = {
        "literal_harm": harm.literal_harm,
        "harm_state": harm.harm_state,
        "harm_referent": harm.harm_referent,
        "implicit_sarcasm": irony.implicit_sarcasm,
        "shell_operator": shell.operator,
    }
    for key, value in expected.items():
        assert observed[key] == value
