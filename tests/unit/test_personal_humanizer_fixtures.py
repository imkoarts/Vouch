from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.enums import GenerationMode
from app.schemas.content import EvidenceItem, EvidencePacket, GenerationVariant
from app.services.editorial_quality import assess_variant
from app.services.semantic_review import review_structure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "docs" / "personal-humanizer" / "evals"


def _fixtures() -> list[object]:
    result = []
    for path in sorted(EVAL_ROOT.glob("*.json")):
        for fixture in json.loads(path.read_text(encoding="utf-8")):
            result.append(pytest.param(path.name, fixture, id=f"{path.stem}:{fixture['id']}"))
    return result


@pytest.mark.parametrize(("filename", "fixture"), _fixtures())
def test_packaged_humanizer_fixture_executes(filename: str, fixture: dict[str, object]) -> None:
    expected = set(fixture.get("expected_issues", []))
    if filename == "micro-reply-fixtures.json":
        packet = EvidencePacket(
            mode=GenerationMode.REPLY,
            topic="micro reply fixture",
            items=(
                EvidenceItem(
                    evidence_id="source",
                    source_type="x_post",
                    text=str(fixture["source"]),
                    verification_status="direct_source",
                ),
            ),
            language="en",
            requested_format="reply",
        )
        variant = GenerationVariant(
            label=str(fixture["id"]),
            text=str(fixture["draft"]),
            similarity_risk=0.0,
            angle_id="reply-fixture",
            angle_type=str(fixture.get("angle_type", "plain_update")),
            evidence_ids=("source",),
            confidence="medium",
            factual_claims=(),
            attributed_claims=(),
            uncertainty_markers=(),
        )
        report = assess_variant(
            variant,
            evidence_packet=packet,
            recent_posts=tuple(str(item) for item in fixture.get("recent_posts", [])),
            minimum_specificity=0.0,
            minimum_evidence=0.0,
            minimum_naturalness=0.0,
        )
        actual = {issue.code for issue in report.issues}
        assert expected <= actual
        if fixture.get("expected_pass") is False:
            assert not report.passed
        return

    actual = set(review_structure(str(fixture["draft"]), mode=str(fixture.get("mode", ""))))
    assert expected <= actual
