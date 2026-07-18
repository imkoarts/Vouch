from __future__ import annotations

import json
from pathlib import Path

from app.schemas.content import GenerationVariant
from app.services.editorial_quality import assess_variant

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "generation_quality_golden.json"


def test_golden_fixture_catalog_has_fifty_public_safe_cases() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert len(fixtures) >= 50
    assert len({fixture["id"] for fixture in fixtures}) == len(fixtures)
    assert all(fixture["mode"] for fixture in fixtures)
    assert all("known_bad_output" in fixture for fixture in fixtures)


def test_known_placeholder_regressions_fail_deterministic_gate() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    high_risk = [
        fixture
        for fixture in fixtures
        if any(
            marker in fixture["known_bad_output"].casefold()
            for marker in (
                "details",
                "verification",
                "community is watching",
                "real test",
                "significant implications",
                "execution will be key",
                "one report says",
                "if that account is accurate",
            )
        )
    ]

    for index, fixture in enumerate(high_risk):
        report = assess_variant(
            GenerationVariant(
                label=fixture["id"],
                text=fixture["known_bad_output"],
                similarity_risk=0.0,
                angle_id=f"angle-{index}",
                angle_type="plain_update",
                evidence_ids=("evidence-1",),
                confidence="low",
                factual_claims=(),
                attributed_claims=(),
                uncertainty_markers=(),
            )
        )
        assert not report.passed, fixture["id"]
