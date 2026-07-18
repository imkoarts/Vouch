from __future__ import annotations

import json
from pathlib import Path

from app.services.benchmark import scenarios, write_benchmark_manifest


def test_public_benchmark_reports_six_templates_and_sixty_deterministic_entries(
    tmp_path: Path,
) -> None:
    suite = scenarios()
    assert len(suite) == 60
    assert len({item.id for item in suite}) == 60
    assert {item.category for item in suite} == {
        "unsupported_claim",
        "attribution_integrity",
        "inappropriate_humor",
        "missing_caveat",
        "false_rejection",
        "approval_integrity",
    }

    report = write_benchmark_manifest(tmp_path, openai_key_available=False)
    assert report["unique_prompt_count"] == 6
    assert report["independent_manual_validation"] is False
    assert report["evaluation_status"] == "live_comparative_evaluation_pending"
    assert report["direct_gpt_5_6"]["status"] == "not_measured"
    assert report["vouch_pipeline"]["status"] == "not_measured"
    assert len(json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))) == 60
