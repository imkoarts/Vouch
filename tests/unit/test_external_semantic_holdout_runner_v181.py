from __future__ import annotations

import json
from pathlib import Path

from scripts.run_external_semantic_holdout import HoldoutDocument, main, run_holdout


def test_external_holdout_runner_passes_semantic_expectations() -> None:
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "1.0",
            "holdout_id": "auditor-example",
            "cases": [
                {
                    "id": "human-harm-irony",
                    "source": "Two patients died after the outage.",
                    "reply": "full marks for the outage",
                    "expect": {
                        "harm_referent": "human",
                        "literal_harm": True,
                        "implicit_sarcasm": True,
                        "suitable_for_humor": False,
                        "semantic_candidate_eligible": False,
                        "required_issue_codes": ["HUMOR_ON_TRAGEDY"],
                        "required_source_event_types": ["death"],
                        "required_source_evidence_spans": ["died"],
                    },
                }
            ],
        }
    )

    summary = run_holdout(document)

    assert summary["all_passed"] is True
    assert summary["passed"] == 1


def test_external_holdout_runner_reports_mismatch_without_mutating_expectation() -> None:
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "1.0",
            "holdout_id": "auditor-mismatch",
            "cases": [
                {
                    "id": "technical-control",
                    "source": "The deployment pipeline died overnight.",
                    "expect": {"harm_referent": "human"},
                }
            ],
        }
    )

    summary = run_holdout(document)

    assert summary["all_passed"] is False
    result = summary["results"][0]
    assert result["failures"] == ["harm_referent: expected 'human', observed 'non_human'"]


def test_external_holdout_runner_cli_writes_schema_and_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {
        "schema_version": "1.0",
        "holdout_id": "auditor-cli",
        "cases": [
            {
                "id": "definition-control",
                "source": "",
                "reply": "a square is a polygon with four sides",
                "expect": {"shell_operator": "none"},
            }
        ],
    }
    input_path = tmp_path / "holdout.json"
    output_path = tmp_path / "result.json"
    schema_path = tmp_path / "schema.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_external_semantic_holdout.py",
            str(input_path),
            "--output",
            str(output_path),
            "--write-schema",
            str(schema_path),
        ],
    )

    assert main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["all_passed"] is True
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["title"] == "HoldoutDocument"
