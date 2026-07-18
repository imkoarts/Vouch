from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.schemas.content import (
    EvaluationRelation,
    SemanticCoverageResolution,
    SemanticInspection,
    SemanticProviderProposal,
)
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator
from scripts.run_external_semantic_holdout_v2 import HoldoutDocument, main, run_holdout


def _possible_evaluation_proposal(source: str, reply: str) -> SemanticProviderProposal:
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    event_id = next(
        event.event_id for event in local.deterministic.source_events if event.event_type == "death"
    )
    resolutions = tuple(
        SemanticCoverageResolution(
            item_id=item.item_id,
            side="reply",
            clause_id=item.clause_id,
            clause_span=item.clause_span,
            unresolved_span=item.unresolved_span,
            category="possible_evaluation",
            disposition="typed_relation",
            evidence_spans=(reply,),
        )
        for coverage in local.deterministic.reply_coverage
        for item in coverage.unresolved_items
        if item.category == "possible_evaluation"
    )
    return SemanticProviderProposal(
        inspection=SemanticInspection(
            evaluations=(
                EvaluationRelation(
                    target_event_id=event_id,
                    evaluation_type="reward",
                    relation_subtype="reward",
                    polarity="positive",
                    literal_reading_plausible=False,
                    confidence="high",
                    evidence_spans=(reply,),
                ),
            ),
        ),
        coverage_resolutions=resolutions,
    )


def test_v2_runner_exposes_deterministic_escalation_and_offline_uncertainty() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "2.0",
            "holdout_id": "offline-audit",
            "cases": [
                {
                    "id": "offline-uncertain",
                    "source": source,
                    "reply": reply,
                    "mode": "offline_unresolved",
                    "expect": {
                        "escalation_required": True,
                        "provider_used": False,
                        "semantic_candidate_eligible": False,
                        "required_material_categories": ["possible_evaluation"],
                        "required_canonical_candidate_issue_codes": [
                            "HUMOR_INTENT_UNCERTAIN",
                            "HUMOR_ON_TRAGEDY",
                        ],
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))

    assert summary["all_passed"] is True
    observed = summary["results"][0]["observed"]
    assert "deterministic_inspection" in observed
    assert "escalation_decision" in observed
    assert "canonical_inspection" in observed
    assert observed["provider_request_count"] == 0


def test_v2_runner_uses_auditor_provider_proposal_through_canonical_coordinator() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    proposal = _possible_evaluation_proposal(source, reply)
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "2.0",
            "holdout_id": "provider-audit",
            "cases": [
                {
                    "id": "provider-irony",
                    "source": source,
                    "reply": reply,
                    "mode": "provider_assisted",
                    "provider_proposal": proposal.model_dump(mode="json"),
                    "expect": {
                        "provider_used": True,
                        "provider_validation_error_count": 0,
                        "implicit_sarcasm": True,
                        "humor_safety_required": True,
                        "semantic_candidate_eligible": False,
                        "required_canonical_candidate_issue_codes": ["HUMOR_ON_TRAGEDY"],
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))

    assert summary["all_passed"] is True
    assert summary["results"][0]["observed"]["provider_request_count"] == 1


def test_v2_runner_surfaces_provider_validation_failure_without_erasing_uncertainty() -> None:
    source = "A clinic reported eight deaths after the blackout."
    reply = "a ceremonial sash for the grid"
    proposal = _possible_evaluation_proposal(source, reply)
    invalid = proposal.model_copy(
        update={
            "inspection": proposal.inspection.model_copy(
                update={
                    "evaluations": (
                        proposal.inspection.evaluations[0].model_copy(
                            update={"evidence_spans": ("altered ceremonial quote",)}
                        ),
                    )
                }
            )
        }
    )
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "2.0",
            "holdout_id": "invalid-provider-audit",
            "cases": [
                {
                    "id": "invalid-span",
                    "source": source,
                    "reply": reply,
                    "mode": "provider_assisted",
                    "provider_proposal": invalid.model_dump(mode="json"),
                    "expect": {
                        "provider_used": True,
                        "escalation_required": True,
                        "semantic_candidate_eligible": False,
                        "required_material_categories": ["possible_evaluation"],
                        "required_provider_validation_errors": [
                            "evaluation span is not exact: 'altered ceremonial quote'"
                        ],
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))

    assert summary["all_passed"] is True
    observed = summary["results"][0]["observed"]
    assert "possible_evaluation" in observed["reply_unresolved_categories"]


def test_v2_runner_cli_writes_versioned_schema_and_result(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "schema_version": "2.0",
        "holdout_id": "cli-audit",
        "cases": [
            {
                "id": "deterministic-positive",
                "source": "The clinic restored power and patients recovered safely.",
                "mode": "deterministic_only",
                "expect": {"literal_harm": False, "escalation_required": False},
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
            "run_external_semantic_holdout_v2.py",
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
    assert schema["properties"]["schema_version"]["const"] == "2.0"


def test_v2_runner_exposes_result_state_roles_and_exact_unresolved_span() -> None:
    source = "The collision left two passengers glassy-eyed and barely rousable."
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "2.0",
            "holdout_id": "result-state-audit",
            "cases": [
                {
                    "id": "offline-human-result-state",
                    "source": source,
                    "mode": "offline_unresolved",
                    "expect": {
                        "escalation_required": True,
                        "provider_used": False,
                        "required_material_categories": ["possible_harm"],
                        "required_source_unresolved_categories": ["possible_harm"],
                        "required_source_unresolved_spans": ["glassy-eyed and barely rousable"],
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))

    assert summary["all_passed"] is True
    observed = summary["results"][0]["observed"]
    assert observed["source_unresolved_spans"] == ["glassy-eyed and barely rousable"]
    assert observed["canonical_inspection"]["source_coverage_complete"] is False
