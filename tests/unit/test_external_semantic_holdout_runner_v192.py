from __future__ import annotations

import asyncio

from app.schemas.content import SemanticCoverageResolution, SemanticProviderProposal
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator
from scripts.run_external_semantic_holdout_v4 import (
    HoldoutDocument,
    migrate_legacy_document,
    run_holdout,
)


def test_v4_runner_exposes_request_local_coverage_items() -> None:
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "4.0",
            "holdout_id": "v4-local",
            "cases": [
                {
                    "id": "offline",
                    "source": "The incident left two workers too unsteady to walk.",
                    "reply": "excellent work from the incident",
                    "mode": "offline_unresolved",
                    "metadata_says_humor": True,
                    "reaction_type": "dry_reframe",
                    "expect": {
                        "escalation_required": True,
                        "semantic_candidate_eligible": False,
                        "required_material_categories": ["possible_harm"],
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))
    observed = summary["results"][0]["observed"]

    assert summary["all_passed"]
    assert observed["deterministic_inspection"]["source_unresolved_items"][0]["item_id"]
    assert observed["escalation_decision"]["unresolved_item_ids"]


def test_v4_runner_provider_case_uses_current_item_identity() -> None:
    source = "The incident left two workers too unsteady to walk."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = local.deterministic.source_coverage[0].unresolved_items[0]
    proposal = SemanticProviderProposal(
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side="source",
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category="possible_harm",
                disposition="harmless",
                reason="The auditor classifies this synthetic state as harmless.",
                evidence_spans=(item.unresolved_span,),
            ),
        )
    )
    document = HoldoutDocument.model_validate(
        {
            "schema_version": "4.0",
            "holdout_id": "v4-provider",
            "cases": [
                {
                    "id": "provider",
                    "source": source,
                    "mode": "provider_assisted",
                    "provider_proposal": proposal.model_dump(mode="json"),
                    "expect": {
                        "provider_used": True,
                        "provider_validation_error_count": 0,
                        "semantic_candidate_eligible": True,
                    },
                }
            ],
        }
    )

    summary = asyncio.run(run_holdout(document))

    assert summary["all_passed"]
    request = summary["results"][0]["observed"]["provider_requests"][0]
    assert request["unresolved_item_ids"] == [item.item_id]


def test_v2_migration_reports_changed_request_identity() -> None:
    source = "The incident left two workers too unsteady to walk."
    item = (
        SemanticExtractionCoordinator()
        .inspect_local(source)
        .deterministic.source_coverage[0]
        .unresolved_items[0]
    )
    raw = {
        "schema_version": "2.0",
        "holdout_id": "legacy",
        "cases": [
            {
                "id": "legacy",
                "source": source,
                "mode": "provider_assisted",
                "provider_proposal": {
                    "inspection": {},
                    "coverage_resolutions": [
                        {
                            "side": "source",
                            "clause_span": source,
                            "unresolved_span": item.unresolved_span,
                            "category": "unknown_predicate",
                            "disposition": "unresolved",
                            "reason": "legacy",
                            "evidence_spans": [item.unresolved_span],
                        }
                    ],
                },
                "expect": {},
            }
        ],
    }

    migrated, report = migrate_legacy_document(raw)
    resolution = migrated["cases"][0]["provider_proposal"]["coverage_resolutions"][0]

    assert resolution["category"] == item.category
    assert resolution["item_id"] == item.item_id
    assert any("unknown_predicate->possible_harm" in entry for entry in report)
