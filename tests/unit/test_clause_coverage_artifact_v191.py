from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.schemas.content import (
    SemanticCoverage,
    SemanticCoverageResolution,
    SemanticEntity,
    SemanticEvent,
    SemanticInspection,
    SemanticProviderProposal,
    SemanticUnresolvedItem,
)
from app.schemas.release_artifacts import ArtifactDigest, ModuleDigestPair, ReleaseArtifactSet
from app.services.semantic_adjudication import adjudicate_humor_intent
from app.services.semantic_extraction_coordinator import SemanticExtractionCoordinator
from app.services.semantic_provider_validation import validate_semantic_provider_proposal
from app.services.semantic_reconciliation import reconcile_semantic_provider_proposal
from scripts.build_artifact_set import ArtifactSetError, _semantic_digests, _wheel_metadata_version
from scripts.run_external_semantic_holdout_v3 import migrate_legacy_document


def _categories(result, *, source: bool = True) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {category for item in coverage for category in item.unresolved_categories}


def _spans(result, *, source: bool = True) -> set[str]:
    coverage = result.canonical.source_coverage if source else result.canonical.reply_coverage
    return {span for item in coverage for span in item.unresolved_spans}


@pytest.mark.parametrize(
    "separator",
    (", and ", "; ", ". "),
)
def test_clause_local_predicate_cannot_borrow_later_human_entity(separator: str) -> None:
    source = (
        f"The telemetry relay died{separator}the chemical spill left two operators "
        "disoriented and unable to stand."
    )
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert not any(event.event_type == "death" for event in result.canonical.source_events)
    assert "possible_harm" in _categories(result)
    assert any("unable to stand" in span for span in _spans(result))
    for event in result.canonical.source_events:
        if event.predicate == "died" and event.affected_entity is not None:
            assert event.affected_entity.entity_type not in {"person", "group_of_people"}
            assert event.clause_id == event.affected_entity.clause_id


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "The ammonia leak left two warehouse workers woozy and unable to stay upright.",
            "woozy and unable to stay upright",
        ),
        ("The chemical spill kept three nurses too dizzy to stand.", "too dizzy to stand"),
        (
            "After an equipment failure, one miner remained unable to recognize his coworkers.",
            "unable to recognize his coworkers",
        ),
        ("The incident left one technician needing help to breathe.", "needing help to breathe"),
        ("The collision left two passengers unable to grip with one hand.", "unable to grip"),
    ),
)
def test_structural_human_result_state_is_possible_harm_without_closed_event_or_state_list(
    source: str,
    expected: str,
) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert result.escalation.required
    assert "possible_harm" in _categories(result)
    assert any(expected in span or span in expected for span in _spans(result))
    assert not all(item.complete for item in result.canonical.source_coverage)


@pytest.mark.parametrize(
    "source",
    (
        "The rollout left two users unable to locate the export button.",
        "The policy left residents reliant on the evening bus.",
        "The update left the application barely responsive.",
        "The storm left two hikers waiting inside the station.",
        "The keynote left the audience speechless.",
        "The outage left the server dependent on a backup node.",
    ),
)
def test_result_state_negative_controls_do_not_become_human_harm(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result)


@pytest.mark.parametrize(
    "source",
    (
        "The leak did not leave any technician woozy or confused.",
        "The spill might leave workers unable to stand.",
        "If the fire leaves residents unable to focus, crews will move them.",
        "Responders feared the collision could leave a driver unable to answer.",
    ),
)
def test_unasserted_result_states_are_not_asserted_harm(source: str) -> None:
    result = SemanticExtractionCoordinator().inspect_local(source)

    assert "possible_harm" not in _categories(result)
    assert not any(
        event.explicit_outcome and event.assertion_state in {"asserted", "reported"}
        for event in result.canonical.source_events
    )


def test_fatal_source_plus_incomplete_reply_fails_closed_candidate_locally() -> None:
    source = "A clinic reported nine deaths after the blackout."
    reply = "a ceremonial token for the blackout"
    result = SemanticExtractionCoordinator().inspect_local(source, reply)
    intent = adjudicate_humor_intent(
        result.canonical,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )

    assert result.escalation.required
    assert not result.canonical.reply_coverage[0].complete
    assert "HUMOR_INTENT_UNCERTAIN" in intent.issue_codes
    assert intent.humor_safety_required


def test_validated_plain_function_prevents_fatal_source_fallback() -> None:
    people = SemanticEntity(
        clause_id="C1",
        text_span="people",
        entity_type="group_of_people",
        confidence="high",
    )
    inspection = SemanticInspection(
        source_entities=(people,),
        source_events=(
            SemanticEvent(
                clause_id="C1",
                event_id="E1",
                predicate="deaths",
                affected_entity=people,
                event_type="death",
                polarity="negative",
                explicit_outcome=True,
                assertion_state="reported",
                evidence_spans=("deaths",),
            ),
        ),
        reply_coverage=(
            SemanticCoverage(
                coverage_id="reply-C1",
                clause_id="C1",
                clause_span="the report is incomplete",
                unresolved_items=(
                    SemanticUnresolvedItem(
                        item_id="reply-C1-item",
                        side="reply",
                        clause_id="C1",
                        clause_span="the report is incomplete",
                        unresolved_span="incomplete",
                        category="unknown_predicate",
                    ),
                ),
                unresolved_spans=("incomplete",),
                unresolved_categories=("unknown_predicate",),
                complete=False,
            ),
        ),
        reply_communicative_function="plain_observation",
        reply_communicative_function_confidence="high",
        reply_communicative_evidence_spans=("the report is incomplete",),
    )
    intent = adjudicate_humor_intent(
        inspection,
        metadata_says_humor=False,
        reaction_type="none",
        template_humor=False,
    )

    assert "HUMOR_INTENT_UNCERTAIN" not in intent.issue_codes


def _first_item(source: str, reply: str = "") -> tuple[SemanticInspection, SemanticUnresolvedItem]:
    local = SemanticExtractionCoordinator().inspect_local(source, reply)
    item = next(
        item
        for coverage in (*local.deterministic.source_coverage, *local.deterministic.reply_coverage)
        for item in coverage.unresolved_items
    )
    return local.deterministic, item


def test_provider_resolution_requires_current_request_local_item_identity() -> None:
    deterministic, item = _first_item("The incident left two workers too unsteady to walk.")
    proposal = SemanticProviderProposal(
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id="stale-item",
                side=item.side,
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="harmless",
                reason="synthetic",
                evidence_spans=(item.unresolved_span,),
            ),
        )
    )
    errors = validate_semantic_provider_proposal(
        deterministic,
        proposal,
        source_clauses=(item.clause_span,),
        reply_clauses=(),
    )

    assert any("stale item id" in error for error in errors)


@pytest.mark.parametrize("field", ("category", "clause_id", "unresolved_span"))
def test_provider_resolution_rejects_changed_request_item_fields(field: str) -> None:
    deterministic, item = _first_item("The incident left two workers too unsteady to walk.")
    values = item.model_dump()
    if field == "category":
        values[field] = "unknown_predicate"
    elif field == "clause_id":
        values[field] = "C999"
    else:
        values[field] = "unsteady"
    proposal = SemanticProviderProposal(
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side=item.side,
                clause_id=values["clause_id"],
                clause_span=item.clause_span,
                unresolved_span=values["unresolved_span"],
                category=values["category"],
                disposition="harmless",
                reason="synthetic",
                evidence_spans=(item.unresolved_span,),
            ),
        )
    )
    errors = validate_semantic_provider_proposal(
        deterministic,
        proposal,
        source_clauses=(item.clause_span,),
        reply_clauses=(),
    )

    assert errors


def test_valid_item_id_resolution_removes_only_named_uncertainty() -> None:
    deterministic, item = _first_item("The incident left two workers too unsteady to walk.")
    proposal = SemanticProviderProposal(
        coverage_resolutions=(
            SemanticCoverageResolution(
                item_id=item.item_id,
                side=item.side,
                clause_id=item.clause_id,
                clause_span=item.clause_span,
                unresolved_span=item.unresolved_span,
                category=item.category,
                disposition="harmless",
                reason="The state is an ordinary non-medical control in this synthetic case.",
                evidence_spans=(item.unresolved_span,),
            ),
        )
    )
    assert (
        validate_semantic_provider_proposal(
            deterministic,
            proposal,
            source_clauses=(item.clause_span,),
            reply_clauses=(),
        )
        == ()
    )
    canonical = reconcile_semantic_provider_proposal(deterministic, proposal)

    assert canonical.source_coverage[0].complete
    assert canonical.source_coverage[0].unresolved_items == ()


def test_legacy_holdout_provider_resolution_is_migrated_explicitly() -> None:
    source = "The incident left two workers too unsteady to walk."
    local = SemanticExtractionCoordinator().inspect_local(source)
    item = local.deterministic.source_coverage[0].unresolved_items[0]
    raw = {
        "schema_version": "2.0",
        "holdout_id": "legacy",
        "cases": [
            {
                "id": "legacy-case",
                "source": source,
                "reply": "",
                "mode": "provider_assisted",
                "provider_proposal": {
                    "inspection": {},
                    "coverage_resolutions": [
                        {
                            "side": "source",
                            "clause_span": source,
                            "unresolved_span": item.unresolved_span,
                            "category": item.category,
                            "disposition": "harmless",
                            "reason": "synthetic",
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

    assert migrated["schema_version"] == "3.0"
    assert resolution["item_id"] == item.item_id
    assert resolution["clause_id"] == item.clause_id
    assert report


def test_semantic_module_digest_rejects_stale_wheel_module(tmp_path: Path) -> None:
    wheel = tmp_path / "fake.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for relative in (
            "app/schemas/content.py",
            "app/services/semantic_composition.py",
            "app/services/semantic_extraction.py",
            "app/services/semantic_extraction_coordinator.py",
            "app/services/semantic_provider_validation.py",
            "app/services/semantic_reconciliation.py",
            "app/services/semantic_adjudication.py",
        ):
            content = Path(relative).read_bytes()
            if relative.endswith("semantic_adjudication.py"):
                content += b"\n# stale\n"
            archive.writestr(relative, content)

    with pytest.raises(ArtifactSetError, match="module mismatch"):
        _semantic_digests(Path.cwd(), wheel)


def test_wheel_metadata_version_parser_rejects_missing_version(tmp_path: Path) -> None:
    wheel = tmp_path / "fake.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("x.dist-info/METADATA", "Metadata-Version: 2.4\n")
    with zipfile.ZipFile(wheel) as archive, pytest.raises(ArtifactSetError):
        _wheel_metadata_version(archive)


def test_artifact_manifest_schema_binds_source_and_wheel_hashes() -> None:
    digest = "a" * 64
    manifest = ReleaseArtifactSet(
        application_version="0.19.4",
        source_zip=ArtifactDigest(filename="source.zip", sha256=digest, size_bytes=1),
        wheel=ArtifactDigest(
            filename="x_content_bot-0.19.4-py3-none-any.whl", sha256=digest, size_bytes=1
        ),
        external_runner=ArtifactDigest(filename="runner.py", sha256=digest, size_bytes=1),
        external_schema=ArtifactDigest(filename="schema.json", sha256=digest, size_bytes=1),
        source_tree_digest=digest,
        semantic_module_digests={
            "app/services/semantic_adjudication.py": ModuleDigestPair(
                source_sha256=digest,
                wheel_sha256=digest,
            )
        },
        source_skill_hash=digest,
        wheel_skill_hash=digest,
        source_runtime_hash=digest,
        wheel_runtime_hash=digest,
        test_count=1,
        migration_head="0006_dashboard_chat_messages",
        release_file_count=1,
    )

    assert json.loads(manifest.model_dump_json())["application_version"] == "0.19.4"
